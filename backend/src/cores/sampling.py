import os
import logging
import json
from shutil import rmtree

import ffmpeg
import numpy as np
import cv2
from tqdm import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # to disable the warning message
from transnetv2 import TransNetV2

from utils import get_nvidia_decoder, get_video_codec

logger = logging.getLogger(__name__)

class FrameSampler:
    def __init__(self,
                 video_root_dir: str,
                 output_root_dir: str,
                 video_metadata_save_path: str,
                 batch_size=8,
                 use_gpu=True):
        self.video_root_dir = video_root_dir
        self.output_root_dir = output_root_dir
        self.video_metadata_save_path = video_metadata_save_path
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.video_paths = [os.path.join(dirpath, filename)
                            for dirpath, _, filenames in os.walk(video_root_dir)
                            for filename in filenames]

        self.model = TransNetV2()

        self.metadata = dict()
        for path in self.video_paths:
            video_id = os.path.splitext(os.path.basename(path))[0]
            self.metadata[video_id] = dict(path=path)

    def extract_frames(self, video_path: str, shots: np.ndarray):
        """Extract keyframes from detected shots."""
        video_relpath = os.path.relpath(video_path, self.video_root_dir)
        video_relpath_splitext = os.path.splitext(video_relpath)[0]
        shot_root_dir = os.path.join(self.output_root_dir, video_relpath_splitext)
        os.makedirs(shot_root_dir)

        # Create extracting positions: start, 1/3, 2/3, and end indices
        all_positions = [
            [
                start,
                int(start + (end - start) * 1/3),
                int(start + (end - start) * 2/3),
                end
            ] for start, end in shots]

        # Save frames for each shot and position
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                logger.error(f'Failed to open video file {video_path}')
                return

            total_frames = 0
            frame_idx = 0
            for shot_idx, positions in enumerate(all_positions):
                shot_id = f"S{shot_idx:05}"
                frame_dir = os.path.join(shot_root_dir, shot_id)
                os.mkdir(frame_dir)

                next_pos_idx = 0
                while next_pos_idx < len(positions):
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f'Failed to read frame at {frame_idx} in {video_path}')
                        frame_idx += 1
                        continue

                    if frame_idx == positions[next_pos_idx]:
                        try:
                            frame_id = f"F{frame_idx:06}"
                            frame_path = os.path.join(frame_dir, f"{frame_id}.jpg")
                            cv2.imwrite(frame_path, frame)

                        except Exception as e:
                            logger.error(f"Failed to write frame at {frame_idx} in {video_path}: {e}")

                        # Skip duplicated positions
                        while next_pos_idx < len(positions) and frame_idx == positions[next_pos_idx]:
                            next_pos_idx += 1

                    frame_idx += 1

                # Mark the two between frames as selected
                extracted_frames = [os.path.join(frame_dir, file)
                                    for file in os.listdir(frame_dir)]
                extracted_frames.sort()
                num_frames = len(extracted_frames)

                old_path = extracted_frames[(num_frames - 1) // 2]
                old_name, ext = os.path.splitext(old_path)
                new_path = old_name + '_selected' + ext
                os.rename(old_path, new_path)

                old_path = extracted_frames[(num_frames + 1) // 2]
                old_name, ext = os.path.splitext(old_path)
                new_path = old_name + '_selected' + ext
                os.rename(old_path, new_path)

                total_frames += num_frames

            return total_frames

        finally:
            cap.release()

    # Modified version of https://github.com/soCzech/TransNetV2/blob/master/inference/transnetv2.py
    def predict_frames(self, frames: np.ndarray):
        """Make predictions for the given frames."""
        # Return [batch_size, 100, 27, 48, 3] instead of [1, 100, 27, 48, 3] for faster inference
        def input_iterator():
            """
            Return windows of size 100 where the first/last 25 frames are from the previous/next batch
            the first and last window must be padded by copies of the first and last frame of the video.
            """
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            windows = []
            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                windows.append(out[np.newaxis])
                ptr += 50
                if len(windows) == self.batch_size:
                    yield np.concatenate(windows)
                    windows = []

            if windows:
                yield np.concatenate(windows)

        predictions = []
        for inp_batch in input_iterator():
            single_frame_pred, all_frames_pred = self.model.predict_raw(inp_batch)
            predictions.extend([(single_frame_pred.numpy()[i, 25:75, 0],
                                 all_frames_pred.numpy()[i, 25:75, 0])
                                 for i in range(inp_batch.shape[0])])

        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames

    def predict_video(self, video_file: str):
        """Make predictions for the given video."""
        try:
            if self.use_gpu:
                codec_name = get_video_codec(video_file)
                nvidia_decoder = get_nvidia_decoder(codec_name)
                if nvidia_decoder is None:
                    msg = f"Unsupported codec: {codec_name}"
                    logger.error(msg)
                    raise RuntimeError(msg)

                video_stream, _ = (
                    ffmpeg
                    .input(video_file, hwaccel='cuda', vcodec=nvidia_decoder)
                    .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
                    .run(capture_stdout=True, capture_stderr=True)
                )
            else:
                video_stream, _ = (
                    ffmpeg
                    .input(video_file)
                    .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
                    .run(capture_stdout=True, capture_stderr=True)
                )
            frames = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])

        except ffmpeg.Error as e:
            err = e.stderr.decode() if hasattr(e, 'stderr') else e
            logger.error(f"ffmpeg decoding failed for {video_file}: {err}")
            return None, None

        except Exception as e:
            logger.error(f"Error decoding video {video_file}: {e}")
            return None, None

        return self.predict_frames(frames)

    def save_metadata(self):
        with open(self.video_metadata_save_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def run(self):
        """Detect shots from a video and extract keyframes."""
        if os.path.exists(self.output_root_dir):
            rmtree(self.output_root_dir)
        os.makedirs(self.output_root_dir)

        desc = 'Sampling keyframes'
        total_shots = 0
        total_frames = 0
        for video_path in tqdm(self.video_paths, desc, len(self.video_paths)):
            # Detect shots
            preds, _ = self.predict_video(video_path)
            if preds is None:
                logger.warning(f"Skipping {video_path} due to decoding error.")
                return

            shots = self.model.predictions_to_scenes(preds)
            total_shots += len(shots)

            # Save start timestamps of each shot to metadata
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            self.metadata[video_id]['shots'] = [shot[0] for shot in shots]

            # Extract frames from each shot
            num_frames = self.extract_frames(video_path, shots)
            if num_frames:
                total_frames += num_frames

        self.save_metadata()

        logger.info(f"Successfully extracted {total_frames} frames from {total_shots} shots")
