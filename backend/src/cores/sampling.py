import os
import logging
from shutil import rmtree

import ffmpeg
import numpy as np
import cv2
from tqdm import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # to disable the warning message
from transnetv2 import TransNetV2

from cores.models import VideoModel, ShotModel, FrameModel
from cores.utils import get_nvidia_decoder, get_video_codec

logger = logging.getLogger(__name__)

class FrameSampler:
    def __init__(self,
                 video_data: list[VideoModel],
                 output_dir: str,
                 batch_size=8,
                 num_workers=4,
                 use_gpu=True):
        self.video_data = video_data
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.model = TransNetV2()

    def detect_shots(self, video: VideoModel):
        video_path = video['path']

        # Predict shots
        preds, _ = self.predict_video(video_path)
        if preds is None:
            logger.warning(f"Skipping {video_path} due to decoding error.")
            return

        predicted_shots = self.model.predictions_to_scenes(preds)

        # Construct shots' info
        video_id = video['id'].split('_')[-1]
        shots: list[ShotModel] = [{
            'id': f'{video_id}_S{idx:05d}',
            'start': int(start),
            'end': int(end),
            'videoId': video_id
        } for idx, (start, end) in enumerate(predicted_shots)]

        return shots

    def extract_frames(self, video: VideoModel, shots: list[ShotModel]) -> list[FrameModel]:
        """Extract keyframes from detected shots."""
        frame_dir = os.path.join(self.output_dir, video['id'])
        try:
            os.mkdir(frame_dir)

        except Exception as e:
            logger.error(f"Failed to create directory {frame_dir}: {e}")
            return

        # Create extracting positions: start, 1/3, 2/3, and end indices
        shot_ids = []
        all_positions = []
        for shot in shots:
            start = shot['start']
            end = shot['end']

            shot_ids.append(shot['id'].split('_')[-1])
            all_positions.append([
                start,
                int(start + (end - start) * 1/3),
                int(start + (end - start) * 2/3),
                end
            ])

        # Flatten and sort unique positions for efficient sequential reading
        unique_positions = np.unique(all_positions)
        next_pos_idx = 0

        all_frame_info = []
        video_path = video['path']
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                logger.error(f'Failed to open video file {video_path}')
                return

            frame_idx = 0
            saved_frames = {}
            while next_pos_idx < len(unique_positions):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f'Failed to read frame at {frame_idx} in {video_path}')
                    frame_idx += 1
                    continue

                if frame_idx == unique_positions[next_pos_idx]:
                    saved_frames[frame_idx] = frame
                    next_pos_idx += 1

                frame_idx += 1

            # Save frames for each shot and position
            for shot_id, positions in zip(shot_ids, all_positions):
                shot_frames: list[FrameModel] = []
                for pos in positions:
                    frame = saved_frames.get(pos)
                    if frame is not None:
                        try:
                            frame_id = f"{shot_id:05}_F{pos:05}"
                            frame_path = os.path.join(frame_dir, f"{frame_id}.jpg")
                            cv2.imwrite(frame_path, frame)

                            shot_frames.append({
                                'id': frame_id,
                                'shotId': shot_id,
                                'selected': False,
                                'path': frame_path
                            })

                        except Exception as e:
                            logger.error(f"Failed to write frame at {pos} in {video_path}: {e}")
                    else:
                        logger.warning(f"Frame at {pos} not found in {video_path}")

                # Select the two between frames for embedding
                total_frames = len(shot_frames)
                shot_frames[(total_frames - 1) // 2]['selected'] = True
                shot_frames[(total_frames + 1) // 2]['selected'] = True

                all_frame_info.extend(shot_frames)

        finally:
            cap.release()
            return all_frame_info

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
                    logger.error(f"Unsupported codec: {codec_name}")
                    raise RuntimeError(f"Unsupported codec: {codec_name}")

                video_stream, _ = (
                    ffmpeg
                    .input(video_file, hwaccel='cuda', vcodec=nvidia_decoder)
                    .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
                    .run(capture_stdout=True, capture_stderr=True)
                )
            else:
                video_stream, _ = (
                    ffmpeg
                    .input(video_file, threads=self.num_workers)
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

    def run(self) -> tuple[list[ShotModel], list[FrameModel]]:
        """Detect shots from a video and extract keyframes."""
        if os.path.exists(self.output_dir):
            rmtree(self.output_dir)
        os.mkdir(self.output_dir)

        all_shot_info = []
        all_frame_info = []
        desc = 'Sampling keyframes'
        for video in tqdm(self.video_data, desc, len(self.video_data)):
            shots = self.detect_shots(video)
            if not shots:
                continue

            # Extract frames from each shot
            frames = self.extract_frames(video, shots)

            all_shot_info.extend(shots)
            all_frame_info.extend(frames)

        logger.info(f"Successfully extracted {len(all_frame_info)} frames from {len(all_shot_info)} shots")

        return all_shot_info, all_frame_info
