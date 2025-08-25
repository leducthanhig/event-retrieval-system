import os
import logging
import json
from shutil import rmtree

import ffmpeg
import numpy as np
from tqdm import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # to disable the warning message
from transnetv2 import TransNetV2

from utils import get_decoder

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
        os.makedirs(shot_root_dir, exist_ok=True)

        # Calculate all frame positions to extract
        all_positions = []
        position_info = {}  # Maps position to (shot_idx, position_type)

        for shot_idx, (start, end) in enumerate(shots):
            positions = [
                start,
                int(start + (end - start) * 1/3),
                int(start + (end - start) * 2/3),
                end
            ]

            # Remove duplicates within a shot
            unique_positions = sorted(set(positions))

            for pos_idx, pos in enumerate(unique_positions):
                all_positions.append(pos)

                # Track which shot this frame belongs to and its type
                position_type = None

                # Handle short shots with fewer than 3 unique positions
                if len(unique_positions) == 2:
                    # Mark the first frame as "selected_1" and the second as "selected_2"
                    if pos_idx == 0:
                        position_type = "selected_1"
                    elif pos_idx == 1:
                        position_type = "selected_2"
                elif len(unique_positions) > 2:
                    # For longer shots, mark 1/3 and 2/3 positions as selected
                    if pos == positions[1]:
                        position_type = "selected_1"
                    elif pos == positions[2]:
                        position_type = "selected_2"

                position_info[pos] = (shot_idx, position_type)

        # Sort positions for easier processing
        all_positions = sorted(set(all_positions))

        # Create output temp directory
        temp_dir = os.path.join(self.output_root_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Build select filter for all frames at once
            select_conditions = '+'.join([f"eq(n,{pos})" for pos in all_positions])

            # Extract all frames in one operation
            decoder = get_decoder(video_path, self.use_gpu)
            configs = {'vcodec': decoder}
            if self.use_gpu and decoder.endswith('cuvid'):
                configs['hwaccel'] = 'cuda'
            (
                ffmpeg
                .input(video_path, **configs)
                .filter('select', select_conditions)
                .output(os.path.join(temp_dir, 'frame_%06d.jpg'), vsync=0)
                .run(quiet=True)
            )

            # Organize extracted frames into shot directories
            total_frames = 0
            frame_files = sorted(os.listdir(temp_dir))

            for i, frame_file in enumerate(frame_files):
                if i >= len(all_positions):
                    break

                pos = all_positions[i]
                shot_idx, position_type = position_info[pos]

                # Create shot directory
                shot_id = f"S{shot_idx:05}"
                frame_dir = os.path.join(shot_root_dir, shot_id)
                os.makedirs(frame_dir, exist_ok=True)

                # Rename and move file
                frame_id = f"F{pos:06}"
                src_path = os.path.join(temp_dir, frame_file)

                if position_type:
                    dst_path = os.path.join(frame_dir, f"{frame_id}_selected.jpg")
                else:
                    dst_path = os.path.join(frame_dir, f"{frame_id}.jpg")

                os.rename(src_path, dst_path)
                total_frames += 1

            return total_frames

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            return 0

        finally:
            # Clean up temp directory
            rmtree(temp_dir, ignore_errors=True)

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
            decoder = get_decoder(video_file, self.use_gpu)
            configs = {'vcodec': decoder}
            if self.use_gpu and decoder.endswith('cuvid'):
                configs['hwaccel'] = 'cuda'

            video_stream, err = (
                ffmpeg
                .input(video_file, **configs)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='48x27')
                .run(capture_stdout=True, capture_stderr=True)
            )

            frames = np.frombuffer(video_stream, np.uint8).reshape((-1, 27, 48, 3))

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
            self.metadata[video_id]['shots'] = [shot[0] for shot in shots.tolist()]

            # Extract frames from each shot
            num_frames = self.extract_frames(video_path, shots)
            if num_frames:
                total_frames += num_frames

        self.save_metadata()

        logger.info(f"Successfully extracted {total_frames} frames from {total_shots} shots")
