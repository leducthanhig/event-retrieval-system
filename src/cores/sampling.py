import os
from shutil import rmtree
from concurrent.futures import ThreadPoolExecutor

import ffmpeg
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

from transnetv2 import TransNetV2

class FrameSampler:
    def __init__(self,
                 video_dir: str,
                 output_dir: str,
                 batch_size=8,
                 num_workers=1):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = TransNetV2()

    def extract_shot_frames(self, video_path: str, shots: np.ndarray):
        """Extract keyframes from detected shots."""
        file_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = os.path.join(self.output_dir, file_name)
        os.mkdir(frame_dir)

        # Create extracting positions: the start, end and 3 quantiles
        all_positions = np.array([
            [
                start,
                start + (end - start) * 0.25,
                start + (end - start) * 0.5,
                start + (end - start) * 0.75,
                end
            ] for start, end in shots], np.int32)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f'Failed to open video file {video_path}')

        # Iterate through each shot and extract specified frames
        for shot, positions in enumerate(all_positions):
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if not ret:
                    print(f'Failed to extract frame at {pos}')
                    continue
                # Save to disk for later uses
                cv2.imwrite(
                    os.path.join(frame_dir, f'S{shot:05}_F{pos:05}.jpg'), frame)

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
        video_stream, _ = (
            ffmpeg
            .input(video_file)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
            .run(capture_stdout=True, capture_stderr=True)
        )

        frames = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        return self.predict_frames(frames)

    def process_video(self, video_file: str):
        """Process the given video file."""
        # Predict shots
        preds, _ = self.predict_video(video_file)
        shots = self.model.predictions_to_scenes(preds)

        # Extract frames from each shot
        self.extract_shot_frames(video_file, shots)

        # Return shot info
        file_name = os.path.splitext(os.path.basename(video_file))[0]
        return [{
            'id': f'{file_name}_S{idx:05d}',
            'start': int(start),
            'end': int(end),
            'path': video_file
        } for idx, (start, end) in enumerate(shots)]

    def run(self):
        """Detect shots from a video and extract keyframes. Return the shots' info."""
        if os.path.exists(self.output_dir):
            rmtree(self.output_dir)
        os.mkdir(self.output_dir)

        files = [os.path.join(dirpath, filename)
                 for dirpath, _, filenames in os.walk(self.video_dir)
                 for filename in filenames]
        shot_infos = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            desc = 'Sampling keyframes'
            results = tqdm(executor.map(self.process_video, files), desc, len(files))
            for result in results:
                shot_infos.extend(result)

        return pd.DataFrame(shot_infos)
