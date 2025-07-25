import os
from shutil import rmtree
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # to disable the warning message
from transnetv2 import TransNetV2

class FrameSampler:
    def __init__(self, video_dir: str, output_dir: str, num_workers=1):
        self.video_dir = video_dir
        self.output_dir = output_dir
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

    def process_video(self, video_file: str):
        file_path = os.path.join(self.video_dir, video_file)

        # Predict shots
        _, preds, _ = model.predict_video(file_path)
        shots = model.predictions_to_scenes(preds)

        # Extract frames from each shot
        self.extract_shot_frames(file_path, shots)

        # Return shot info
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        return [{
            'id': f'{file_name}_S{idx:05d}',
            'start': int(start),
            'end': int(end),
            'path': file_path
        } for idx, (start, end) in enumerate(shots)]

    def run(self):
        """
        Detect shots from a video and extract keyframes.
        Return the shots' info.
        """
        if os.path.exists(self.output_dir):
            rmtree(self.output_dir)
        os.mkdir(self.output_dir)

        files = os.listdir(self.video_dir)
        shot_infos = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            desc = 'Sampling keyframes'
            results = tqdm(executor.map(self.process_video, files), desc, len(files))
            for result in results:
                shot_infos.extend(result)

        return pd.DataFrame(shot_infos)
