import os
from typing_extensions import Callable
from os.path import basename, dirname

import ffmpeg
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights)
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights)
from torchvision.models.video import (
    s3d, S3D_Weights)
import clip

class ImageDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 transforms: Callable,
                 permute_channels=True):
        self.image_paths = [os.path.join(dirpath, filename)
                            for dirpath, _, filenames in os.walk(image_dir)
                            for filename in filenames]
        self.transforms = transforms
        self.permute_channels = permute_channels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path)

        # Permute (H, W, C) to (C, H, W)
        if self.permute_channels:
            image = np.array(image)
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)

        image = self.transforms(image)
        return image, path

class VideoDataset(Dataset):
    def __init__(self,
                 video_infos: list[dict],
                 transforms: Callable,
                 num_frames=128):
        self.video_infos = video_infos
        self.transforms = transforms
        self.num_frames = num_frames
        self.min_num_frames = 14
        self.resize = (256, 256)

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx: int):
        shot_id, start, end, path = self.video_infos[idx].values()
        frames = self.sample_frames(path, start, end)
        frames = [torch.from_numpy(frame.copy()).permute(2, 0, 1)
                  for frame in frames]
        frames = torch.stack(frames)
        frames = self.transforms(frames)
        return frames, shot_id

    def sample_frames(self, video_path: str, start: int, end: int):
        """Sample uniformly a specified number of frames."""
        # Get fps of the video
        probe = ffmpeg.probe(video_path)
        video_info = next(stream for stream in probe['streams']
                          if stream['codec_type'] == 'video')
        framerate = eval(video_info['avg_frame_rate'])

        # Calculate segment info
        segment_frames = end - start + 1
        dur = segment_frames / framerate

        # Define new size
        size = f'{self.resize[0]}x{self.resize[1]}'

        # For segments with fewer frames than min_num_frames, duplicate actual frames
        if segment_frames < self.min_num_frames:
            # Extract all frames at original framerate
            video_stream, _ = (
                ffmpeg
                .input(video_path, ss=start/framerate, t=dur)
                .filter('scale', size)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, capture_stderr=True)
            )
            frames = np.frombuffer(video_stream, np.uint8).reshape((-1, *self.resize, 3))

            # Duplicate frames to reach at least self._min_num_frames
            if len(frames) < self.min_num_frames:
                # Repeat frames to reach minimum self._min_num_frames
                indices = np.linspace(0, len(frames) - 1, self.min_num_frames, dtype=int)
                frames = frames[indices]

            # Further duplicate to reach num_frames
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            return frames[indices]

        # For segments with enough frames, use FFmpeg's fps filter
        else:
            fps = self.num_frames / dur
            video_stream, _ = (
                ffmpeg
                .input(video_path, ss=start/framerate, t=dur)
                .filter('fps', fps)
                .filter('scale', size)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, capture_stderr=True)
            )
            frames = np.frombuffer(video_stream, np.uint8).reshape((-1, *self.resize, 3))

            # Ensure we have exactly num_frames
            if len(frames) != self.num_frames:
                indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
                frames = frames[indices]

            return frames

class FeatureExtractor:
    def __init__(self,
                 frame_dir: str,
                 shot_infos: list[dict],
                 clip_model='ViT-L/14',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.frame_dir = frame_dir
        self.shot_infos = shot_infos
        self.clip_model = clip_model
        self.device = device

    @torch.no_grad()
    def extract_spatial_features(self,
                                 batch_size=16,
                                 num_workers=4) -> tuple[list[np.ndarray], list[str]]:
        """Extract the general appearance/spatial features."""
        model, transforms = clip.load(self.clip_model, self.device)
        model.eval()

        dataset = ImageDataset(self.frame_dir, transforms, False)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        all_features = []
        all_paths = []
        progress_bar = tqdm(desc='Extracting spatial features', total=len(dataset))
        for batch, paths in loader:
            batch = batch.to(self.device)

            features = model.encode_image(batch)
            features /= features.norm(dim=-1, keepdim=True)

            all_features.extend(features.cpu().numpy())
            all_paths.extend(paths)

            progress_bar.update(batch.size(0))

        progress_bar.close()
        return all_features, all_paths

    @torch.no_grad()
    def extract_image_concept(self,
                              batch_size=8,
                              num_workers=4) -> tuple[list[str], list[str]]:
        """Extract the global class label/concept."""
        weights = EfficientNet_B3_Weights.DEFAULT
        model = efficientnet_b3(weights=weights)
        model.eval()
        model.to(self.device)
        transforms = weights.transforms()

        dataset = ImageDataset(self.frame_dir, transforms)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        labels = []
        all_paths = []
        progress_bar = tqdm(desc='Extracting image concept', total=len(dataset))
        for batch, paths in loader:
            batch = batch.to(self.device)

            logits = model(batch)
            all_predicted_idx = logits.argmax(dim=-1)

            labels.extend([weights.meta['categories'][idx]
                        for idx in all_predicted_idx])
            all_paths.extend(paths)

            progress_bar.update(batch.size(0))

        progress_bar.close()
        return labels, all_paths

    # Use batch size of 1 due to the inconsistant of video resolutions
    @torch.no_grad()
    def extract_objects_and_locations(self,
                                      batch_size=1,
                                      num_workers=4) -> tuple[list[dict], list[str]]:
        """Extract objects and their locations."""
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        model.eval()
        model.to(self.device)
        transforms = weights.transforms()

        dataset = ImageDataset(self.frame_dir, transforms)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        all_object_infos = []
        all_paths = []
        progress_bar = tqdm(desc='Extracting object and location', total=len(dataset))
        for batch, paths in loader:
            batch = batch.to(self.device)

            all_predictions = model(batch)

            for preds in all_predictions:
                object_list = []
                for box, label, score in zip(*preds.values()):
                    x1, y1, x2, y2 = box.tolist()
                    object_list.append({
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'objectName': weights.meta['categories'][label],
                        'score': score.item()
                    })

                all_object_infos.append(object_list)
                all_paths.extend(paths)

            progress_bar.update(batch.size(0))

        progress_bar.close()
        return all_object_infos, all_paths

    @torch.no_grad()
    def extract_shot_action(self,
                            batch_size=8,
                            num_workers=4) -> tuple[list[str], list[str]]:
        """Extract the action in the shots."""
        weights = S3D_Weights.DEFAULT
        model = s3d(weights=weights)
        model.eval()
        model.to(self.device)
        transforms = weights.transforms()

        dataset = VideoDataset(self.shot_infos, transforms)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        labels = []
        all_ids = []
        progress_bar = tqdm(desc='Extracting image action', total=len(dataset))
        for batch, shot_ids in loader:
            batch = batch.to(self.device)

            logits = model(batch)

            all_predicted_idx = logits.argmax(dim=-1)
            labels.extend([weights.meta['categories'][idx]
                        for idx in all_predicted_idx])
            all_ids.extend(shot_ids)

            progress_bar.update(batch.size(0))

        progress_bar.close()
        return labels, all_ids

    def run(self):
        all_features, paths = self.extract_spatial_features()
        metadata = pd.DataFrame({'path': paths, 'features': all_features})

        labels, paths = self.extract_image_concept()
        df = pd.DataFrame({'path': paths, 'concept': labels})
        metadata = pd.merge(metadata, df, 'inner', 'path')

        all_object_infos, paths = self.extract_objects_and_locations()
        df = pd.DataFrame({'path': paths, 'obj_loc': all_object_infos})
        metadata = pd.merge(metadata, df, 'inner', 'path')

        labels, ids = self.extract_shot_action()
        shot_actions = pd.DataFrame({'shot_id': ids, 'action': labels})

        metadata['shot_id'] = metadata['path'].apply(
            lambda path: f"{basename(dirname(path))}_{basename(path).split('_')[0]}"
        )
        metadata = pd.merge(metadata, shot_actions, 'inner', 'shot_id')

        all_features = np.stack(metadata.pop('features').to_numpy())
        return all_features, metadata
