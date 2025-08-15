import os
import logging
from typing import Callable

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from open_clip import create_model_and_transforms

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # to disable the warning message
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    def __init__(self,
                 image_paths: list[str],
                 transforms: Callable | None = None):
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path)

        if self.transforms:
            image = self.transforms(image)
        else:
            image = T.ToTensor()(image)

        return image, path

class FeatureExtractor:
    def __init__(self,
                 image_paths: list[str],
                 clip_model_name: str,
                 clip_pretrained: str,
                 dino_model_name: str,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.image_paths = image_paths
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.dino_model_name = dino_model_name
        self.device = device

    @torch.no_grad()
    def extract_spatial_features(self,
                                 batch_size=16,
                                 num_workers=4) -> tuple[list[np.ndarray], list[str]]:
        """Extract the general appearance/spatial features."""
        model, _, transforms = create_model_and_transforms(self.clip_model_name,
                                                           self.clip_pretrained,
                                                           device=self.device)
        model.eval()
        dataset = ImageDataset(self.image_paths, transforms)
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

        logger.info(f"Successfully extracted {len(all_features)} spatial feature vectors")

        return all_features, all_paths

    @torch.no_grad()
    def extract_deep_visual_features(self,
                                     batch_size=16,
                                     num_workers=4) -> tuple[list[np.ndarray], list[str]]:
        """Extract the deep visual features."""
        model = AutoModel.from_pretrained(self.dino_model_name, device_map=self.device)

        processor = AutoImageProcessor.from_pretrained(self.dino_model_name,
                                                       use_fast=True,
                                                       return_tensors='pt')
        dataset = ImageDataset(self.image_paths, processor)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=FeatureExtractor._collate_fn)

        all_features = []
        all_paths = []
        progress_bar = tqdm(desc='Extracting deep visual features', total=len(dataset))
        for batch, paths in loader:
            batch = batch.to(self.device)

            features = model(pixel_values=batch).pooler_output
            features /= features.norm(dim=-1, keepdim=True)

            all_features.extend(features.cpu().numpy())
            all_paths.extend(paths)

            progress_bar.update(batch.size(0))

        progress_bar.close()

        logger.info(f"Successfully extracted {len(all_features)} deep visual feature vectors")

        return all_features, all_paths

    @staticmethod
    def _collate_fn(batch: tuple[dict[str, torch.Tensor], str]):
        images = []
        paths = []
        for image, path in batch:
            images.append(image['pixel_values'])
            paths.append(path)

        pixel_values = torch.concat(images)
        return pixel_values, paths
