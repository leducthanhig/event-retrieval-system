import logging
from typing import Callable

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights)
from open_clip import create_model_and_transforms

logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    def __init__(self,
                 image_paths: list[str],
                 transforms: Callable,
                 permute_channels=True):
        self.image_paths = image_paths
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

class FeatureExtractor:
    def __init__(self,
                 image_paths: list[str],
                 clip_model: str,
                 clip_pretrained: str,
                 obj_det_tol=0.5,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.image_paths = image_paths
        self.obj_det_tol = obj_det_tol
        self.device = device

        self.extractors = {
            'spatial_feature': self.init_spatial_feature_extractor(clip_model, clip_pretrained),
            'object_feature': self.init_object_feature_extractor()
        }

    def init_spatial_feature_extractor(self, model_name, pretrained):
        model, _, preprocess = create_model_and_transforms(model_name, pretrained, device=self.device)
        model.eval()
        dataset = ImageDataset(self.image_paths, preprocess, permute_channels=False)

        return {
            'model': model,
            'dataset': dataset
        }

    def init_object_feature_extractor(self):
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, progress=False)
        model.eval()
        model.to(self.device)
        transforms = weights.transforms()
        dataset = ImageDataset(self.image_paths, transforms)

        return {
            'model': model,
            'categories': weights.meta['categories'],
            'dataset': dataset
        }

    @torch.no_grad()
    def extract_spatial_features(self,
                                 batch_size=16,
                                 num_workers=4) -> tuple[np.ndarray, list[str]]:
        """Extract the general appearance/spatial features."""
        extractor = self.extractors['spatial_feature']
        model = extractor['model']
        dataset = extractor['dataset']
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

        return np.stack(all_features), all_paths

    # Use batch size of 1 due to the inconsistant of video resolutions
    @torch.no_grad()
    def extract_object_features(self,
                                batch_size=1,
                                num_workers=4) -> tuple[list[list[dict]], list[str]]:
        """Extract objects and their locations."""
        extractor = self.extractors['object_feature']
        model = extractor['model']
        categories = extractor['categories']
        dataset = extractor['dataset']
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        all_object_info = []
        all_paths = []
        progress_bar = tqdm(desc='Extracting object and location', total=len(dataset))
        for batch, paths in loader:
            batch = batch.to(self.device)

            all_predictions = model(batch)

            for preds in all_predictions:
                object_list = []
                for box, label, score in zip(*preds.values()):
                    if score < self.obj_det_tol:
                        break
                    x1, y1, x2, y2 = box.tolist()
                    object_list.append({
                        'label': categories[label],
                        'xmin': x1,
                        'ymin': y1,
                        'xmax': x2,
                        'ymax': y2,
                        'score': score.item()
                    })

                all_object_info.append(object_list)
            all_paths.extend(paths)

            progress_bar.update(batch.size(0))

        progress_bar.close()

        logger.info(f"Successfully extracted {len(all_object_info)} object features")

        return all_object_info, all_paths
