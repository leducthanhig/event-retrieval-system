import logging
from typing_extensions import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights)
from open_clip import create_model_and_transforms

from cores.models import FrameModel, ObjectFeatureModel

from configs import CLIP_MODEL, CLIP_WEIGHTS

logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    def __init__(self,
                 image_data: list[FrameModel],
                 transforms: Callable,
                 permute_channels=True):
        self.image_data = image_data
        self.transforms = transforms
        self.permute_channels = permute_channels

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx: int):
        data = self.image_data[idx]
        image = Image.open(data['path'])

        # Permute (H, W, C) to (C, H, W)
        if self.permute_channels:
            image = np.array(image)
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)

        image = self.transforms(image)
        return image, data['id']

class FeatureExtractor:
    def __init__(self,
                 image_data: list[FrameModel],
                 clip_model=CLIP_MODEL,
                 clip_weights=CLIP_WEIGHTS,
                 obj_det_tol=0.5,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.image_data = image_data
        self.clip_model = clip_model
        self.clip_weights = clip_weights
        self.obj_det_tol = obj_det_tol
        self.device = device

        self.image_data_selected = [image for image in image_data if image['selected']]

    @torch.no_grad()
    def extract_spatial_features(self,
                                 batch_size=16,
                                 num_workers=4) -> tuple[list[np.ndarray], list[str]]:
        """Extract the general appearance/spatial features."""
        model, _, preprocess = create_model_and_transforms(self.clip_model,
                                                           self.clip_weights,
                                                           device=self.device)
        model.eval()

        dataset = ImageDataset(self.image_data_selected, preprocess, permute_channels=False)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        all_features = []
        all_ids = []
        progress_bar = tqdm(desc='Extracting spatial features', total=len(dataset))
        for batch, ids in loader:
            batch = batch.to(self.device)

            features = model.encode_image(batch)
            features /= features.norm(dim=-1, keepdim=True)

            all_features.extend(features.cpu().numpy())
            all_ids.extend(ids)

            progress_bar.update(batch.size(0))

        progress_bar.close()

        logger.info(f"Successfully extracted {len(all_features)} spatial feature vectors")

        return all_features, all_ids

    # Use batch size of 1 due to the inconsistant of video resolutions
    @torch.no_grad()
    def extract_objects_and_locations(self,
                                      batch_size=1,
                                      num_workers=4) -> tuple[list[ObjectFeatureModel], list[str]]:
        """Extract objects and their locations."""
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, progress=False)
        model.eval()
        model.to(self.device)
        transforms = weights.transforms()

        dataset = ImageDataset(self.image_data_selected, transforms)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        all_object_info = []
        all_ids = []
        progress_bar = tqdm(desc='Extracting object and location', total=len(dataset))
        for batch, ids in loader:
            batch = batch.to(self.device)

            all_predictions = model(batch)

            for preds in all_predictions:
                object_list: list[ObjectFeatureModel] = []
                for box, label, score in zip(*preds.values()):
                    if score < self.obj_det_tol:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    object_list.append({
                        'label': weights.meta['categories'][label],
                        'xmin': x1,
                        'ymin': y1,
                        'xmax': x2,
                        'ymax': y2,
                        'score': score.item()
                    })

                all_object_info.append(object_list)
            all_ids.extend(ids)

            progress_bar.update(batch.size(0))

        progress_bar.close()

        logger.info(f"Successfully extracted {len(all_object_info)} object features")

        return all_object_info, all_ids

    def run(self) -> tuple[np.ndarray, list[FrameModel]]:
        """Run all available feature extractors."""
        metadata = pd.DataFrame(self.image_data)

        all_features, ids = self.extract_spatial_features()
        df_feat = pd.DataFrame({'id': ids, 'features': all_features})
        metadata = pd.merge(metadata, df_feat, 'left', 'id')

        all_object_info, ids = self.extract_objects_and_locations()
        df_obj = pd.DataFrame({'id': ids, 'objects': all_object_info})
        metadata = pd.merge(metadata, df_obj, 'left', 'id')

        metadata['objects'] = metadata['objects'].apply(lambda x: x if isinstance(x, list) else [])
        metadata = metadata.sort_values('path')
        all_features = np.stack(metadata.pop('features').dropna().to_numpy())
        return all_features, metadata.to_dict(orient='records')
