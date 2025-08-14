import os
import time
import logging
from typing import Literal
from collections import defaultdict

import numpy as np
import torch
import faiss
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # to disable the warning message
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self,
                 clip_index_path: str,
                 dino_index_path: str,
                 metadata: list[str],
                 clip_model_name: str,
                 clip_pretrained: str,
                 dino_model_name: str,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.metadata = metadata
        self.device = device

        # Init CLIP model and tokenizer
        self.clip_model, _, self.clip_transforms = create_model_and_transforms(clip_model_name,
                                                                               clip_pretrained,
                                                                               device=device)
        self.tokenizer = get_tokenizer(clip_model_name)

        # Init DINO model and processor
        self.dino_model = AutoModel.from_pretrained(dino_model_name, device_map=device)
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name,
                                                                 use_fast=True,
                                                                 return_tensors='pt',
                                                                 device=device)

        # Load indices
        self.indices = dict()
        self.load(clip_index_path, 'clip')
        self.load(dino_index_path, 'dino')

    @torch.no_grad()
    def search_by_text(self, text_query: str, k=10):
        """Search for relevant images to the text query."""
        tokenized_text = self.tokenizer([text_query]).to(self.device)
        features = self.clip_model.encode_text(tokenized_text)
        features /= features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy()
        return self.vector_search(features, 'clip', k)

    @torch.no_grad()
    def search_by_image(self, image_query_path: str, k=10):
        """Search for relevant images to the image query."""
        image = Image.open(image_query_path)
        processed_image = self.dino_processor(image)
        features = self.dino_model(**processed_image)
        features /= features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy()
        return self.vector_search(features, 'dino', k)

    def vector_search(self,
                      query_vector: np.ndarray,
                      index_name: str = 'clip',
                      k=10,
                      nprobe: int | None = None):
        """Search for relevant images."""
        if index_name not in self.indices:
            msg = f"Index '{index_name}' not found in loaded indices."
            logger.error(msg)
            raise KeyError(msg)

        if isinstance(self.indices[index_name], faiss.IndexIVFFlat):
            if not nprobe:
                nprobe = 10
            self.indices[index_name].nprobe = nprobe
            logger.info(f"Set nprobe to {nprobe}")

        # Search index
        k = min(k, self.indices[index_name].ntotal)
        start = time.time()
        distances, indices = self.indices[index_name].search(query_vector.reshape(1, -1), k)
        search_time = time.time() - start

        # Prepare results
        results = [(self.metadata[idx], float(dist))
                   for dist, idx in zip(distances[0], indices[0])]
        if not results:
            logger.warning("No matches found!")
        else:
            logger.info(f"Found {len(results)} matches in {search_time} second(s).")

        # Normalize before returning
        return Retriever.normalize_consine_distances(results)

    def search(self,
               text_query: str,
               image_query_path: str | None = None,
               weights: list[float] | None = None,
               pooling_method: Literal['avg', 'max'] = 'max',
               k=10):
        """Perform multimodal search for all given queries."""
        # Perform the primary text search
        all_results = [self.search_by_text(text_query, k)]

        # Perform other searches if provided
        if image_query_path:
            all_results.append(self.search_by_image(image_query_path, k))

        # Combine results if needed
        if len(all_results) == 1:
            final_results = all_results[0]
        else:
            if not weights:
                msg = "Multiple queries were given but no weights are specified."
                logger.error(msg)
                raise RuntimeError(msg)

            final_results = self.combine_modalities_results(all_results, weights)

        # Combine frames and return
        return self.combine_frames(final_results, pooling_method)

    def load(self, index_path: str, index_name: str):
        """Load faiss index from disk."""
        logger.info(f"Loading index from {index_path}")
        self.indices[index_name] = faiss.read_index(index_path)

        # Determine index type from loaded index
        if isinstance(self.indices[index_name], faiss.IndexFlatL2):
            logger.info("Loaded a Flat index")
        elif isinstance(self.indices[index_name], faiss.IndexIVFFlat):
            logger.info("Loaded an IVF index")
        else:
            logger.warning(f"Loaded index of unknown type: {type(self.indices[index_name])}")

        logger.info(f"Loaded index with {self.indices[index_name].ntotal} vectors")

    @staticmethod
    def combine_modalities_results(all_modal_results: list[list[tuple[str, float]]],
                                   weights: list[float]) -> list[tuple[str, float]]:
        """Combine results from all modalities."""
        if len(all_modal_results) != len(weights):
            msg = "Mismatched between the number of modal results and weights"
            logger.error(msg)
            raise RuntimeError(msg)

        # Convert lists to dictionaries for efficient lookups
        dicts = [defaultdict(float, results) for results in all_modal_results]

        # Get the union of keys from all dictionaries
        all_keys = set()
        for d in dicts:
            all_keys = all_keys.union(set(d.keys()))

        # Combine results with weighted scores
        combined_results = [
            (key, sum([w * d[key] for w, d in zip(weights, dicts)]))
            for key in all_keys
        ]

        # Sort results by the combined score in descending order
        combined_results.sort(key=lambda x: x[1], reverse=True)

        return combined_results

    @staticmethod
    def combine_frames(results: list[tuple[str, float]],
                       pooling_method: Literal['avg', 'max'] = 'max') -> list[dict]:
        """Combine frames in a same shot."""
        # Get pooling function
        if pooling_method == 'avg':
            pooling = np.average
        elif pooling_method == 'max':
            pooling = np.amax
        else:
            msg = f"Unsupported pooling method: {pooling_method}"
            logger.error(msg)
            raise RuntimeError(msg)

        # Group frames by video and shot
        grouped_frames = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for path, score in results:
            # path format: /dir/video_id/shot_id/frame_id.ext
            dirname = os.path.dirname(path)
            shot_id = os.path.basename(dirname)
            dirname = os.path.dirname(dirname)
            video_id = os.path.basename(dirname)
            grouped_frames[video_id][shot_id]['paths'].append(path)
            grouped_frames[video_id][shot_id]['scores'].append(score)

        # Combine results with the specified pooling method
        combined_results = []
        for video_id in grouped_frames:
            for shot_id in grouped_frames[video_id]:
                shot = grouped_frames[video_id][shot_id]

                idx = np.argmax(shot['scores'])
                thumbnail_path = shot['paths'][idx]

                combined_results.append({
                    'video_id': video_id,
                    'shot_id': shot_id,
                    'thumbnail': thumbnail_path,
                    'score': pooling(shot['scores'])
                })

        # Sort results by the combined score in descending order
        combined_results.sort(key=lambda x: x['score'], reverse=True)

        return combined_results

    @staticmethod
    def normalize_consine_distances(results: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Normalize Consine distances from vector search results."""
        distances = np.array([dis for _, dis in results])
        dis_min = distances.min()
        dis_max = distances.max()
        dis_range = dis_max - dis_min
        return [(path, 1 - (dis - dis_min) / dis_range)
                for path, dis in results]

    @staticmethod
    def combine_results(all_results: list[list[dict]],
                        weights: list[float]) -> list[dict]:
        """Combine the results from multi-model search by the given weights."""
        # Group results
        grouped_results = defaultdict(lambda: defaultdict())
        for results in all_results:
            for result in results:
                video_id = result['video_id']
                shot_id = result['shot_id']
                grouped_results[video_id][shot_id] = {
                    'thumbnail': result['thumbnail'],
                    'scores': [0] * len(weights)
                }

        for idx, results in enumerate(all_results):
            for result in results:
                video_id = result['video_id']
                shot_id = result['shot_id']
                score = result['score']
                grouped_results[video_id][shot_id]['scores'][idx] = score

        # Combine results with the given weights
        combined_results = []
        for video_id in grouped_results:
            for shot_id in grouped_results[video_id]:
                shot = grouped_results[video_id][shot_id]
                combined_score = sum([w * score
                                      for w, score in zip(weights, shot['scores'])])
                combined_results.append({
                    'video_id': video_id,
                    'shot_id': shot_id,
                    'thumbnail': shot['thumbnail'],
                    'score': combined_score,
                })

        # Sort results by the combined score in descending order
        combined_results.sort(key=lambda x: x['score'], reverse=True)

        return combined_results
