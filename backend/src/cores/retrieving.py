import os
import time
import logging
from typing import Literal
from collections import defaultdict

import numpy as np
import torch
import faiss
from PIL import Image
from PIL.ImageFile import ImageFile
from open_clip import create_model_and_transforms, get_tokenizer
from elasticsearch import Elasticsearch

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # to disable the warning message
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self,
                 clip_index_path: str | list[str],
                 dino_index_path: str,
                 clip_model_name: str | list[str],
                 clip_pretrained: str | list[str],
                 dino_model_name: str,
                 metadata: list[str],
                 elastic_host: str,
                 elastic_api_key: str,
                 media_info_index_name: str,
                 transcriptions_index_name: str,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.metadata = metadata
        self.media_info_index_name = media_info_index_name
        self.transcriptions_index_name = transcriptions_index_name
        self.device = device

        # Store paths instead of loading indices
        self.clip_index_paths = [clip_index_path] if isinstance(clip_index_path, str) else clip_index_path
        self.dino_index_path = dino_index_path

        # Store model configurations instead of loading models
        self.clip_model_names = [clip_model_name] if isinstance(clip_model_name, str) else clip_model_name
        self.clip_pretraineds = [clip_pretrained] if isinstance(clip_pretrained, str) else clip_pretrained
        self.dino_model_name = dino_model_name

        # Initialize containers
        self.indices = {}
        self.clip_model = defaultdict(defaultdict)
        self.dino_model = None
        self.dino_processor = None

        # Connect to Elasticsearch cluster
        self.client = Elasticsearch(elastic_host, api_key=elastic_api_key)
        logger.info(f"Sucessfully connected to Elasticsearch cluster at {elastic_host}")

        # Preload first CLIP model and its index
        if self.clip_model_names and self.clip_pretraineds:
            first_model = self.clip_model_names[0]
            first_pretrained = self.clip_pretraineds[0]
            logger.info(f"Preloading first CLIP model: {first_model} with {first_pretrained}")

            # Load the first CLIP model
            self.load_clip_model(first_model, first_pretrained)

            # Load the corresponding index
            first_index_name = f"{first_model}_{first_pretrained}"
            for path in self.clip_index_paths:
                if first_index_name in path:
                    self.load_index(path)
                    logger.info(f"Preloaded index for {first_index_name}")
                    break

    def get_clip_model(self, model_name: str, pretrained: str):
        """Lazily load a CLIP model if not already loaded."""
        if pretrained not in self.clip_model or model_name not in self.clip_model[pretrained]:
            self.load_clip_model(model_name, pretrained)
        return self.clip_model[pretrained][model_name]

    def load_clip_model(self, model_name: str, pretrained: str):
        """Load a specific CLIP model."""
        logger.info(f"Loading CLIP model {model_name} with pretrained {pretrained}")
        model, _, _ = create_model_and_transforms(model_name, pretrained, device=self.device)
        tokenizer = get_tokenizer(model_name)
        # keep the bound encoder callable (uses the text submodule)
        encoder = model.encode_text

        # drop the visual tower to free RAM
        if hasattr(model, 'visual'):
            try:
                del model.visual
                logger.info('Dropped the visual tower to free RAM')
            except Exception:
                pass

        # help Python / CUDA free memory
        import gc
        gc.collect()
        if self.device != 'cpu':
            try:
                torch.cuda.empty_cache()
                logger.info('Cleared CUDA cache')
            except Exception:
                pass

        self.clip_model[pretrained][model_name] = dict(encoder=encoder, tokenizer=tokenizer)

    def get_dino_model(self):
        """Lazily load the DINO model if not already loaded."""
        if self.dino_model is None:
            self.load_dino_model()
        return self.dino_model, self.dino_processor

    def load_dino_model(self):
        """Load the DINO model and processor."""
        logger.info(f"Loading DINO model {self.dino_model_name}")
        self.dino_model = AutoModel.from_pretrained(self.dino_model_name, device_map=self.device)
        self.dino_processor = AutoImageProcessor.from_pretrained(self.dino_model_name,
                                                            use_fast=True,
                                                            return_tensors='pt',
                                                            device=self.device)

    def get_index(self, index_name: str):
        """Lazily load a Faiss index if not already loaded."""
        if index_name not in self.indices:
            # For CLIP models - extract model/pretrained from index name
            if index_name.startswith(tuple(model_name for model_name in self.clip_model_names)):
                for path in self.clip_index_paths:
                    if index_name in path:
                        self.load_index(path)
                        break
            # For DINO model
            elif index_name == os.path.splitext(os.path.basename(self.dino_index_path))[0].removeprefix('index_'):
                self.load_index(self.dino_index_path)
            else:
                msg = f"Index '{index_name}' not found in configured indices."
                logger.error(msg)
                raise KeyError(msg)

        return self.indices[index_name]

    def load_index(self, index_path: str):
        """Load faiss index from disk."""
        logger.info(f"Loading index from {index_path}")
        index_name = os.path.splitext(os.path.basename(index_path))[0].removeprefix('index_')
        self.indices[index_name] = faiss.read_index(index_path)

        # Determine index type from loaded index
        if isinstance(self.indices[index_name], faiss.IndexFlatL2):
            logger.info("Loaded a Flat index")
        elif isinstance(self.indices[index_name], faiss.IndexIVFFlat):
            logger.info("Loaded an IVF index")
        else:
            logger.warning(f"Loaded index of unknown type: {type(self.indices[index_name])}")

        logger.info(f"Loaded index with {self.indices[index_name].ntotal} vectors")

    @torch.no_grad()
    def search_by_text(self, text_query: str, model_name: str, pretrained: str, k=10):
        """Search for relevant images to the text query."""
        model = self.get_clip_model(model_name, pretrained)
        tokenized_text = model['tokenizer']([text_query]).to(self.device)
        features = model['encoder'](tokenized_text)
        features /= features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy()
        return self.vector_search(features, f"{model_name}_{pretrained}", k)

    @torch.no_grad()
    def search_by_image(self, image_query: str | ImageFile, index_name: str, k=10):
        """Search for relevant images to the image query."""
        if isinstance(image_query, str):
            image = Image.open(image_query)
        else:
            image = image_query

        dino_model, dino_processor = self.get_dino_model()
        processed_image = dino_processor(image)
        features = dino_model(**processed_image).pooler_output
        features /= features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy()
        return self.vector_search(features, index_name, k)

    def vector_search(self,
                      query_vector: np.ndarray,
                      index_name: str,
                      k=10,
                      nprobe: int | None = None):
        """Perform vector search on Faiss index."""
        index = self.get_index(index_name)

        if isinstance(index, faiss.IndexIVFFlat):
            if not nprobe:
                nprobe = 10
            index.nprobe = nprobe
            logger.info(f"Set nprobe to {nprobe}")

        # Search index
        k = min(k, index.ntotal)
        start = time.time()
        distances, indices = index.search(query_vector.reshape(1, -1), k)
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

    def media_info_search(self, text_query: str, k=10) -> list[tuple[str, float]]:
        """Search for relevant media info."""
        query = {
            'multi_match': {
                'query': text_query,
                'fields': ['title', 'description', 'keywords'],
            },
        }
        results = self.full_text_search(query, self.media_info_index_name, 'video_id', k)
        return [(res[0]['video_id'], res[1]) for res in results]

    def transcription_search(self, text_query: str, k=10):
        """Search for relevant transcriptions."""
        query = {
            'match': {
                'text': {
                    'query': text_query,
                },
            },
        }
        return self.full_text_search(query, self.transcriptions_index_name, ['id', 'video_id'], k)

    def full_text_search(self,
                         query: dict,
                         index_name: str,
                         return_fields: str | list[str] | None = None,
                         k=10):
        """Perform full-text search on Elasticsearch index."""
        # Construct the query body
        body = {
            'query': query,
            'size': k,
        }

        if return_fields:
            body['_source'] = return_fields

        # Run search
        response = self.client.search(index=index_name, body=body)

        # Prepare results
        results = [(doc['_source'], doc['_score'])
                   for doc in response['hits']['hits']]
        if not results:
            logger.warning("No matches found!")
            return []
        else:
            logger.info(f"Found {len(results)} matches in {response['took'] / 1000} second(s).")

        # Normalize before returning
        return Retriever.normalize_bm25_scores(results)

    @staticmethod
    def combine_frame_results(all_results: list[list[tuple[str, float]]],
                              weights: list[float]) -> list[tuple[str, float]]:
        """Combine frame-level results."""
        if len(all_results) != len(weights):
            msg = "Mismatched between the number of result set and weights"
            logger.error(msg)
            raise RuntimeError(msg)

        # Convert lists to dictionaries for efficient lookups
        dicts = [defaultdict(float, results) for results in all_results]

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
    def combine_shot_results(all_results: list[list[dict]],
                             weights: list[float]) -> list[dict]:
        """Combine shot-level results."""
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

                # Assume that the video-level results are at the last of the list
                # For results which are video-level
                # Assign score for all existing shots
                if not shot_id:
                    for shot in grouped_results[video_id]:
                        grouped_results[video_id][shot]['scores'][idx] = score
                else:
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
    def normalize_bm25_scores(results: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
        """Normalize BM25 scores from full-text search results."""
        scores = np.array([score for _, score in results])
        score_min = scores.min()
        score_max = scores.max()
        score_range = score_max - score_min
        return [(path, (score - score_min) / score_range)
                for path, score in results]
