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
from elasticsearch import Elasticsearch

from utils import encode_object_bbox

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self,
                 vector_index_path: str,
                 metadata: list[str],
                 clip_model: str,
                 clip_weights: str,
                 elastic_host: str,
                 elastic_api_key: str,
                 elastic_index_name: str,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.metadata = metadata
        self.elastic_index_name = elastic_index_name
        self.device = device

        self.model, _, self.preprocess = create_model_and_transforms(clip_model,
                                                                     clip_weights,
                                                                     device=device)
        self.tokenizer = get_tokenizer(clip_model)

        # Load index and metadata
        self.load(vector_index_path)

        # Connect to Elasticsearch cluster
        self.client = Elasticsearch(elastic_host, api_key=elastic_api_key)
        logger.info(f"Sucessfully connected to Elasticsearch cluster at {elastic_host}")

    @torch.no_grad()
    def search_by_text(self, text_query: str, k=10):
        """Search for relevant images to the text query."""
        tokenized_text = self.tokenizer([text_query]).to(self.device)
        features = self.model.encode_text(tokenized_text)
        features /= features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy()
        return self.semantic_search(features, k)

    @torch.no_grad()
    def search_by_image(self, image_query_path: str, k=10):
        """Search for relevant images the image query."""
        image = Image.open(image_query_path)
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(processed_image)
        features /= features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy()
        return self.semantic_search(features, k)

    def semantic_search(self, query_vector: np.ndarray, k=10, nprobe: int = None):
        """Search for relevant images."""
        if isinstance(self.vector_index, faiss.IndexIVFFlat):
            if not nprobe:
                nprobe = 10
            self.vector_index.nprobe = nprobe
            logger.info(f"Set nprobe to {nprobe}")

        # Search index
        k = min(k, self.vector_index.ntotal)
        start = time.time()
        distances, indices = self.vector_index.search(query_vector.reshape(1, -1), k)
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

    def search_by_object_counts(self, counts: list[tuple[str, int]], k=10):
        """Search by the number of instances of each object."""
        text_query = ' '.join(label
                              for label, count in counts
                              for _ in range(count))
        return self.full_text_search(text_query, 'objects', k)

    def search_by_object_locations(self, bboxes: list[dict], k=10):
        """Search by the bounding box location of each object."""
        text_query = ' '.join([encode_object_bbox(bbox) for bbox in bboxes])
        return self.full_text_search(text_query, 'locations', k)

    def full_text_search(self, text_query: str, search_field: str, k=10):
        """Search for similar textual encoded attribute."""
        # Construct the query
        query = {
            'query': {
                'match': {
                    search_field: {
                        'query': text_query
                    }
                }
            },
            '_source': 'path',
            'size': k
        }
        # Run search
        response = self.client.search(index=self.elastic_index_name, body=query)

        # Prepare results
        results = [(doc['_source']['path'], doc['_score'])
                   for doc in response['hits']['hits']]
        if not results:
            logger.warning("No matches found!")
        else:
            logger.info(f"Found {len(results)} matches in {response['took'] / 1000} second(s).")

        # Normalize before returning
        return Retriever.normalize_bm25_scores(results)

    def search(self,
               text_query: str,
               object_counts: list[tuple[str, int]] = None,
               object_bboxes: list[dict] = None,
               weights: list[float] = None,
               pooling_method: Literal['avg', 'max'] = 'max',
               k=10):
        """Perform multimodal search for all given queries."""
        # Perform the primary text search
        all_results = [self.search_by_text(text_query, k)]

        # Perform other searches if provided
        if object_counts:
            all_results.append(self.search_by_object_counts(object_counts, k))
        if object_bboxes:
            all_results.append(self.search_by_object_locations(object_bboxes, k))

        # Combine results if needed
        if len(all_results) == 1:
            final_results = all_results[0]
        else:
            final_results = self.combine_modalities_results(all_results, weights)

        # Combine frames and return
        return self.combine_frames(final_results, pooling_method)

    def load(self, index_path: str):
        """Load the index and metadata from disk."""
        logger.info(f"Loading index from {index_path}")
        self.vector_index = faiss.read_index(index_path)

        # Determine index type from loaded index
        if isinstance(self.vector_index, faiss.IndexFlatL2):
            logger.info("Loaded a Flat index")
        elif isinstance(self.vector_index, faiss.IndexIVFFlat):
            logger.info("Loaded an IVF index")
        else:
            logger.warning(f"Loaded index of unknown type: {type(self.vector_index)}")

        logger.info(f"Loaded index with {self.vector_index.ntotal} vectors")
        logger.info(f"Metadata contains {len(self.metadata)} entries")

    def combine_modalities_results(self,
                                   all_modal_results: list[list[tuple[str, float]]],
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

    def combine_frames(self,
                       results: list[tuple[str, float]],
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
    def normalize_bm25_scores(results: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Normalize BM25 scores from full-text search results."""
        scores = np.array([score for _, score in results])
        score_min = scores.min()
        score_max = scores.max()
        score_range = score_max - score_min
        return [(path, (score - score_min) / score_range)
                for path, score in results]
