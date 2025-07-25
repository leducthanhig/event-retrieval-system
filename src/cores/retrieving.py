import logging
import json

import numpy as np
import torch
import clip
import faiss
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Source code: https://github.com/bnsreenu/python_for_microscopists/blob/master/350%20-%20Efficient%20Image%20Retrieval%20with%20Vision%20Transformer%20(ViT)%20and%20FAISS/retrieval_system.py
class Retriever:
    def __init__(self,
                 index_path: str = None,
                 metadata_path: str = None,
                 nprobe=10,
                 clip_model='ViT-L/14',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.nprobe = nprobe
        self.model, self.preprocess = clip.load(clip_model, device)
        self.device = device
        self.feature_dim = self.model.state_dict()["text_projection"].shape[1]
        logger.info(f"Initializing retriever with dimension: {self.feature_dim}")

        # Load existing index and metadata if provided
        if index_path and metadata_path:
            self.load(index_path, metadata_path)

    def create_index(self,
                     all_features: np.ndarray,
                     metadata: list[dict]):
        """Index all given feature vectors."""
        # Calculate n_regions based on number of vectors
        num_vectors = len(all_features)
        n_regions = min(int(4 * np.sqrt(num_vectors)), num_vectors // 2)

        # Initialize new FAISS IVF index
        logger.info(f"Creating new IVF index with {n_regions} regions")
        self.quantizer = faiss.IndexFlatL2(self.feature_dim)
        self.index = faiss.IndexIVFFlat(self.quantizer, self.feature_dim,
                                        n_regions, faiss.METRIC_L2)
        self.index.nprobe = self.nprobe

        # Train index
        logger.info("Training IVF index...")
        self.index.train(all_features)
        logger.info("Index training completed")

        # Add to index
        self.index.add(all_features)
        logger.info(f"Total vectors in index: {self.index.ntotal}")

        # Update metadata
        self.metadata = metadata

        logger.info(f"Successfully indexed {len(all_features)} vectors")

    @torch.no_grad()
    def search_by_text(self, text_query: str, k=10):
        """Search for relevant images to the text query."""
        tokenized_text = clip.tokenize([text_query]).to(self.device)
        features = self.model.encode_text(tokenized_text)
        features /= features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy()
        return self.search(features, k)

    @torch.no_grad()
    def search_by_image(self, image_query_path: str, k=10):
        """Search for similar images the image query."""
        image = Image.open(image_query_path)
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(processed_image)
        features /= features.norm(dim=-1, keepdim=True)
        features = features.cpu().numpy()
        return self.search(features, k)

    def search(self, query_features: np.ndarray, k=10) -> list[tuple[dict, float]]:
        """Search for similar images."""
        if not self.index.is_trained:
            raise RuntimeError("Index has not been trained. Add images first.")

        # Search index
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_features.reshape(1, -1), k)

        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((self.metadata[idx], float(dist)))
            logger.info(f"Match found: {self.metadata[idx]['path']} with distance {dist:.3f}")

        # Sort results by distance (smaller is better)
        results.sort(key=lambda x: x[1])

        if not results:
            logger.warning("No matches found!")
        else:
            logger.info(f"Found {len(results)} matches")

        return results

    def save(self, index_path: str, metadata_path: str):
        """Save the index and metadata to disk."""
        faiss.write_index(self.index, index_path)

        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)

        logger.info(f"Saved index with {self.index.ntotal} vectors")
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")

    def load(self, index_path: str, metadata_path: str):
        """Load the index and metadata from disk."""
        logger.info(f"Loading index from {index_path}")
        self.index = faiss.read_index(index_path)

        # Set nprobe for loaded index
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = self.nprobe
            logger.info(f"Set nprobe to {self.nprobe} for loaded IVF index")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        logger.info(f"Metadata contains {len(self.metadata)} entries")
