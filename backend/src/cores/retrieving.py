import logging
import json
import time
from typing_extensions import Literal

import numpy as np
import torch
import faiss
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer

from cores.models import FrameModel
from configs import CLIP_MODEL, CLIP_WEIGHTS

logger = logging.getLogger(__name__)

# Source code: https://github.com/bnsreenu/python_for_microscopists/blob/master/350%20-%20Efficient%20Image%20Retrieval%20with%20Vision%20Transformer%20(ViT)%20and%20FAISS/retrieval_system.py
class Retriever:
    # Define preset configurations
    PRESETS = {
        "high_accuracy": {
            "index_type": "flat",  # Use flat index for highest accuracy
            "nprobe": None,        # Not applicable for flat index
            "n_regions_factor": None,  # Not applicable for flat index
            "description": "Highest accuracy but slower search speed"
        },
        "balanced": {
            "index_type": "ivf_flat",
            "nprobe": 10,          # Default value
            "n_regions_factor": 4,  # Default factor for n_regions calculation
            "description": "Balance between accuracy and speed"
        },
        "high_speed": {
            "index_type": "ivf_flat",
            "nprobe": 5,           # Lower nprobe for faster search
            "n_regions_factor": 8,  # More regions for faster search
            "description": "Fastest search speed with reduced accuracy"
        }
    }

    def __init__(self,
                 index_path: str = None,
                 metadata_path: str = None,
                 preset: Literal['high_accuracy', 'balanced', 'high_speed'] = 'balanced',  # Default preset
                 nprobe: int = None,        # Override preset if provided
                 clip_model=CLIP_MODEL,
                 clip_weights=CLIP_WEIGHTS,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Validate preset
        if preset not in self.PRESETS:
            raise ValueError(f"Invalid preset '{preset}'. Choose from: {list(self.PRESETS.keys())}")

        self.preset = preset
        preset_config = self.PRESETS[preset]

        # Use provided nprobe if specified, otherwise use preset value
        self.nprobe = nprobe if nprobe is not None else preset_config['nprobe']
        self.index_type = preset_config['index_type']
        self.n_regions_factor = preset_config['n_regions_factor']

        self.device = device

        # Log preset info
        logger.info(f"Using '{preset}' preset: {preset_config['description']}")
        logger.info(f"Preset configuration: index_type={self.index_type}, nprobe={self.nprobe}")

        self.model, _, self.preprocess = create_model_and_transforms(clip_model,
                                                                     clip_weights,
                                                                     device=device)
        self.tokenizer = get_tokenizer(CLIP_MODEL)
        self.feature_dim = next(reversed(self.model.state_dict().values())).shape[0]
        logger.info(f"Initializing retriever with dimension: {self.feature_dim}")

        # Load existing index and metadata if provided
        if index_path and metadata_path:
            self.load(index_path, metadata_path)

    def create_index(self,
                     all_features: np.ndarray,
                     metadata: list[FrameModel]):
        """Index all given feature vectors based on the preset configuration."""
        num_vectors = len(all_features)

        if self.index_type == 'flat':
            # For high accuracy, use a flat index (exact search)
            logger.info("Creating new Flat index for high accuracy")
            self.index = faiss.IndexFlatL2(self.feature_dim)

            # Flat index doesn't need training
            self.index.add(all_features)
            logger.info(f"Total vectors in index: {self.index.ntotal}")

        elif self.index_type == 'ivf_flat':
            # Calculate n_regions based on number of vectors and preset factor
            n_regions = min(int(self.n_regions_factor * np.sqrt(num_vectors)), num_vectors // 2)

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
        tokenized_text = self.tokenizer([text_query]).to(self.device)
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

    def search(self, query_features: np.ndarray, k=10) -> list[tuple[FrameModel, float]]:
        """Search for similar images."""
        # Check if index exists and is initialized
        if not hasattr(self, 'index'):
            raise RuntimeError("Index has not been created or loaded.")

        # For IVF indexes, check if trained
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            raise RuntimeError("Index has not been trained. Add images first.")

        # Search index
        k = min(k, self.index.ntotal)
        start = time.time()
        distances, indices = self.index.search(query_features.reshape(1, -1), k)
        search_time = time.time() - start

        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((self.metadata[idx], float(dist)))

        if not results:
            logger.warning("No matches found!")
        else:
            logger.info(f"Found {len(results)} matches in {search_time} second(s).")

        return results

    def save(self, index_path: str, metadata_path: str):
        """Save the index and metadata to disk."""
        faiss.write_index(self.index, index_path)

        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved index with {self.index.ntotal} vectors")
        logger.info(f"Saved index to \"{index_path}\" and metadata to \"{metadata_path}\"")

    def load(self, index_path: str, metadata_path: str):
        """Load the index and metadata from disk."""
        logger.info(f"Loading index from {index_path}")
        self.index = faiss.read_index(index_path)

        # Determine index type from loaded index for proper handling
        if isinstance(self.index, faiss.IndexFlatL2):
            self.index_type = 'flat'
            logger.info("Loaded a Flat index")
        elif isinstance(self.index, faiss.IndexIVFFlat):
            self.index_type = 'ivf_flat'
            # Apply preset's nprobe to loaded index
            self.index.nprobe = self.nprobe
            logger.info(f"Loaded an IVF index, set nprobe to {self.nprobe}")
        else:
            logger.warning(f"Loaded index of unknown type: {type(self.index)}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        logger.info(f"Metadata contains {len(self.metadata)} entries")
