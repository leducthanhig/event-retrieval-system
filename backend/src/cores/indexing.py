import logging
from typing_extensions import Literal

import numpy as np
import faiss
from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger(__name__)

class VectorIndexer:
    # Define preset configurations
    PRESETS = {
        "high_accuracy": {
            "index_type": "flat",  # Use flat index for highest accuracy
            "n_regions_factor": None,  # Not applicable for flat index
            "description": "Highest accuracy but slower search speed"
        },
        "balanced": {
            "index_type": "ivf_flat",
            "n_regions_factor": 4,  # Default factor for n_regions calculation
            "description": "Balance between accuracy and speed"
        },
        "high_speed": {
            "index_type": "ivf_flat",
            "n_regions_factor": 8,  # More regions for faster search
            "description": "Fastest search speed with reduced accuracy"
        }
    }

    def __init__(self, preset: Literal['high_accuracy', 'balanced', 'high_speed'] = 'balanced'):
        # Validate preset
        if preset not in self.PRESETS:
            raise ValueError(f"Invalid preset '{preset}'. Choose from: {list(self.PRESETS.keys())}")

        preset_config = self.PRESETS[preset]
        self.index_type = preset_config['index_type']
        self.n_regions_factor = preset_config['n_regions_factor']

        # Log preset info
        logger.info(f"Using '{preset}' preset: {preset_config['description']}")
        logger.info(f"Preset configuration: index_type={self.index_type}")

    def create_index(self, vectors: np.ndarray, save_path: str):
        """Index all given feature vectors based on the preset configuration."""
        feature_dim = vectors.shape[-1]

        if self.index_type == 'flat':
            # For high accuracy, use a flat index (exact search)
            logger.info("Creating new Flat index for high accuracy")
            self.index = faiss.IndexFlatL2(feature_dim)

            # Flat index doesn't need training
            self.index.add(vectors)
            logger.info(f"Total vectors in index: {self.index.ntotal}")

        elif self.index_type == 'ivf_flat':
            # Calculate n_regions based on number of vectors and preset factor
            num_vectors = len(vectors)
            n_regions = min(int(self.n_regions_factor * np.sqrt(num_vectors)), num_vectors // 2)

            # Initialize new FAISS IVF index
            logger.info(f"Creating new IVF index with {n_regions} regions")
            quantizer = faiss.IndexFlatL2(feature_dim)
            self.index = faiss.IndexIVFFlat(quantizer, feature_dim,
                                            n_regions, faiss.METRIC_L2)

            # Train index
            logger.info("Training IVF index...")
            self.index.train(vectors)
            logger.info("Index training completed")

            # Add to index
            self.index.add(vectors)
            logger.info(f"Total vectors in index: {self.index.ntotal}")

        logger.info(f"Successfully indexed {len(vectors)} vectors")

        self.save(save_path)

    def save(self, index_path: str):
        """Save the index data to disk."""
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved index with {self.index.ntotal} vectors")
        logger.info(f"Saved index to \"{index_path}\"")

class TextIndexer:
    def __init__(self, host: str, api_key: str):
        self.client = Elasticsearch(host, api_key=api_key)
        logger.info(f"Sucessfully connected to Elasticsearch cluster at {host}")

    def create_index(self, index_name: str, documents: list[dict], mapping: dict = None):
        # Create index (if not exists) with optional mapping
        if not self.client.indices.exists(index=index_name):
            body = {'mappings': mapping} if mapping else {}
            self.client.indices.create(index=index_name, body=body)
            logger.info(f"Created index '{index_name}'")

        # Remove all existing documents
        query = {
            'query': {
                'match_all': {}
            }
        }
        self.client.delete_by_query(index=index_name, body=query)

        # Prepare actions for bulk insert
        actions = [
            {
                '_index': index_name,
                '_source': doc
            }
            for doc in documents
        ]
        # Bulk insert
        helpers.bulk(self.client, actions)
        logger.info(f"Inserted {len(documents)} documents into '{index_name}'")
