import logging

import numpy as np
import faiss
from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger(__name__)

class VectorIndexer:
    def __init__(self):
        self.indices: dict[str, faiss.IndexFlatIP] = {}

    def create_index(self, vectors: np.ndarray, index_name: str, save_path: str):
        """Index all given feature vectors."""
        feature_dim = vectors.shape[-1]
        self.indices[index_name] = faiss.IndexFlatIP(feature_dim)
        logger.info(f"Created Flat index for {feature_dim}-dimension vectors")

        self.indices[index_name].add(vectors)
        logger.info(f"Added {self.indices[index_name].ntotal} vectors")

        self.save(index_name, save_path)

    def save(self, index_name: str, index_path: str):
        """Save the index data to disk."""
        faiss.write_index(self.indices[index_name], index_path)
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
