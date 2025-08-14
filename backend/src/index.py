import logging

import numpy as np

from cores.indexing import VectorIndexer

from configs import (
    CLIP_VECTOR_DATA_PATH,
    DINO_VECTOR_DATA_PATH,
    CLIP_INDEX_SAVE_PATH,
    DINO_INDEX_SAVE_PATH,
    FAISS_PRESET,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

if __name__ == '__main__':
    data_paths = [
        CLIP_VECTOR_DATA_PATH,
        DINO_VECTOR_DATA_PATH,
    ]
    save_paths = [
        CLIP_INDEX_SAVE_PATH,
        DINO_INDEX_SAVE_PATH,
    ]
    for data_path, save_path in zip(data_paths, save_paths):
        # Load data from disk
        vector_data = np.load(data_path)

        # Index features
        vec_indexer = VectorIndexer(FAISS_PRESET)
        vec_indexer.create_index(vector_data, save_path)
