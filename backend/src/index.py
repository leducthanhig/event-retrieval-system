import logging
import pickle

from cores.indexing import VectorIndexer

from configs import (
    VECTOR_DATA_PATH,
    FAISS_PRESET,
    FAISS_SAVE_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

if __name__ == '__main__':
    # Load data from disk
    with open(VECTOR_DATA_PATH, 'rb') as f:
        vector_data = pickle.load(f)

    # Index features
    vec_indexer = VectorIndexer(FAISS_PRESET)
    vec_indexer.create_index(vector_data['all_vectors'], FAISS_SAVE_PATH)
