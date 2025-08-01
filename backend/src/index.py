import os
import logging
import pickle
import json

from cores.indexing import VectorIndexer, TextIndexer
from utils import encode_object_bbox

from configs import (
    VECTOR_DATA_PATH,
    OBJECT_DATA_PATH,
    FAISS_PRESET,
    FAISS_SAVE_PATH,
    ELASTIC_HOST,
    ELASTIC_INDEX_NAME
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

    with open(OBJECT_DATA_PATH, 'r') as f:
        object_data = json.load(f)

    # Index features
    vec_indexer = VectorIndexer(FAISS_PRESET)
    vec_indexer.create_index(vector_data['all_vectors'], FAISS_SAVE_PATH)

    obj_det_tol = 0.75
    docs = [{
        'objects': ' '.join([obj['label']
                             for obj in objects
                             if obj['score'] >= obj_det_tol]),
        'locations': ' '.join([encode_object_bbox(obj)
                               for obj in objects
                               if obj['score'] >= obj_det_tol]),
        'path': path
    } for path, objects in zip(object_data['paths'], object_data['all_objects'])]
    mapping = {
        'properties': {
            'objects': {'type': 'text'},
            'locations': {'type': 'text'},
            'path': {'type': 'keyword'}
        }
    }
    es_indexer = TextIndexer(ELASTIC_HOST, api_key=os.environ['ES_LOCAL_API_KEY'])
    es_indexer.create_index(ELASTIC_INDEX_NAME, docs, mapping)
