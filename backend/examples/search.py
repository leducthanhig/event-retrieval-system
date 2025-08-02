import os
import logging
import pickle
import json

from ..src.cores.retrieving import Retriever

from ..src.configs import (
    CLIP_MODEL,
    CLIP_PRETRAINED,
    VECTOR_DATA_PATH,
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
    with open(VECTOR_DATA_PATH, 'rb') as f:
        metadata = pickle.load(f)['paths']
    retriever = Retriever(FAISS_SAVE_PATH,
                        metadata,
                        CLIP_MODEL,
                        CLIP_PRETRAINED,
                        ELASTIC_HOST,
                        os.environ['ES_LOCAL_API_KEY'],
                        ELASTIC_INDEX_NAME)

    print(json.dumps(retriever.search('a photo of a car',
                                      object_counts=[('car', 3)],
                                      weights=[0.5, 0.5]), indent=2))
