import os
import json
import logging

import numpy as np
from dotenv import load_dotenv

from cores.indexing import VectorIndexer, TextIndexer

from configs import (
    DATA_ROOT_DIR,
    CLIP_MODELS,
    DINO_VECTOR_DATA_PATH,
    DINO_INDEX_SAVE_PATH,
    MEDIA_INFO_DIR,
    WHISPER_OUTPUT_PATH,
    DOT_ENV_FILE,
    ELASTIC_HOST,
    MEDIA_INFO_INDEX_NAME,
    TRANSCRIPTION_INDEX_NAME,
)

# Load environment variables from the .env file
load_dotenv(DOT_ENV_FILE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

if __name__ == '__main__':
    # Index vectors
    data_paths = [
        *[f"{DATA_ROOT_DIR}/vectors_{model}_{pretrained}.npy"
          for model, pretrained in CLIP_MODELS],
        DINO_VECTOR_DATA_PATH,
    ]
    save_paths = [
        *[f"{DATA_ROOT_DIR}/index_{model}_{pretrained}.bin"
          for model, pretrained in CLIP_MODELS],
        DINO_INDEX_SAVE_PATH,
    ]
    vec_indexer = VectorIndexer()
    for data_path, save_path in zip(data_paths, save_paths):
        # Load data from disk
        vector_data = np.load(data_path)

        # Index features
        vec_indexer.create_index(vector_data, '', save_path)

    # Index text info
    es_indexer = TextIndexer(ELASTIC_HOST, api_key=os.environ['ES_LOCAL_API_KEY'])

    # Index media info
    fields = ['title', 'description', 'keywords', 'publish_date']
    docs = []
    for file in os.listdir(MEDIA_INFO_DIR):
        with open(os.path.join(MEDIA_INFO_DIR, file), encoding='utf-8') as f:
            info = json.load(f)
        video_id = os.path.splitext(file)[0]
        docs.append({'video_id': video_id} | {key: info[key] for key in fields})

    mapping = {
        'properties': {
            'video_id': {
                'type': 'keyword',
                'index': 'false'
            },
            'title': {
                'type': 'text'
            },
            'description': {
                'type': 'text'
            },
            'keywords': {
                'type': 'keyword'
            },
            'publish_date': {
                'type': 'date',
                'format': 'dd/MM/yyyy'
            }
        }
    }
    es_indexer.create_index(MEDIA_INFO_INDEX_NAME, docs, mapping)

    # Index transcriptions
    with open(WHISPER_OUTPUT_PATH, encoding='utf-8') as f:
        segments = json.load(f)
    fields = ['id', 'video_id', 'text']
    docs = [{key: seg[key] for key in fields} for seg in segments]
    mapping = {
        'properties': {
            'id': {
                'type': 'integer',
            },
            'video_id': {
                'type': 'keyword',
                'index': 'false'
            },
            'text': {
                'type': 'text'
            }
        }
    }
    es_indexer.create_index(TRANSCRIPTION_INDEX_NAME, docs, mapping)
