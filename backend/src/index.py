import os
import logging
import pickle
import json

from cores.sampling import FrameSampler
from cores.extracting import FeatureExtractor
from cores.retrieving import Retriever
from cores.models import VideoModel

from configs import (
    INP_VIDEO_DIR,
    OUT_FRAME_DIR,
    VIDEO_DATA_PATH,
    SHOT_DATA_PATH,
    FEATURES_PATH,
    METADATA_PATH,
    INDEX_DATA_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

if __name__ == '__main__':
    # Create video data
    video_data: list[VideoModel] = [{
        'id': os.path.splitext(filename)[0],
        'path': os.path.join(dirname, filename)
    }
    for dirname, _, filenames in os.walk(INP_VIDEO_DIR)
    for filename in filenames]

    # Save video data to disk
    with open(VIDEO_DATA_PATH, 'w') as f:
        json.dump(video_data, f, indent=2)

    # Sample frames
    sampler = FrameSampler(video_data, OUT_FRAME_DIR)
    shot_data, frame_data = sampler.run()

    # Save shot and frame data to disk
    with open(SHOT_DATA_PATH, 'w') as f:
        json.dump(shot_data, f, indent=2)
    with open(METADATA_PATH, 'w') as f:
        json.dump(frame_data, f, indent=2)

    # Extract features
    extractor = FeatureExtractor(frame_data)
    all_features, metadata = extractor.run()

    # Save the extracted feature vectors and metadata to disk
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(all_features, f)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Index all feature vector
    retriever = Retriever()
    retriever.create_index(all_features, metadata)

    # Save index data and metadata to disk
    retriever.save(INDEX_DATA_PATH, METADATA_PATH)
