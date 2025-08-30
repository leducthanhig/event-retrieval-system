import os
import logging
import json

import numpy as np

from cores.extracting import FeatureExtractor

from configs import (
    OUT_FRAME_DIR,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    CLIP_VECTOR_DATA_PATH,
    DINO_MODEL,
    DINO_VECTOR_DATA_PATH,
    FRAME_DATA_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

if __name__ == '__main__':
    with open(FRAME_DATA_PATH) as f:
        frame_paths = json.load(f)
    extractor = FeatureExtractor(frame_paths, CLIP_MODEL, CLIP_PRETRAINED, DINO_MODEL)

    all_vectors, paths = extractor.extract_spatial_features()

    sorted_pairs = sorted(zip(paths, all_vectors), key=lambda x: x[0])
    paths, all_vectors = zip(*sorted_pairs)
    np.save(CLIP_VECTOR_DATA_PATH, np.stack(all_vectors))

    all_vectors, paths = extractor.extract_deep_visual_features()

    sorted_pairs = sorted(zip(paths, all_vectors), key=lambda x: x[0])
    paths, all_vectors = zip(*sorted_pairs)
    np.save(DINO_VECTOR_DATA_PATH, np.stack(all_vectors))

    with open(FRAME_DATA_PATH, 'w') as f:
        json.dump(list(paths), f, indent=2)
