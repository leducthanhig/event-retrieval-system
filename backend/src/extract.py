import os
import logging
import pickle

from cores.sampling import FrameSampler
from cores.extracting import FeatureExtractor

from configs import (
    INP_VIDEO_DIR,
    VIDEO_METADATA_PATH,
    OUT_FRAME_DIR,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    VECTOR_DATA_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

if __name__ == '__main__':
    # Sample frames
    sampler = FrameSampler(INP_VIDEO_DIR, OUT_FRAME_DIR, VIDEO_METADATA_PATH)
    sampler.run()

    # Extract features
    image_paths = [os.path.join(dirpath, filename)
                   for dirpath, _, filenames in os.walk(OUT_FRAME_DIR)
                   for filename in filenames
                   if os.path.splitext(filename)[0].endswith('_selected')]
    extractor = FeatureExtractor(image_paths, CLIP_MODEL, CLIP_PRETRAINED)

    all_vectors, paths = extractor.extract_spatial_features()
    vector_data = dict(all_vectors=all_vectors, paths=paths)
    with open(VECTOR_DATA_PATH, 'wb') as f:
        pickle.dump(vector_data, f)
