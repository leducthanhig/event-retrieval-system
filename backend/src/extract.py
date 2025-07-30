import os
import logging
import pickle
import json

from cores.sampling import FrameSampler
from cores.extracting import FeatureExtractor

from configs import (
    INP_VIDEO_DIR,
    OUT_FRAME_DIR,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    VECTOR_DATA,
    OBJECT_DATA,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

if __name__ == '__main__':
    # Sample frames
    sampler = FrameSampler(INP_VIDEO_DIR, OUT_FRAME_DIR)
    sampler.run()

    # Extract features
    image_paths = [os.path.join(dirpath, filename)
                   for dirpath, _, filenames in os.walk(OUT_FRAME_DIR)
                   for filename in filenames
                   if os.path.splitext(filename)[0].endswith('_selected')]
    extractor = FeatureExtractor(image_paths, CLIP_MODEL, CLIP_PRETRAINED)

    all_vectors, paths = extractor.extract_spatial_features()
    vector_data = dict(all_vectors=all_vectors, paths=paths)
    with open(VECTOR_DATA, 'wb') as f:
        pickle.dump(vector_data, f)

    all_objects, paths = extractor.extract_object_features()
    object_data = dict(paths=paths, all_objects=all_objects)
    with open(OBJECT_DATA, 'w') as f:
        json.dump(object_data, f, indent=2)
