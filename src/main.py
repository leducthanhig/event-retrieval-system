import pickle
import json

from cores.sampling import FrameSampler
from cores.extracting import FeatureExtractor
from cores.retrieving import Retriever

from configs import (
    VIDEO_DIR,
    FRAME_DIR,
    SHOT_INFOS_PATH,
    FEATURES_PATH,
    METADATA_PATH,
    INDEX_PATH
)

if __name__ == '__main__':
    sampler = FrameSampler(VIDEO_DIR, FRAME_DIR, batch_size=16)
    shot_infos = sampler.run()

    shot_infos.to_json(SHOT_INFOS_PATH, 'records', indent=2)

    with open(SHOT_INFOS_PATH) as f:
        shot_infos = json.load(f)
    extractor = FeatureExtractor(FRAME_DIR, shot_infos)
    all_features, metadata = extractor.run()

    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(all_features, f)
    metadata.to_json(METADATA_PATH, 'records', indent=2)

    with open(FEATURES_PATH, 'rb') as f:
        all_features = pickle.load(f)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    retriever = Retriever()
    retriever.create_index(all_features, metadata)
    retriever.save(INDEX_PATH, METADATA_PATH)
