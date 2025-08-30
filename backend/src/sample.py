import logging
import json

from cores.sampling import ShotDetector, FrameSampler

from configs import (
    VIDEO_DIR,
    VIDEO_METADATA_PATH,
    OUT_FRAME_DIR,
    FRAME_DATA_PATH,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

if __name__ == '__main__':
    detector = ShotDetector(VIDEO_DIR, VIDEO_METADATA_PATH)
    detector.run()

    with open(VIDEO_METADATA_PATH) as f:
        metadata = json.load(f)
    sampler = FrameSampler(metadata, OUT_FRAME_DIR)
    frame_paths = sampler.run()

    with open(FRAME_DATA_PATH, 'w') as f:
        json.dump(frame_paths, f, indent=2)
