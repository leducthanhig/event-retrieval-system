import logging

from cores.sampling import FrameSampler

from configs import (
    VIDEO_DIR,
    VIDEO_METADATA_PATH,
    OUT_FRAME_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

if __name__ == '__main__':
    sampler = FrameSampler(VIDEO_DIR, OUT_FRAME_DIR, VIDEO_METADATA_PATH)
    sampler.run()
