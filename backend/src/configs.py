INP_VIDEO_DIR = 'data/videos'
OUT_FRAME_DIR = 'data/keyframes'
VIDEO_METADATA_PATH = 'data/metadata.json'

CLIP_MODEL = 'ViT-L-16-SigLIP-256'
CLIP_PRETRAINED = 'webli'

VECTOR_DATA_PATH = f'data/vectors_{CLIP_MODEL}_{CLIP_PRETRAINED}.pkl'

FAISS_PRESET = 'high_accuracy'
FAISS_SAVE_PATH = f'data/index_{CLIP_MODEL}_{CLIP_PRETRAINED}.bin'

STATIC_IMAGE_PATH = 'images'
STATIC_VIDEO_PATH = 'videos'

MODELS = [
    ('ViT-L-16-SigLIP-256', 'webli'),
    ('ViT-L-14-quickgelu', 'dfn2b'),
]

DEFAULT_MODEL = {
    'name': MODELS[0][0],
    'pretrained': MODELS[0][1],
}
