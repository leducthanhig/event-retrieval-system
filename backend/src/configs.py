DATA_ROOT_DIR = 'data'

INP_VIDEO_DIR = f"{DATA_ROOT_DIR}/videos"
OUT_FRAME_DIR = f"{DATA_ROOT_DIR}/keyframes"
VIDEO_METADATA_PATH = f"{DATA_ROOT_DIR}/metadata.json"

PROCESSED_FRAME_DATA_PATH = f"{DATA_ROOT_DIR}/processed_frame_data.json"

CLIP_MODEL = 'ViT-L-16-SigLIP-256'
CLIP_PRETRAINED = 'webli'
CLIP_VECTOR_DATA_PATH = f"{DATA_ROOT_DIR}/vectors_{CLIP_MODEL}_{CLIP_PRETRAINED}.npy"

DINO_MODEL = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
DINO_VECTOR_DATA_PATH = f"{DATA_ROOT_DIR}/vectors_{DINO_MODEL.replace('/', '-')}.npy"

FAISS_PRESET = 'high_accuracy'
CLIP_INDEX_SAVE_PATH = f"{DATA_ROOT_DIR}/index_{CLIP_MODEL}_{CLIP_PRETRAINED}.bin"
DINO_INDEX_SAVE_PATH = f"{DATA_ROOT_DIR}/index_{DINO_MODEL.replace('/', '-')}.bin"

STATIC_IMAGE_PATH = 'images'
STATIC_VIDEO_PATH = 'videos'

CLIP_MODELS = [
    ('ViT-L-16-SigLIP-256', 'webli'),
    ('ViT-L-14-quickgelu', 'dfn2b'),
]
DEFAULT_CLIP_MODEL = {
    'name': CLIP_MODELS[0][0],
    'pretrained': CLIP_MODELS[0][1],
}
