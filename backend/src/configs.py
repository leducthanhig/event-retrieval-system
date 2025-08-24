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

WHISPER_OUTPUT_PATH = 'data/whisper.json'

CLIP_INDEX_SAVE_PATH = f"{DATA_ROOT_DIR}/index_{CLIP_MODEL}_{CLIP_PRETRAINED}.bin"
DINO_INDEX_SAVE_PATH = f"{DATA_ROOT_DIR}/index_{DINO_MODEL.replace('/', '-')}.bin"

MEDIA_INFO_DIR = 'data/media-info'

DOT_ENV_FILE = 'backend/.env'

ELASTIC_HOST = 'http://localhost:9200/'
MEDIA_INFO_INDEX_NAME = 'media-info'
TRANSCRIPTION_INDEX_NAME = 'l25-transcription'

STATIC_IMAGE_PATH = 'images'
STATIC_VIDEO_PATH = 'https://huggingface.co/datasets/leducthanhig/hcmc-aic-2025/resolve/main/videos'

CLIP_MODELS = [
    ('ViT-B-16-SigLIP2-384', 'webli'),
    ('ViT-L-16-SigLIP-256', 'webli'),
    ('ViT-L-14-quickgelu', 'dfn2b'),
]
DEFAULT_CLIP_MODEL = {
    'name': CLIP_MODELS[0][0],
    'pretrained': CLIP_MODELS[0][1],
}
