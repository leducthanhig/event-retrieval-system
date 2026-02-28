# Root folder for all generated/consumed backend data artifacts.
DATA_ROOT_DIR = 'data'

# Input video directory for offline preprocessing and local video serving.
VIDEO_DIR = f"{DATA_ROOT_DIR}/videos"
# Output directory for sampled keyframes, mounted by the API as static images.
OUT_FRAME_DIR = f"{DATA_ROOT_DIR}/keyframes"
# JSON file storing per-video metadata (path, fps, shot boundaries).
VIDEO_METADATA_PATH = f"{DATA_ROOT_DIR}/metadata.json"

# JSON file storing sorted keyframe paths used by extraction/retrieval.
FRAME_DATA_PATH = f"{DATA_ROOT_DIR}/frame_data.json"

# Default single-model CLIP backbone for feature extraction scripts.
CLIP_MODEL = 'ViT-L-16-SigLIP-256'
# Default pretrained weights tag paired with CLIP_MODEL.
CLIP_PRETRAINED = 'webli'
# Output .npy path for CLIP feature vectors produced by extract.py.
CLIP_VECTOR_DATA_PATH = f"{DATA_ROOT_DIR}/vectors_{CLIP_MODEL}_{CLIP_PRETRAINED}.npy"

# DINO model id used for deep visual feature extraction and image search.
DINO_MODEL = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
# Output .npy path for DINO feature vectors produced by extract.py.
DINO_VECTOR_DATA_PATH = f"{DATA_ROOT_DIR}/vectors_{DINO_MODEL.replace('/', '-')}.npy"

# Whisper transcription JSON consumed by indexing and L25 sampling flow.
WHISPER_OUTPUT_PATH = 'data/whisper.json'

# Legacy single-model CLIP index path (multi-model indexing derives paths from CLIP_MODELS).
CLIP_INDEX_SAVE_PATH = f"{DATA_ROOT_DIR}/index_{CLIP_MODEL}_{CLIP_PRETRAINED}.bin"
# Faiss index path for DINO vectors used by backend image retrieval.
DINO_INDEX_SAVE_PATH = f"{DATA_ROOT_DIR}/index_{DINO_MODEL.replace('/', '-')}.bin"

# Directory containing per-video media-info JSON files for metadata indexing.
MEDIA_INFO_DIR = 'data/media-info'

# Environment file loaded by backend app and indexing scripts.
DOT_ENV_FILE = 'backend/.env'

# Elasticsearch endpoint for indexing and online text retrieval.
ELASTIC_HOST = 'http://localhost:9200/'
# Elasticsearch index name for media metadata (title/description/keywords).
MEDIA_INFO_INDEX_NAME = 'media-info'
# Elasticsearch index name for ASR transcription segments.
TRANSCRIPTION_INDEX_NAME = 'l25-transcription'

# URL path prefix used when exposing keyframe images from OUT_FRAME_DIR.
STATIC_IMAGE_PATH = 'images'
# URL path prefix used by backend endpoint that serves/redirects video files.
STATIC_VIDEO_PATH = 'videos'

# CLIP model list used for multi-model text retrieval and index loading.
CLIP_MODELS = [
    ('ViT-B-16-SigLIP2-384', 'webli'),
    # ('ViT-L-16-SigLIP-256', 'webli'),
    # ('ViT-L-14-quickgelu', 'dfn2b'),
]
# Default CLIP selection used when clients do not specify a model.
DEFAULT_CLIP_MODEL = {
    'name': CLIP_MODELS[0][0],
    'pretrained': CLIP_MODELS[0][1],
}
