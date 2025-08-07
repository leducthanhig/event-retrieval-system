import time
import logging
import json
import pickle
from typing import List, Dict, Literal, Annotated
from collections import defaultdict

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cores.retrieving import Retriever
from utils import get_avg_fps

from configs import (
    INP_VIDEO_DIR,
    OUT_FRAME_DIR,
    VIDEO_METADATA_PATH,
)

MODELS = [
    ('ViT-L-16-SigLIP-256', 'webli'),
    ('ViT-L-14-quickgelu', 'dfn2b'),
]

STATIC_IMAGE_PATH = 'images'
STATIC_VIDEO_PATH = 'videos'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

def init_retrievers(models: list[tuple[str, str]]) -> defaultdict[str, defaultdict[str, Retriever]]:
    retrievers = defaultdict(defaultdict)
    for model, pretrained in models:
        with open(f'data/vectors_{model}_{pretrained}.pkl', 'rb') as f:
            metadata = pickle.load(f)['paths']

        retrievers[pretrained][model] = Retriever(f'data/index_{model}_{pretrained}.bin',
                                                  metadata,
                                                  model,
                                                  pretrained)
    return retrievers

def load_metadata(video_metadata_path: str) -> Dict[str, Dict[str, str | List]]:
    with open(video_metadata_path) as f:
        metadata = json.load(f)
    return metadata

logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(f"/{STATIC_IMAGE_PATH}", StaticFiles(directory=OUT_FRAME_DIR), name="images")
app.mount(f"/{STATIC_VIDEO_PATH}", StaticFiles(directory=INP_VIDEO_DIR), name="videos")

retrievers = init_retrievers(MODELS)
metadata = load_metadata(VIDEO_METADATA_PATH)

class Shot(BaseModel):
    video_id: str
    shot_id: str
    thumbnail: str
    score: float

class SearchResponse(BaseModel):
    took: float
    found: int
    results: List[Shot]

class ShotResponse(BaseModel):
    video_path: str
    start: float
    end: float

@app.post("/search")
async def search(q: str,
                 pooling_method: Annotated[Literal['avg', 'max'], Body(embed=True)] = 'max',
                 model: Annotated[str, Body(embed=True)] = MODELS[0][0],
                 pretrained: Annotated[str, Body(embed=True)] = MODELS[0][1],
                 top: int = 10) -> SearchResponse:
    start = time.time()
    results = retrievers[pretrained][model].search(q, pooling_method=pooling_method, k=top)
    took = time.time() - start

    for res in results:
        res['thumbnail'] = res['thumbnail'].replace(OUT_FRAME_DIR, STATIC_IMAGE_PATH)

    return SearchResponse(took=took, found=len(results), results=results)

@app.get('/shots/{video_id}/{shot_id}')
def get_shot_timestamps(video_id: str, shot_id: str) -> ShotResponse:
    try:
        video_metadata = metadata[video_id]
    except KeyError:
        msg = f"Invalid video id: {video_id}"
        logger.error(msg)
        raise RuntimeError(msg)

    shots = video_metadata['shots']
    idx = int(shot_id[1:])
    if idx >= len(shots):
        msg = f"Invalid shot id: {shot_id}"
        logger.error(msg)
        raise RuntimeError(msg)

    path = video_metadata['path']
    fps = get_avg_fps(path)
    start = shots[idx] / fps
    end = None if idx == len(shots) - 1 else (shots[idx + 1] - 1) / fps
    public_path = path.replace(INP_VIDEO_DIR, STATIC_VIDEO_PATH)

    return ShotResponse(video_path=public_path, start=start, end=end)
