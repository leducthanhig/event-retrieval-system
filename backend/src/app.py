import os
import time
import logging
import json
import pickle
import tempfile
from typing import List, Dict, Literal, Annotated

import ffmpeg
from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cores.retrieving import Retriever
from utils import get_avg_fps

from configs import (
    OUT_FRAME_DIR,
    VIDEO_METADATA_PATH,
    CLIP_MODEL,
    CLIP_PRETRAINED,
    VECTOR_DATA_PATH,
    FAISS_SAVE_PATH,
    ELASTIC_HOST,
    ELASTIC_INDEX_NAME
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

def init_retriever():
    with open(VECTOR_DATA_PATH, 'rb') as f:
        metadata = pickle.load(f)['paths']

    return Retriever(FAISS_SAVE_PATH,
                     metadata,
                     CLIP_MODEL,
                     CLIP_PRETRAINED,
                     ELASTIC_HOST,
                     os.environ['ES_LOCAL_API_KEY'],
                     ELASTIC_INDEX_NAME)

def load_metadata() -> Dict[str, Dict[str, str | List]]:
    with open(VIDEO_METADATA_PATH) as f:
        metadata = json.load(f)
    return metadata

logger = logging.getLogger(__name__)

app = FastAPI()
retriever = init_retriever()
metadata = load_metadata()

app.mount("/images", StaticFiles(directory=OUT_FRAME_DIR), name="images")

class ObjectBBox(BaseModel):
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class ObjectCounting(BaseModel):
    label: str
    count: int

class Shot(BaseModel):
    video_id: str
    shot_id: str
    thumbnail: str
    score: float

class SearchResponse(BaseModel):
    took: float
    found: int
    results: List[Shot]

@app.get("/search")
async def search(q: str,
                 bboxes: List[ObjectBBox] = None,
                 counts: List[ObjectCounting] = None,
                 weights: Annotated[list[float], Body()] = None,
                 pooling_method: Annotated[Literal['avg', 'max'], Body()] = 'max',
                 top: int = 10) -> SearchResponse:
    bboxes_dict = None
    if bboxes:
        bboxes_dict = [bbox.model_dump() for bbox in bboxes]

    counts_tuple = None
    if counts:
        counts_tuple = [tuple(count.model_dump().values()) for count in counts]

    start = time.time()
    results = retriever.search(q, counts_tuple, bboxes_dict, weights, pooling_method, top)
    took = time.time() - start

    for res in results:
        res['thumbnail'] = res['thumbnail'].replace(OUT_FRAME_DIR, 'images')

    return {
        'took': took,
        'found': len(results),
        'results': results
    }

@app.get('/videos/{video_id}/{shot_id}')
def get_shot(video_id: str, shot_id: str):
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
    if start < len(shots) - 1:
        end = (shots[idx + 1] - 1) / fps
        video_stream, _ = (
            ffmpeg
            .input(path, ss=start, to=end)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, capture_stderr=True)
        )
    else:
        video_stream, _ = (
            ffmpeg
            .input(path, ss=start)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, capture_stderr=True)
        )

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_stream)

    response = FileResponse(temp_file.name)

    # Clean up the temporary file after sending
    @response.background
    def cleanup():
        os.remove(temp_file.name)

    return response
