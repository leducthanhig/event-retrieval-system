import os
import time
import logging
import json
import pickle
import tempfile
from typing import List, Dict, Literal, Annotated

import ffmpeg
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
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

app.mount("/images", StaticFiles(directory=OUT_FRAME_DIR), name="images")

retriever = init_retriever()
metadata = load_metadata()

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

@app.post("/search")
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
def get_shot(video_id: str,
             shot_id: str,
             background_tasks: BackgroundTasks) -> FileResponse:
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

    # Create a temporary file for the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file_path = temp_file.name

    # Use ffmpeg to cut the video and save it in MP4 format
    path = video_metadata['path']
    fps = get_avg_fps(path)
    start = shots[idx] / fps
    if idx < len(shots) - 1:
        end = (shots[idx + 1] - 1) / fps
        ffmpeg_cmd = ffmpeg.input(path, ss=start, to=end)
    else:
        ffmpeg_cmd = ffmpeg.input(path, ss=start)

    out, err = (
        ffmpeg_cmd
        .output(temp_file_path, vcodec='libx264', acodec='copy')
        .overwrite_output()
        .run()
    )

    # Add cleanup task to delete the temporary file after sending
    background_tasks.add_task(os.remove, temp_file_path)

    # Send the temporary file to the client
    return FileResponse(temp_file_path,
                        media_type='video/mp4',
                        filename=f"{video_id}_{shot_id}.mp4")
