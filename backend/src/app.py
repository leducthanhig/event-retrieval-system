import os
import time
import logging
import json
import pickle
from typing import Literal, Annotated
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
    STATIC_IMAGE_PATH,
    STATIC_VIDEO_PATH,
    MODELS,
    DEFAULT_MODEL,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

class SearchModel(BaseModel):
    model_name: str
    pretrained: str

class Shot(BaseModel):
    video_id: str
    shot_id: str
    thumbnail: str
    score: float

class SearchResponse(BaseModel):
    took: float
    found: int
    results: list[Shot]

class ShotResponse(BaseModel):
    video_path: str
    start: float
    end: float

class App(FastAPI):
    def __init__(self,
                 origins: list[str],
                 mount_paths: list[tuple[str, str]],
                 video_metadata_path: str,
                 models: list[tuple[str, str]]):
        super().__init__()
        self.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.init_retrievers(models)
        self.load_metadata(video_metadata_path)
        self.mount_dirs(mount_paths)
        self.init_routes()

    def mount_dirs(self, mount_paths: list[tuple[str, str]]):
        for path, dir in mount_paths:
            self.mount(path, StaticFiles(directory=dir), name=os.path.basename(path))

    def init_retrievers(self, models: list[tuple[str, str]]):
        self.retrievers: defaultdict[str, defaultdict[str, Retriever]] = defaultdict(defaultdict)
        for model, pretrained in models:
            with open(f'data/vectors_{model}_{pretrained}.pkl', 'rb') as f:
                metadata = pickle.load(f)['paths']

            self.retrievers[pretrained][model] = Retriever(f'data/index_{model}_{pretrained}.bin',
                                                           metadata,
                                                           model,
                                                           pretrained)

    def load_metadata(self, video_metadata_path: str):
        with open(video_metadata_path) as f:
            self.metadata: dict[str, dict[str, str | list[int]]] = json.load(f)

    def init_routes(self):
        @self.post("/search")
        async def search(q: str,
                         models: Annotated[SearchModel | list[SearchModel], Body()] = DEFAULT_MODEL,
                         weights: Annotated[list[float], Body()] = None,
                         pooling_method: Annotated[Literal['avg', 'max'], Body()] = 'max',
                         top: int = 10) -> SearchResponse:
            if not isinstance(models, list):
                models = [models]

            # Validate parameters
            if len(models) > 1:
                if not weights:
                    msg = 'Multi-model mode is used but no weights are provided'
                    logger.error(msg)
                    raise RuntimeError(msg)
                if len(models) != len(weights):
                    msg = 'Number of given `model` and `weights` does not match'
                    logger.error(msg)
                    raise RuntimeError(msg)

            # Perform search
            start = time.time()
            all_results = []
            for model in models:
                model = model.model_dump()
                model_name = model['model_name']
                pretrained = model['pretrained']
                retriever = self.retrievers[pretrained][model_name]
                all_results.append(
                    retriever.search(q, pooling_method=pooling_method, k=top))
            if len(all_results) > 1:
                results = Retriever.combine_results(all_results, weights)
            else:
                results = all_results[0]
            took = time.time() - start

            # Post-process file paths
            for res in results:
                res['thumbnail'] = res['thumbnail'].replace(OUT_FRAME_DIR, STATIC_IMAGE_PATH)

            return SearchResponse(took=took, found=len(results), results=results)

        @self.get('/shots/{video_id}/{shot_id}')
        def get_shot_timestamps(video_id: str, shot_id: str) -> ShotResponse:
            try:
                video_metadata = self.metadata[video_id]
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

origins = [
    "http://localhost:5173",
]

mount_paths = [
    (f"/{STATIC_IMAGE_PATH}", OUT_FRAME_DIR),
    (f"/{STATIC_VIDEO_PATH}", INP_VIDEO_DIR),
]

app = App(origins, mount_paths, VIDEO_METADATA_PATH, MODELS)
