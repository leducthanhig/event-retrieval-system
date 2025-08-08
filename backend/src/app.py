import os
import time
import logging
import json
import pickle
from typing import Literal
from collections import defaultdict

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

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
    name: str
    pretrained: str

class SearchBody(BaseModel):
    models: SearchModel | list[SearchModel] = SearchModel(**DEFAULT_MODEL)
    weights: list[float] | None = None
    pooling_method: Literal['avg', 'max'] = 'max'

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

class RewriteRequest(BaseModel):
    text: str
    model: Literal['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.0-flash'] = 'gemini-2.5-flash-lite'
    clip_model: SearchModel = SearchModel(**DEFAULT_MODEL)
    thinking: bool = False

class RewriteResponse(BaseModel):
    rewritten_query: str

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

        self.genai_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    def mount_dirs(self, mount_paths: list[tuple[str, str]]):
        """Mounts directories to specific paths for serving static files."""
        for path, dir in mount_paths:
            self.mount(path, StaticFiles(directory=dir), name=os.path.basename(path))

    def init_retrievers(self, models: list[tuple[str, str]]):
        """Initializes retrievers for each model and pretrained weights."""
        self.retrievers: defaultdict[str, defaultdict[str, Retriever]] = defaultdict(defaultdict)
        for model, pretrained in models:
            with open(f'data/vectors_{model}_{pretrained}.pkl', 'rb') as f:
                metadata = pickle.load(f)['paths']

            self.retrievers[pretrained][model] = Retriever(f'data/index_{model}_{pretrained}.bin',
                                                           metadata,
                                                           model,
                                                           pretrained)

    def load_metadata(self, video_metadata_path: str):
        """Loads video metadata from a JSON file."""
        with open(video_metadata_path) as f:
            self.metadata: dict[str, dict[str, str | list[int]]] = json.load(f)

    def init_routes(self):
        """Initializes API routes for the application."""
        @self.post("/search")
        async def search(q: str, body: SearchBody, top: int = 10) -> SearchResponse:
            """Searches for relevant video shots based on the query."""
            models = body.models
            if not isinstance(models, list):
                models = [models]

            # Validate parameters
            if len(models) > 1:
                if not body.weights:
                    msg = 'Multi-model mode is used but no weights are provided'
                    logger.error(msg)
                    raise RuntimeError(msg)
                if len(models) != len(body.weights):
                    msg = 'Number of given `model` and `weights` does not match'
                    logger.error(msg)
                    raise RuntimeError(msg)

            # Perform search
            start = time.time()
            all_results = []
            for model in models:
                retriever = self.retrievers[model.pretrained][model.name]
                all_results.append(
                    retriever.search(q, pooling_method=body.pooling_method, k=top))
            if len(all_results) > 1:
                results = Retriever.combine_results(all_results, body.weights)
            else:
                results = all_results[0]
            took = time.time() - start

            # Post-process file paths
            for res in results:
                res['thumbnail'] = res['thumbnail'].replace(OUT_FRAME_DIR, STATIC_IMAGE_PATH)

            return SearchResponse(took=took, found=len(results), results=results)

        @self.get('/shots/{video_id}/{shot_id}')
        async def get_shot_timestamps(video_id: str, shot_id: str) -> ShotResponse:
            """Retrieves the start and end timestamps for a specific video shot."""
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

        @self.post('/rewrite')
        async def rewrite(req: RewriteRequest) -> RewriteResponse:
            """Rewrites a query to make it more descriptive for CLIP models."""
            prompt = f"Rewrite the query '{req.text}' in English to be more descriptive " \
                     f"for CLIP {req.clip_model.name} trained on {req.clip_model.pretrained}. " \
                     f"Use Google Search if needed. " \
                     f"Just return the rewritten query prefixed with 'Rewritten query:'."
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]

            tools = [
                types.Tool(googleSearch=types.GoogleSearch()),
            ]

            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1 if req.thinking else 0),
                tools=tools,
            )

            response = self.genai_client.models.generate_content(
                model=req.model,
                contents=contents,
                config=generate_content_config,
            )

            rewritten = ''
            for part in response.candidates[0].content.parts:
                if "Rewritten query:" in part.text:
                    rewritten = part.text.split("Rewritten query:")[1].strip()

            return RewriteResponse(rewritten_query=rewritten)

origins = [
    "http://localhost:5173",
]

mount_paths = [
    (f"/{STATIC_IMAGE_PATH}", OUT_FRAME_DIR),
    (f"/{STATIC_VIDEO_PATH}", INP_VIDEO_DIR),
]

app = App(origins, mount_paths, VIDEO_METADATA_PATH, MODELS)
