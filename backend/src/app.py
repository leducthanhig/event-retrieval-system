import os
import time
import logging
import json
from typing import Literal, Annotated

from fastapi import FastAPI, Form, UploadFile, Depends, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from cores.retrieving import Retriever
from utils import get_avg_fps

from configs import (
    INP_VIDEO_DIR,
    OUT_FRAME_DIR,
    VIDEO_METADATA_PATH,
    PROCESSED_FRAME_DATA_PATH,
    DATA_ROOT_DIR,
    STATIC_IMAGE_PATH,
    STATIC_VIDEO_PATH,
    CLIP_MODELS,
    DEFAULT_CLIP_MODEL,
    DINO_MODEL,
    DINO_INDEX_SAVE_PATH,
)

# Load environment variables from the .env file
load_dotenv('backend/.env')

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
    models: SearchModel | list[SearchModel] = SearchModel(**DEFAULT_CLIP_MODEL)
    clip_weights: list[float] | None = None
    weights: list[float] | None = None
    image_query: UploadFile | None = None
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
    clip_model: SearchModel = SearchModel(**DEFAULT_CLIP_MODEL)
    thinking: bool = False

class RewriteResponse(BaseModel):
    rewritten_query: str

class App(FastAPI):
    def __init__(self,
                 origins: list[str],
                 mount_paths: list[tuple[str, str]],
                 **kwargs):
        super().__init__(**kwargs)

        self.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.init_retriever()
        self.load_metadata()
        self.mount_dirs(mount_paths)
        self.init_routes()

        self.genai_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    def mount_dirs(self, mount_paths: list[tuple[str, str]]):
        """Mounts directories to specific paths for serving static files."""
        for path, dir in mount_paths:
            self.mount(path, StaticFiles(directory=dir), name=os.path.basename(path))

    def init_retriever(self):
        """Initializes retriever."""
        clip_index_path = [os.path.join(DATA_ROOT_DIR, f"index_{m}_{p}.bin")
                           for m, p in CLIP_MODELS]
        model, pretrained = zip(*CLIP_MODELS)
        with open(PROCESSED_FRAME_DATA_PATH) as f:
            metadata = json.load(f)
        self.retriever = Retriever(clip_index_path,
                                   DINO_INDEX_SAVE_PATH,
                                   list(model),
                                   list(pretrained),
                                   DINO_MODEL,
                                   metadata)

    def load_metadata(self):
        """Loads video metadata from a JSON file."""
        with open(VIDEO_METADATA_PATH) as f:
            self.metadata: dict[str, dict[str, str | list[int]]] = json.load(f)

    def init_routes(self):
        """Initializes API routes for the application."""
        @self.post("/search")
        async def search(
            q: str,
            body: SearchBody = Depends(App.parse_search_body),
            top: int = 10) -> SearchResponse:
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

            # Search by text
            all_clip_results = []
            for model in models:
                all_clip_results.append(self.retriever.search_by_text(q,
                                                                      model.name,
                                                                      model.pretrained,
                                                                      top))

            if len(all_clip_results) > 1:
                all_results.append(
                    Retriever.combine_frame_results(all_clip_results, body.clip_weights))
            else:
                all_results.append(all_clip_results[0])

            # Search by image
            if body.image_query:
                image_query = Image.open(body.image_query.file)
                index_name = os.path.splitext(
                    os.path.basename(DINO_INDEX_SAVE_PATH))[0].removeprefix('index_')
                all_results.append(self.retriever.search_by_image(image_query, index_name, top))

            # Combine frames
            all_combined_results = [Retriever.combine_frames(results, body.pooling_method)
                                    for results in all_results]

            # Combine results if needed
            if len(all_combined_results) > 1:
                final_results = Retriever.combine_shot_results(all_combined_results, body.weights)
            else:
                final_results = all_combined_results[0]

            took = time.time() - start

            # Post-process file paths
            for res in final_results:
                res['thumbnail'] = res['thumbnail'].replace(OUT_FRAME_DIR, STATIC_IMAGE_PATH)

            return SearchResponse(took=took, found=len(final_results), results=final_results)

        @self.post('/similar_search')
        async def similar_search(image_query: Annotated[UploadFile, File()],
                                 top: int = 10) -> SearchResponse:
            image_query = Image.open(image_query.file)
            index_name = os.path.splitext(
                os.path.basename(DINO_INDEX_SAVE_PATH))[0].removeprefix('index_')

            start = time.time()
            results = Retriever.combine_frames(
                self.retriever.search_by_image(image_query, index_name, top))
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
                     f"and aligned with web-style captions " \
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

    @staticmethod
    async def parse_search_body(
        models: str = Form(...),
        clip_weights: str | None = Form(None),
        weights: str | None = Form(None),
        pooling_method: Literal['avg', 'max'] = Form('max'),
        image_query: UploadFile | None = None,
    ) -> SearchBody:
        """Parses form data and reconstructs the SearchBody model."""
        models_parsed = json.loads(models)
        clip_weights_parsed = json.loads(clip_weights) if clip_weights else None
        weights_parsed = json.loads(weights) if weights else None

        # Convert models to SearchModel objects
        if isinstance(models_parsed, list):
            models_obj = [SearchModel(**m) if isinstance(m, dict) else m for m in models_parsed]
        else:
            models_obj = [SearchModel(**models_parsed) if isinstance(models_parsed, dict) else models_parsed]

        return SearchBody(
            models=models_obj,
            clip_weights=clip_weights_parsed,
            weights=weights_parsed,
            image_query=image_query,
            pooling_method=pooling_method
        )

origins = [
    "http://localhost:5173",
]

mount_paths = [
    (f"/{STATIC_IMAGE_PATH}", OUT_FRAME_DIR),
    (f"/{STATIC_VIDEO_PATH}", INP_VIDEO_DIR),
]

app = App(origins, mount_paths)
