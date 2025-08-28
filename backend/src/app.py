import os
import time
import logging
import json
from typing import Literal

from fastapi import FastAPI, UploadFile, File, Query, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from cores.retrieving import Retriever

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
    DOT_ENV_FILE,
    ELASTIC_HOST,
    MEDIA_INFO_INDEX_NAME,
    TRANSCRIPTION_INDEX_NAME,
)

# Load environment variables from the .env file
load_dotenv(DOT_ENV_FILE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

class SearchModel(BaseModel):
    name: str
    pretrained: str

class SearchRequest(BaseModel):
    text_query: str | None = None
    models: list[SearchModel] | None = None
    model_weights: list[float] | None = None
    image_query: UploadFile | None = None
    transcription_query: str | None = None
    metadata_query: str | None = None
    modality_weights: dict[str, float] | None = None
    pooling_method: Literal['avg', 'max'] = 'max'
    top: int = 10

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
    fps: float

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
                 **kwargs,
        ):
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
                                   metadata,
                                   ELASTIC_HOST,
                                   os.environ['ES_LOCAL_API_KEY'],
                                   MEDIA_INFO_INDEX_NAME,
                                   TRANSCRIPTION_INDEX_NAME)

    def load_metadata(self):
        """Loads video metadata from a JSON file."""
        with open(VIDEO_METADATA_PATH) as f:
            self.metadata: dict[str, dict[str, str | list[int]]] = json.load(f)

    def init_routes(self):
        """Initializes API routes for the application."""
        @self.post('/search')
        async def search(req: SearchRequest = Depends(App._parse_search_request)) -> SearchResponse:
            start = time.time()
            all_results, query_types = [], []

            if req.text_query:
                text_shots = self._text_search(req.text_query,
                                               req.models,
                                               req.model_weights,
                                               req.top,
                                               req.pooling_method)
                all_results.append(text_shots)
                query_types.append('text')

            if req.image_query:
                image_shots = self._image_search(req.image_query, req.top, req.pooling_method)
                all_results.append(image_shots)
                query_types.append('image')

            if req.transcription_query:
                transcription_shots = self._transcription_search(req.transcription_query, req.top)
                all_results.append(transcription_shots)
                query_types.append('transcription')

            if req.metadata_query:
                metadata_shots = self._metadata_search(req.metadata_query, req.top)
                all_results.append(metadata_shots)
                query_types.append('metadata')

            if not all_results:
                return SearchResponse(took=0.0, found=0, results=[])

            if len(all_results) > 1:
                # Modality fusion weights
                weights = []
                if req.modality_weights:
                    w = [float(req.modality_weights.get(name, 0.0)) for name in query_types]
                    s = sum(w)
                    if s > 0:
                        weights = [wi / s for wi in w]

                if not weights:
                    weights = [1.0 / len(all_results)] * len(all_results)

                fused = Retriever.combine_shot_results(all_results, weights)
                if req.top > 0:
                    fused = fused[:req.top]

                final_results = fused
            else:
                final_results = all_results[0]

            # Map paths
            for shot in final_results:
                shot['thumbnail'] = shot['thumbnail'].replace(OUT_FRAME_DIR, STATIC_IMAGE_PATH)

            took = time.time() - start
            return SearchResponse(took=took, found=len(final_results), results=final_results)

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
            fps = video_metadata['fps']
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

    def _text_search(self,
                     query: str,
                     models: list[SearchModel],
                     model_weights: list[float] | None,
                     top: int,
                     pooling_method: Literal['avg','max'],
        ) -> list[dict]:
        """Helper for text search."""
        # Per-model frame search
        per_model_frames = [
            self.retriever.search_by_text(query, m.name, m.pretrained, top)
            for m in models
        ]
        # Fuse frames across models (if multi)
        if len(per_model_frames) > 1:
            if not model_weights or len(model_weights) != len(per_model_frames):
                raise RuntimeError("Model weights missing or length mismatch for TEXT multi-model.")
            fused_frames = Retriever.combine_frame_results(per_model_frames, model_weights)
            if top > 0:
                fused_frames = fused_frames[:top]
        else:
            fused_frames = per_model_frames[0]

        return Retriever.combine_frames(fused_frames, pooling_method)

    def _image_search(self,
                      image_file: UploadFile,
                      top: int,
                      pooling_method: Literal['avg','max'],
        ) -> list[dict]:
        """Helper for image search."""
        img = Image.open(image_file.file)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        index_name = os.path.splitext(os.path.basename(DINO_INDEX_SAVE_PATH))[0].removeprefix('index_')
        frames = self.retriever.search_by_image(img, index_name, top)
        return Retriever.combine_frames(frames, pooling_method)

    def _metadata_search(self, query: str, top: int) -> list[dict]:
        """Helper for metadata search."""
        results = self.retriever.media_info_search(query, top)

        final_results = []
        for video_id, score in results:
            # use the selected keyframe of the middle shot as thumbnail
            selected_shot = len(self.metadata[video_id]['shots']) // 2
            shot_id = f"S{selected_shot:05}"
            frame_dir = os.path.join(OUT_FRAME_DIR, video_id, shot_id)
            frames = os.listdir(frame_dir)
            selected_frame = [frame for frame in frames
                              if os.path.splitext(frame)[0].endswith('_selected')]

            if selected_frame:
                thumbnail = os.path.join(frame_dir, selected_frame[0])
            else:
                thumbnail = ''

            final_results.append({
                'video_id': video_id,
                'shot_id': shot_id,
                'thumbnail': thumbnail,
                'score': score,
            })

        return final_results

    def _transcription_search(self, query: str, top: int) -> list[dict]:
        """Helper for transcription search."""
        results = self.retriever.transcription_search(query, top)

        final_results = []
        for info, score in results:
            video_id = info['video_id']
            shot_id = f"S{info['id']:05}"

            frame_dir = os.path.join(OUT_FRAME_DIR, video_id, shot_id)
            frames = os.listdir(frame_dir)
            selected_frame = [frame for frame in frames
                              if os.path.splitext(frame)[0].endswith('_selected')]

            if selected_frame:
                thumbnail = os.path.join(frame_dir, selected_frame[0])
            else:
                thumbnail = ''

            final_results.append({
                'video_id': video_id,
                'shot_id': shot_id,
                'thumbnail': thumbnail,
                'score': score,
            })

        return final_results

    @staticmethod
    def _parse_search_request(
        text_query: str | None = Form(None),
        models: str | None = Form(None),
        model_weights: str | None = Form(None),
        image_query: UploadFile | None = File(None),
        transcription_query: str | None = Form(None),
        metadata_query: str | None = Form(None),
        modality_weights: str | None = Form(None),
        pooling_method: Literal['avg', 'max'] = Form('max'),
        top: int = Query(10),
    ):
        """Parse the request body to a `BaseModel` class/subclass instance."""
        parsed_models = None
        if models:
            raw = json.loads(models)
            if isinstance(raw, list):
                parsed_models = [SearchModel(**m) for m in raw]
            else:
                parsed_models = [SearchModel(**raw)]
        if not parsed_models:
            parsed_models = [SearchModel(**DEFAULT_CLIP_MODEL)]

        parsed_model_weights = json.loads(model_weights) if model_weights else None
        parsed_modality_weights = json.loads(modality_weights) if modality_weights else None

        return SearchRequest(text_query=text_query,
                             models=parsed_models,
                             model_weights=parsed_model_weights,
                             image_query=image_query,
                             transcription_query=transcription_query,
                             metadata_query=metadata_query,
                             modality_weights=parsed_modality_weights,
                             pooling_method=pooling_method,
                             top=top)

origins = [
    "http://localhost:5173",
]

mount_paths = [
    (f"/{STATIC_IMAGE_PATH}", OUT_FRAME_DIR),
    (f"/{STATIC_VIDEO_PATH}", INP_VIDEO_DIR),
]

app = App(origins, mount_paths)
