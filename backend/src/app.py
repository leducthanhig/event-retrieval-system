import os
import time
import logging
import json
from typing import Literal, Annotated

from fastapi import FastAPI, UploadFile, File, Query, Form
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

class SearchBody(BaseModel):
    models: SearchModel | list[SearchModel] = SearchModel(**DEFAULT_CLIP_MODEL)
    weights: list[float] | None = None

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

class Video(BaseModel):
    video_id: str
    video_path: str
    score: float

class VideoResponse(BaseModel):
    took: float
    found: int
    results: list[Video]

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
        """
        @self.post("/search")
        async def search(q: str,
                         body: SearchBody,
                         top: int = 10,
                         pooling_method: Literal['avg', 'max'] = 'max') -> SearchResponse:
            #Searches for relevant video shots based on the query.
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
            # Perform search for each model
            for model in models:
                all_results.append(
                    self.retriever.search_by_text(q, model.name, model.pretrained, top))

            # Fuse results if needed
            if len(all_results) > 1:
                results = Retriever.combine_frame_results(all_results, body.weights)
            else:
                results = all_results[0]

            # Combine frames
            final_results = Retriever.combine_frames(results, pooling_method)

            took = time.time() - start

            # Post-process file paths
            for res in final_results:
                res['thumbnail'] = res['thumbnail'].replace(OUT_FRAME_DIR, STATIC_IMAGE_PATH)

            return SearchResponse(took=took, found=len(final_results), results=final_results)

        @self.post('/similar_search')
        async def similar_search(image_query: Annotated[UploadFile, File()],
                                 top: int = 10,
                                 pooling_method: Literal['avg', 'max'] = 'max') -> SearchResponse:
            image_query = Image.open(image_query.file)
            index_name = os.path.splitext(
                os.path.basename(DINO_INDEX_SAVE_PATH))[0].removeprefix('index_')

            start = time.time()
            results = Retriever.combine_frames(
                self.retriever.search_by_image(image_query, index_name, top), pooling_method)
            took = time.time() - start

            # Post-process file paths
            for res in results:
                res['thumbnail'] = res['thumbnail'].replace(OUT_FRAME_DIR, STATIC_IMAGE_PATH)

            return SearchResponse(took=took, found=len(results), results=results)
        
        @self.get('/video_search')
        async def video_search(text_query: str, top: int = 10) -> VideoResponse:
            start = time.time()

            results = self.retriever.media_info_search(text_query, top)

            final_results = []
            for video_id, score in results:
                path = self.metadata[video_id]['path'].replace(INP_VIDEO_DIR, STATIC_VIDEO_PATH)
                final_results.append(Video(video_id=video_id, video_path=path, score=score))

            took = time.time() - start

            return VideoResponse(took=took, found=len(final_results), results=final_results)
        """
            
        @self.post("/search_unified")
        async def search_unified(
            text_query: str | None = Form(None),
            models: str | None = Form(None),
            model_weights: str | None = Form(None),
            image_query: UploadFile | None = File(None),
            metadata_query: str | None = Form(None),
            modality_weights: str | None = Form(None),
            pooling_method: Literal['avg','max'] = Form('max'),
            top: int = Query(10),
        ) -> SearchResponse:
            start = time.time()

            # Parse models
            parsed_models: list[SearchModel] | None = None
            if models:
                raw = json.loads(models)
                if isinstance(raw, list):
                    parsed_models = [SearchModel(**m) if isinstance(m, dict) else m for m in raw]
                else:
                    parsed_models = [SearchModel(**raw) if isinstance(raw, dict) else raw]

            text_w = json.loads(model_weights) if model_weights else None
            mod_w = json.loads(modality_weights) if modality_weights else None

            # Branches
            used_lists, used_names = [], []

            if text_query:
                if not parsed_models:
                    parsed_models = [SearchModel(**DEFAULT_CLIP_MODEL)]
                text_shots = self._search_text_shots(text_query, parsed_models, text_w, top, pooling_method)
                used_lists.append(text_shots); used_names.append("text")

            if image_query:
                image_shots = self._search_image_shots(image_query, top, pooling_method)
                used_lists.append(image_shots); used_names.append("image")

            if metadata_query:
                metadata_shots = self._search_metadata_shots(metadata_query, top)
                used_lists.append(metadata_shots); used_names.append("metadata")

            if not used_lists:
                return SearchResponse(took=0.0, found=0, results=[])

            # Modality fusion weights
            if mod_w:
                w = [float(mod_w.get(name, 0.0)) for name in used_names]
                s = sum(w)
                weights_vec = [wi / s for wi in w] if s > 0 else [1.0/len(used_lists)] * len(used_lists)
            else:
                weights_vec = [1.0/len(used_lists)] * len(used_lists)

            fused = Retriever.combine_shot_results(used_lists, weights_vec)
            if top is not None and top > 0:
                fused = fused[:top]

            took = time.time() - start
            return SearchResponse(took=took, found=len(fused), results=fused)

        @self.get('/transcription_search')
        async def transcription_search(text_query: str, top: int = 10) -> SearchResponse:
            start = time.time()

            results = self.retriever.transcription_search(text_query, top)

            final_results = []
            for info, score in results:
                video_id = info['video_id']
                shot_id = f"S{info['id']:05}"
                frame_dir = os.path.join(OUT_FRAME_DIR, 'Videos_L25', 'video', video_id, shot_id)
                frames = os.listdir(frame_dir)
                selected_frame = [frame for frame in frames
                                  if os.path.splitext(frame)[0].endswith('_selected')]
                if selected_frame:
                    thumbnail = os.path.join(frame_dir, selected_frame[0])
                    thumbnail = thumbnail.replace(OUT_FRAME_DIR, STATIC_IMAGE_PATH)
                else:
                    thumbnail = ''
                final_results.append(Shot(video_id=video_id,
                                          shot_id=shot_id,
                                          thumbnail=thumbnail,
                                          score=score))

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
        
    # Helpers: return shot-level lists[dict]
    def _search_text_shots(self, text_query: str,
                        models: list[SearchModel],
                        model_weights: list[float] | None,
                        top: int,
                        pooling_method: Literal['avg','max']) -> list[dict]:
        # Per-model frame search
        per_model_frames = [
            self.retriever.search_by_text(text_query, m.name, m.pretrained, top)
            for m in models
        ]
        # Fuse frames across models (if multi)
        if len(per_model_frames) > 1:
            if not model_weights or len(model_weights) != len(per_model_frames):
                raise RuntimeError("Model weights missing or length mismatch for TEXT multi-model.")
            fused_frames = Retriever.combine_frame_results(per_model_frames, model_weights)
        else:
            fused_frames = per_model_frames[0]
        # Pool frames -> shots
        shots = Retriever.combine_frames(fused_frames, pooling_method)
        # Map paths
        for r in shots:
            r['thumbnail'] = r['thumbnail'].replace(OUT_FRAME_DIR, STATIC_IMAGE_PATH)
        return shots
    
    def _search_image_shots(self, image_file: UploadFile,
                            top: int,
                            pooling_method: Literal['avg','max']) -> list[dict]:
        img = Image.open(image_file.file)
        index_name = os.path.splitext(os.path.basename(DINO_INDEX_SAVE_PATH))[0].removeprefix('index_')
        frames = self.retriever.search_by_image(img, index_name, top)
        shots = Retriever.combine_frames(frames, pooling_method)
        for r in shots:
            r['thumbnail'] = r['thumbnail'].replace(OUT_FRAME_DIR, STATIC_IMAGE_PATH)
        return shots

    def _search_metadata_shots(self,
                            metadata_query: str,
                            top: int,
                            max_shots_per_video: int = 5) -> list[dict]:
        hits = self.retriever.full_text_search(metadata_query, top)
        if not hits:
            logger.info("metadata: no ES hits")
            return []

        expanded: list[dict] = []
        remaining = int(top) if top else 50
        logger.info(f"metadata: {len(hits)} video hits; cap={remaining}")

        # Dò 2 gốc: (1) theo config, (2) gốc thực tế .data/keyframe
        roots = [OUT_FRAME_DIR]
        alt = OUT_FRAME_DIR.replace("data/keyframes", ".data/keyframe")
        if alt not in roots:
            roots.append(alt)

        for video_id, vscore in hits:
            if remaining <= 0:
                break

            keyframe_dir = None
            # Tìm <root>/<collection>/video/<video_id>
            for root in roots:
                try:
                    if not os.path.isdir(root):
                        continue
                    for coll in os.listdir(root):
                        coll_path = os.path.join(root, coll, "video", video_id)
                        if os.path.isdir(coll_path):
                            keyframe_dir = coll_path
                            break
                    if keyframe_dir:
                        break
                except Exception as e:
                    logger.debug(f"metadata: scan failed under {root}: {e}")

            if not keyframe_dir:
                logger.warning(f"metadata: keyframe dir not found for video_id={video_id} under {roots}")
                continue

            # Liệt kê shot Sxxxxx
            try:
                shot_dirs = [d for d in os.listdir(keyframe_dir)
                            if d and d[0] in ('S','s') and os.path.isdir(os.path.join(keyframe_dir, d))]
                shot_dirs.sort()
            except Exception as e:
                logger.warning(f"metadata: listdir failed for {keyframe_dir}: {e}")
                continue

            take = min(max_shots_per_video, len(shot_dirs), remaining)
            for shot_id in shot_dirs[:take]:
                shot_path = os.path.join(keyframe_dir, shot_id)
                try:
                    files = [f for f in os.listdir(shot_path)
                            if f.lower().endswith(('.jpg', '.png'))]
                    if not files:
                        continue
                    # Ưu tiên *_selected.*
                    pref = next((f for f in files if f.lower().endswith('_selected.jpg')
                                            or f.lower().endswith('_selected.png')), None)
                    fname = pref or sorted(files)[0]

                    abs_thumb = os.path.join(shot_path, fname)
                    # Map sang URL theo đúng gốc đã tìm thấy
                    # Nếu khớp OUT_FRAME_DIR → STATIC_IMAGE_PATH; nếu khớp alt → thay alt
                    url = abs_thumb.replace("\\", "/")
                    if url.startswith(OUT_FRAME_DIR):
                        url = url.replace(OUT_FRAME_DIR, STATIC_IMAGE_PATH, 1)
                    elif url.startswith(alt):
                        url = url.replace(alt, STATIC_IMAGE_PATH, 1)

                    expanded.append({
                        "video_id": video_id,
                        "shot_id": shot_id,
                        "thumbnail": url,
                        "score": float(vscore),  # score BM25 mức video
                    })
                    remaining -= 1
                    if remaining <= 0:
                        break
                except Exception as e:
                    logger.debug(f"metadata: cannot build thumb for {video_id}/{shot_id}: {e}")

        logger.info(f"metadata: expanded_shots={len(expanded)}")
        return expanded

origins = [
    "http://localhost:5173",
]

mount_paths = [
    (f"/{STATIC_IMAGE_PATH}", OUT_FRAME_DIR),
    (f"/{STATIC_VIDEO_PATH}", INP_VIDEO_DIR),
]

app = App(origins, mount_paths)
