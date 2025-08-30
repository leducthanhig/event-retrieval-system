import os
import logging
import json
from shutil import rmtree
from typing import Callable

import ffmpeg
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torchvision.models import resnet152, ResNet152_Weights
from open_clip import create_model_and_transforms as clip

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # to disable the warning message
from transnetv2 import TransNetV2

from utils import get_decoder, get_avg_fps

logger = logging.getLogger(__name__)


class FrameSampler:
    def __init__(self,
                 metadata: dict[str, dict],
                 output_root_dir: str,
                 fps: float = 2,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 ):
        self.metadata = metadata
        self.output_root_dir = output_root_dir
        self.fps = fps
        self.device = torch.device(device)
        self.model: dict[str, torch.nn.Sequential | torch.Module] = {}
        self.transforms: dict[str, Callable[[Image.Image], torch.Tensor]] = {}
        self.frame_paths = []

        # Initialize ResNet model and transforms
        weights = ResNet152_Weights.DEFAULT
        resnet = resnet152(weights=weights)
        model = torch.nn.Sequential(*list(resnet.children())[:-1]) # Remove final FC
        model.eval().to(self.device)
        transforms = weights.transforms()

        self.model['resnet'] = model
        self.transforms['resnet'] = transforms
        self.resize_size = transforms.resize_size[0]

        # Initialize CLIP model and transforms
        model, _, transforms = clip('ViT-B-32-256', 'datacomp_s34b_b86k', device=self.device)
        model.eval()

        self.model['clip'] = model
        self.transforms['clip'] = transforms
        self.resize_size = max(self.resize_size, transforms.transforms[0].size)

    def extract_frames(self, video_path: str, batch_size=64):
        """Extract video frames at given fps."""
        decoder = get_decoder(video_path, self.device.type == 'cuda')
        configs = {'vcodec': decoder}
        if self.device.type == 'cuda' and decoder.endswith('cuvid'):
            configs['hwaccel'] = 'cuda'

        stream = (
            ffmpeg
            .input(video_path, **configs)
            .filter('fps', self.fps)
            .filter('scale', self.resize_size, self.resize_size)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        )

        proc = ffmpeg.run_async(stream, pipe_stdout=True, quiet=False)

        while True:
            chunk_size = self.resize_size * self.resize_size * 3 * batch_size
            chunk = proc.stdout.read(chunk_size)
            if not chunk:
                break
            frames = np.frombuffer(chunk, dtype=np.uint8).copy()
            frames = frames.reshape(
                (-1, self.resize_size, self.resize_size, 3))
            frames = [Image.fromarray(frame) for frame in frames]
            yield frames

        proc.wait()

    @torch.no_grad()
    def generate_embeddings(self, frames: list[Image.Image]):
        """Generate embeddings from given frames."""
        frames_resnet = [self.transforms['resnet']
                         (frame) for frame in frames]
        frames_resnet = torch.stack(frames_resnet).to(self.device)
        frames_clip = [self.transforms['clip']
                       (frame) for frame in frames]
        frames_clip = torch.stack(frames_clip).to(self.device)

        embeddings = torch.concatenate([
            self.model['resnet'](frames_resnet).squeeze(-1).squeeze(-1),
            self.model['clip'].encode_image(frames_clip),
        ], dim=1)

        return embeddings.cpu().numpy()

    def sample_frames(self, video_id: str, threshold=0.9, batch_size=64):
        """Sample frames bases on cosine similarity of embeddings."""
        sampled_indices = []
        prev_emb = None
        shots = self.metadata[video_id]['shots']
        shot_idx = 0
        fps = self.metadata[video_id]['fps']
        video_path = self.metadata[video_id]['path']
        for i, frames_batch in enumerate(self.extract_frames(video_path, batch_size)):
            embeddings = self.generate_embeddings(frames_batch)

            for j, emb in enumerate(embeddings):
                global_idx = int((i * batch_size + j) * fps / self.fps)
                if shot_idx < len(shots) and global_idx >= shots[shot_idx]:
                    # Reset for new shot
                    shot_idx += 1
                    prev_emb = None
                    sampled_indices.append(global_idx)

                if prev_emb is not None:
                    # Compute cosine similarity
                    sim = np.dot(prev_emb, emb) / np.linalg.norm(prev_emb) / np.linalg.norm(emb)
                    if sim <= threshold:
                        sampled_indices.append(global_idx)

                prev_emb = emb  # Update for next comparison

        logger.info(f"Sampled {len(sampled_indices)} frames")
        return sampled_indices

    def save_frames(self, video_id: str, indices: list[int]):
        """Save frames at given indices."""
        if not indices:
            logger.info("No indices provided, no frames saved.")
            return

        shot_root_dir = os.path.join(self.output_root_dir, video_id)
        if os.path.exists(shot_root_dir):
            rmtree(shot_root_dir, ignore_errors=True)
        os.makedirs(shot_root_dir)

        # Create output temp directory
        temp_dir = os.path.join(shot_root_dir, "temp")
        os.makedirs(temp_dir)

        try:
            # Build select filter for all frames at once
            select_conditions = '+'.join([f"eq(n,{idx})" for idx in indices])

            # Extract all frames in one operation
            video_path = self.metadata[video_id]['path']
            decoder = get_decoder(video_path, self.device.type == 'cuda')
            configs = {'vcodec': decoder}
            if self.device.type == 'cuda' and decoder.endswith('cuvid'):
                configs['hwaccel'] = 'cuda'
            (
                ffmpeg
                .input(video_path, **configs)
                .filter('select', select_conditions)
                .output(os.path.join(temp_dir, 'frame_%06d.jpg'), vsync=0)
                .run(quiet=True)
            )

            # Organize extracted frames into shot directories
            total_frames = 0
            shots = self.metadata[video_id]['shots']
            shot_idx = 0
            frame_dir = ''
            frame_files = sorted(os.listdir(temp_dir))
            for i, frame_file in enumerate(frame_files):
                if shot_idx < len(shots) and indices[i] >= shots[shot_idx]:
                    # Create shot directory
                    frame_dir = os.path.join(shot_root_dir, f"S{shot_idx:05}")
                    os.makedirs(frame_dir)
                    shot_idx += 1

                # Rename and move file
                src_path = os.path.join(temp_dir, frame_file)
                ext = os.path.splitext(frame_file)[1]
                dst_path = os.path.join(frame_dir, f"F{indices[i]:06}{ext}")
                os.rename(src_path, dst_path)
                self.frame_paths.append(dst_path)
                total_frames += 1

            return total_frames

        except ffmpeg.Error as e:
            err = e.stderr.decode() if hasattr(e, 'stderr') else e
            logger.error(f"ffmpeg decoding failed for {video_path}: {err}")
            return 0

        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return 0

        finally:
            # Clean up temp directory
            rmtree(temp_dir, ignore_errors=True)

    def run(self):
        """Extract keyframes from given videos."""
        os.makedirs(self.output_root_dir, exist_ok=True)

        desc = 'Sampling keyframes'
        total_frames = 0
        for video_id in tqdm(self.metadata, desc=desc):
            sampled_indices = self.sample_frames(video_id)
            total_frames += self.save_frames(video_id, sampled_indices)

        logger.info(f"Successfully extracted {total_frames} frames from {len(self.metadata)} videos")
        return self.frame_paths


class ShotDetector:
    def __init__(self,
                 video_root_dir: str,
                 metadata_save_path: str,
                 batch_size=8,
                 use_gpu=torch.cuda.is_available()):
        self.video_root_dir = video_root_dir
        self.metadata_save_path = metadata_save_path
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.video_paths = [os.path.join(dirpath, filename)
                            for dirpath, _, filenames in os.walk(video_root_dir)
                            for filename in filenames]

        self.model = TransNetV2()

        self.metadata = dict()
        for path in self.video_paths:
            video_id = os.path.splitext(os.path.basename(path))[0]
            self.metadata[video_id] = dict(path=path)

    # Modified version of https://github.com/soCzech/TransNetV2/blob/master/inference/transnetv2.py
    def predict_frames(self, frames: np.ndarray):
        """Make predictions for the given frames."""
        # Return [batch_size, 100, 27, 48, 3] instead of [1, 100, 27, 48, 3] for faster inference
        def input_iterator():
            """
            Return windows of size 100 where the first/last 25 frames are from the previous/next batch
            the first and last window must be padded by copies of the first and last frame of the video.
            """
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            windows = []
            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                windows.append(out[np.newaxis])
                ptr += 50
                if len(windows) == self.batch_size:
                    yield np.concatenate(windows)
                    windows = []

            if windows:
                yield np.concatenate(windows)

        predictions = []
        for inp_batch in input_iterator():
            single_frame_pred, all_frames_pred = self.model.predict_raw(inp_batch)
            predictions.extend([(single_frame_pred.numpy()[i, 25:75, 0],
                                 all_frames_pred.numpy()[i, 25:75, 0])
                                 for i in range(inp_batch.shape[0])])

        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames

    def predict_video(self, video_file: str):
        """Make predictions for the given video."""
        try:
            decoder = get_decoder(video_file, self.use_gpu)
            configs = {'vcodec': decoder}
            if self.use_gpu and decoder.endswith('cuvid'):
                configs['hwaccel'] = 'cuda'

            video_stream, err = (
                ffmpeg
                .input(video_file, **configs)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='48x27')
                .run(capture_stdout=True, capture_stderr=True)
            )

            frames = np.frombuffer(video_stream, np.uint8).reshape((-1, 27, 48, 3))

        except ffmpeg.Error as e:
            err = e.stderr.decode() if hasattr(e, 'stderr') else e
            logger.error(f"ffmpeg decoding failed for {video_file}: {err}")
            return None, None

        except Exception as e:
            logger.error(f"Error decoding video {video_file}: {e}")
            return None, None

        return self.predict_frames(frames)

    def save_metadata(self):
        with open(self.metadata_save_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def run(self):
        """Detect shots from given videos."""
        desc = 'Detecting shots'
        total_shots = 0
        for video_path in tqdm(self.video_paths, desc, len(self.video_paths)):
            # Detect shots
            preds, _ = self.predict_video(video_path)
            if preds is None:
                logger.warning(f"Skipping {video_path} due to decoding error.")
                return

            shots = self.model.predictions_to_scenes(preds)
            total_shots += len(shots)

            # Save start timestamps of each shot to metadata
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            self.metadata[video_id]['shots'] = [shot[0] for shot in shots.tolist()]
            self.metadata[video_id]['fps'] = get_avg_fps(video_path)

        self.save_metadata()

        logger.info(f"Successfully detected {total_shots} shots from {len(self.video_paths)} videos")
