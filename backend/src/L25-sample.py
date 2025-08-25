import os
import json
import ffmpeg
from tqdm import tqdm
from shutil import rmtree
from torch.cuda import is_available

from utils import get_avg_fps, get_decoder, get_video_codec
from configs import (
    VIDEO_METADATA_PATH,
    WHISPER_OUTPUT_PATH,
    INP_VIDEO_DIR,
    OUT_FRAME_DIR,
)

def extract_frames(video_path: str, root_dir: str, shots: list[list[int]], use_gpu=is_available()):
    if os.path.exists(root_dir):
        rmtree(root_dir)
    os.makedirs(root_dir)

    # Calculate all frame positions to extract
    all_positions = []
    position_info = {}  # Maps position to (shot_idx, selected)

    for shot_idx, (start, end) in enumerate(shots):
        positions = [
            start,
            int(start + (end - start) / 2),
            end
        ]

        # Remove duplicates within a shot
        unique_positions = sorted(set(positions))

        for pos in unique_positions:
            all_positions.append(pos)
            selected = (pos == positions[(len(unique_positions) - 1) // 2])
            position_info[pos] = (shot_idx, selected)

    # Sort positions for easier processing
    all_positions = sorted(set(all_positions))

    # Create output temp directory
    temp_dir = os.path.join(root_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Build select filter for all frames at once
    select_conditions = '+'.join([f"eq(n,{pos})" for pos in all_positions])

    # Extract all frames in one operation
    decoder = get_decoder(video_path, use_gpu)
    configs = {'vcodec': decoder}
    if use_gpu and decoder.endswith('cuvid'):
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
    frame_files = sorted(os.listdir(temp_dir))

    for i, frame_file in enumerate(frame_files):
        if i >= len(all_positions):
            break

        pos = all_positions[i]
        shot_idx, selected = position_info[pos]

        # Create shot directory
        shot_id = f"S{shot_idx:05}"
        frame_dir = os.path.join(root_dir, shot_id)
        os.makedirs(frame_dir, exist_ok=True)

        # Rename and move file
        frame_id = f"F{pos:06}"
        src_path = os.path.join(temp_dir, frame_file)

        if selected:
            dst_path = os.path.join(frame_dir, f"{frame_id}_selected.jpg")
        else:
            dst_path = os.path.join(frame_dir, f"{frame_id}.jpg")

        os.rename(src_path, dst_path)
        total_frames += 1

    rmtree(temp_dir, ignore_errors=True)
    return total_frames

with open(VIDEO_METADATA_PATH) as f:
    metadata = json.load(f)

with open(WHISPER_OUTPUT_PATH, encoding='utf-8') as f:
    data = json.load(f)

total_frames = 0
shots = []
video_id = ''
video_path = ''
fps = 0.0
for i in tqdm(range(len(data)), total=len(data)):
    if data[i]['video_id'] != video_id:
        # Handle the old video (if needed)
        if all([shots, video_id, video_path, fps]):
            out_dir = os.path.join(OUT_FRAME_DIR, video_id)
            total_frames += extract_frames(video_path, out_dir, shots)
            metadata[video_id] = {
                'path': video_path,
                'shots': [shot[0] for shot in shots],
            }

        # Prepare for the new video
        shots = []
        video_id = data[i]['video_id']
        video_path = os.path.join(INP_VIDEO_DIR, f"{video_id}.mp4")
        fps = get_avg_fps(video_path)

    shots.append([
        int(data[i]['start']*fps),
        int(data[i]['end']*fps) - 1
    ])

if all([shots, video_id, video_path, fps]):
    out_dir = os.path.join(OUT_FRAME_DIR, video_id)
    total_frames += extract_frames(video_path, out_dir, shots)
    metadata[video_id] = {
        'path': video_path,
        'shots': [shot[0] for shot in shots],
    }

print(f"Successfully extracted {total_frames} frames")

with open(VIDEO_METADATA_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)
