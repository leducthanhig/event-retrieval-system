import os
import json
from tqdm import tqdm
from faster_whisper import WhisperModel

from configs import VIDEO_DIR, WHISPER_OUTPUT_PATH

file_paths = sorted([os.path.join(VIDEO_DIR, file)
                     for file in os.listdir(VIDEO_DIR)
                     if file.startswith('L25')])

model = WhisperModel('small', device='cuda', compute_type='float32', num_workers=8)

data = []
for v_idx, file in tqdm(enumerate(file_paths), total=len(file_paths)):
    segments, info = model.transcribe(file, 'vi')
    video_id = os.path.splitext(os.path.basename(file_paths[v_idx]))[0]
    for s_idx, segment in enumerate(segments):
        data.append({
            'video_id': video_id,
            'id': segment.id - 1,
            'start': segment.start,
            'end': segment.end,
            'text': segment.text,
        })

with open(WHISPER_OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
