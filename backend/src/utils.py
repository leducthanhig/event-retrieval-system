import ffmpeg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def get_nvidia_decoder(codec_name: str):
    """Return the NVIDIA decoder name for a given codec."""
    codec_map = {
        'h264': 'h264_cuvid',
        'hevc': 'hevc_cuvid',
        'mpeg4': 'mpeg4_cuvid',
    }
    return codec_map.get(codec_name, None)

def get_video_codec(video_file: str) -> str:
    """Detect the codec name of the video file."""
    probe = ffmpeg.probe(video_file)
    for stream in probe['streams']:
        if stream['codec_type'] == 'video':
            return stream['codec_name']
    raise RuntimeError("No video stream found")

def get_avg_fps(video_path: str) -> float:
    """Get the average frame rate of the video."""
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return eval(video_stream['avg_frame_rate'])

def visualize(metadata: dict, object_conf_thresh: float = None):
    """Visualize an image with bounding boxes and labels."""
    path = metadata['path']
    print('Visualizing image', path)

    # Load image
    img = Image.open(path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw bounding boxes
    for obj in metadata['objects']:
        score = obj['score']
        if object_conf_thresh and score < object_conf_thresh:
            continue

        xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
        label = f"{obj['label']} ({score:.2f})"

        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, label, color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5, pad=1))

    plt.show()

def encode_object_bbox(object_info: dict, src_size=(1280, 720), dst_size=(16, 9)):
    """Convert object bounding box to textual description."""
    fx = dst_size[0] / src_size[0]
    fy = dst_size[1] / src_size[1]

    texts = []
    label = object_info['label']

    xmin = int(object_info['xmin'] * fx)
    xmax = int(object_info['xmax'] * fx)
    ymin = int(object_info['ymin'] * fy)
    ymax = int(object_info['ymax'] * fy)

    for i in range(ymin, ymax + 1):
        for j in range(xmin, xmax + 1):
            texts.append(f"{i}{chr(j + ord('a'))}{label}")

    return ' '.join(texts)
