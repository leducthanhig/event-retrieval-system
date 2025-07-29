import ffmpeg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from cores.models import FrameModel

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

def visualize(metadata: FrameModel, object_conf_thresh: float = None):
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

        rect = patches.Rectangle((xmin, ymin), xmax, ymax, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, label, color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5, pad=1))

    plt.show()
