import ffmpeg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def get_decoder(video_file: str, use_gpu=True):
    """Return the decoder name for a given video."""
    probe = ffmpeg.probe(video_file)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    codec_name = video_stream['codec_name']
    # For AV1, force to use cpu decoding
    if codec_name == 'av1':
        return 'libdav1d'
    if use_gpu:
        return f"{codec_name}_cuvid"
    return codec_name

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
