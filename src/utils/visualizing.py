import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize(metadata: dict):
    """Visualize an image with bounding boxes and labels."""
    # Load image
    img = Image.open(metadata['path'])
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw bounding boxes
    for obj in metadata.get('obj_loc', []):
        score = obj.get('score', None)
        if score < 0.5:
            continue
        x, y, w, h = obj['x'], obj['y'], obj['width'], obj['height']
        class_name = obj.get('objectName', '')
        label = f"{class_name}"
        if score is not None:
            label += f" ({score:.2f})"
        # Rectangle: (x, y), width, height
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, label, color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5, pad=1))

    # Add concept and action as title
    concept = metadata.get('concept', '')
    action = metadata.get('action', '')
    ax.set_title(f"Concept: {concept} | Action: {action}")

    plt.show()
