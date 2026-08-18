"""
Dataset Visualization Module
Renders random annotated images from the dataset with bounding boxes,
class names, and confidence colors, saving output samples for visual inspection.
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = {
    0: "weed",
    1: "crop",
    2: "grass_lawn",
    3: "other"
}

CLASS_COLORS = {
    0: (220, 20, 60),    # Crimson Red for weed
    1: (34, 139, 34),    # Forest Green for crop
    2: (255, 165, 0),   # Orange for grass
    3: (128, 128, 128)  # Gray for other
}

def visualize_sample_images(image_dir="dataset/train/images", label_dir="dataset/train/labels",
                            output_dir="dataset/processed/visualizations", num_samples=5):
    """Draws bounding boxes and labels on sample images and saves to output_dir."""
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Directory '{image_dir}' or '{label_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    img_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not img_files:
        print("No images found for visualization.")
        return

    sample_files = random.sample(img_files, min(num_samples, len(img_files)))
    print(f"Visualizing {len(sample_files)} sample annotated images...")

    for fname in sample_files:
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(image_dir, fname)
        lbl_path = os.path.join(label_dir, base + ".txt")

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            draw = ImageDraw.Draw(img)

            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:])

                            # Convert normalized coords back to absolute pixels
                            xmin = int((xc - bw / 2.0) * w)
                            ymin = int((yc - bh / 2.0) * h)
                            xmax = int((xc + bw / 2.0) * w)
                            ymax = int((yc + bh / 2.0) * h)

                            color = CLASS_COLORS.get(cls_id, (255, 255, 0))
                            cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")

                            # Draw box
                            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
                            
                            # Draw text label box
                            text = f"{cls_name}"
                            draw.rectangle([xmin, max(0, ymin - 20), xmin + len(text)*9, max(0, ymin)], fill=color)
                            draw.text((xmin + 4, max(0, ymin - 18)), text, fill=(255, 255, 255))

            save_path = os.path.join(output_dir, f"vis_{fname}")
            img.save(save_path)
            print(f"Saved visualization: {save_path}")

        except Exception as e:
            print(f"Error visualizing {fname}: {e}")

if __name__ == "__main__":
    visualize_sample_images()
