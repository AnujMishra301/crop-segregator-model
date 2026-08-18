"""
Annotation QA Visualization Tool
Randomly samples annotated images, renders bounding boxes, class names,
annotation line IDs, and saves problem/flagged images to dataset/qa/ for human review.
"""

import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = {
    0: "weed",
    1: "crop",
    2: "grass_lawn",
    3: "other"
}

CLASS_COLORS = {
    0: (220, 20, 60),    # Red for weed
    1: (34, 139, 34),    # Green for crop
    2: (255, 140, 0),   # Orange for grass
    3: (128, 128, 128)  # Gray for other
}

def visualize_qa_annotations(image_dir="dataset/raw/images", label_dir="dataset/raw/labels",
                             review_list_path="dataset/qa/flagged_review_list.json",
                             qa_output_dir="dataset/qa/visualizations", num_samples=10):
    """Visualizes annotated images with bounding boxes, class labels, and annotation line IDs."""
    os.makedirs(qa_output_dir, exist_ok=True)

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Directory '{image_dir}' or '{label_dir}' not found.")
        return

    # Load flagged review items if available
    flagged_files = set()
    if os.path.exists(review_list_path):
        with open(review_list_path, 'r') as f:
            review_items = json.load(f)
            for item in review_items:
                fname = item.get("file", "")
                if fname.endswith(".txt"):
                    base = os.path.splitext(fname)[0]
                    flagged_files.add(base)
                elif fname.endswith(('.jpg', '.png', '.jpeg')):
                    base = os.path.splitext(fname)[0]
                    flagged_files.add(base)

    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not all_images:
        print("No images found for QA visualization.")
        return

    # Prioritize flagged images first, then random sampling
    flagged_image_files = [f for f in all_images if os.path.splitext(f)[0] in flagged_files]
    remaining_image_files = [f for f in all_images if f not in flagged_image_files]

    selected_files = flagged_image_files[:num_samples]
    if len(selected_files) < num_samples:
        needed = num_samples - len(selected_files)
        selected_files.extend(random.sample(remaining_image_files, min(needed, len(remaining_image_files))))

    print(f"QA Visualization: Processing {len(selected_files)} images ({len(flagged_image_files)} flagged)...")

    for fname in selected_files:
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(image_dir, fname)
        lbl_path = os.path.join(label_dir, base + ".txt")

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            draw = ImageDraw.Draw(img)

            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    lines = f.readlines()

                for line_idx, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])

                        xmin = int((xc - bw / 2.0) * w)
                        ymin = int((yc - bh / 2.0) * h)
                        xmax = int((xc + bw / 2.0) * w)
                        ymax = int((yc + bh / 2.0) * h)

                        color = CLASS_COLORS.get(cls_id, (255, 255, 0))
                        cls_name = CLASS_NAMES.get(cls_id, f"Cls{cls_id}")

                        # Draw bounding box
                        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

                        # Display Class Name + Annotation Line ID
                        label_str = f"#{line_idx}:{cls_name}"
                        
                        # Draw label tag background
                        draw.rectangle([xmin, max(0, ymin - 18), xmin + len(label_str)*8, max(0, ymin)], fill=color)
                        draw.text((xmin + 2, max(0, ymin - 16)), label_str, fill=(255, 255, 255))

            # Mark if file was flagged
            is_flagged = base in flagged_files
            prefix = "FLAGGED_" if is_flagged else "QA_"
            save_path = os.path.join(qa_output_dir, f"{prefix}{fname}")
            img.save(save_path)
            print(f"  Saved QA frame: {save_path}")

        except Exception as e:
            print(f"Error rendering QA frame {fname}: {e}")

if __name__ == "__main__":
    visualize_qa_annotations()
