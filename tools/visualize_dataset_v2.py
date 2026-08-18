"""
Dataset V2 Ground Truth Annotation Visualization Engine
Renders bounding boxes and class labels onto dataset_v2 images for QA inspection.
Saves annotated visualization images to dataset_v2/qa/.
"""

import os
import sys
import argparse
import cv2

CLASS_NAMES = {0: "weed", 1: "crop", 2: "grass_lawn", 3: "other"}
CLASS_COLORS = {
    0: (0, 0, 255),    # Red for Weed
    1: (0, 255, 0),    # Green for Crop
    2: (0, 255, 255),  # Yellow for Grass
    3: (255, 255, 0)   # Cyan for Other
}

def visualize_dataset_annotations(img_dir="dataset_v2/images", lbl_dir="dataset_v2/labels", output_dir="dataset_v2/qa", num_samples=10):
    """Draws ground truth boxes and saves sample visualizations to output_dir."""
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"[Notice] Image or label directory not found: '{img_dir}' or '{lbl_dir}'.")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not images:
        print(f"[Notice] No images found in '{img_dir}'.")
        return 0

    rendered_count = 0
    sample_files = images[:num_samples]

    for fname in sample_files:
        base = os.path.splitext(fname)[0]
        img_p = os.path.join(img_dir, fname)
        lbl_p = os.path.join(lbl_dir, base + ".txt")

        img = cv2.imread(img_p)
        if img is None:
            continue

        h, w = img.shape[:2]

        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])

                        x1 = int((xc - bw / 2.0) * w)
                        y1 = int((yc - bh / 2.0) * h)
                        x2 = int((xc + bw / 2.0) * w)
                        y2 = int((yc + bh / 2.0) * h)

                        color = CLASS_COLORS.get(cid, (255, 255, 255))
                        lbl_str = f"{CLASS_NAMES.get(cid, 'other')} ({cid})"

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, lbl_str, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out_path = os.path.join(output_dir, f"qa_vis_{fname}")
        cv2.imwrite(out_path, img)
        rendered_count += 1

    print(f"[Visualizer] Rendered {rendered_count} QA annotation visualization frames -> '{output_dir}'.")
    return rendered_count

def main():
    parser = argparse.ArgumentParser(description="Dataset V2 Annotation Visualizer")
    parser.add_argument("--img_dir", type=str, default="dataset_v2/images", help="Path to images directory")
    parser.add_argument("--lbl_dir", type=str, default="dataset_v2/labels", help="Path to labels directory")
    parser.add_argument("--output_dir", type=str, default="dataset_v2/qa", help="Output QA visualization directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of sample images to visualize (default: 10)")

    args = parser.parse_args()
    visualize_dataset_annotations(args.img_dir, args.lbl_dir, args.output_dir, args.samples)

if __name__ == "__main__":
    main()
