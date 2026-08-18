"""
Dataset V2 Validation Engine
Performs comprehensive quality checks on images and YOLO bounding box labels:
- Verifies normalized coordinate boundaries [0, 1]
- Checks x_min < x_max and y_min < y_max
- Ensures class IDs are strictly in {0: weed, 1: crop, 2: grass_lawn, 3: other}
- Detects orphan images (missing label) and orphan labels (missing image)
- Checks image integrity and readability
"""

import os
import sys
import argparse
import cv2

VALID_CLASSES = {0, 1, 2, 3}
CLASS_NAMES = {0: "weed", 1: "crop", 2: "grass_lawn", 3: "other"}

def validate_dataset_split(img_dir, lbl_dir):
    """Validates an image-label directory pair."""
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        return {"error": f"Directories missing: '{img_dir}' or '{lbl_dir}'"}

    img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    lbl_files = {os.path.splitext(f)[0]: f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')}

    orphan_images = set(img_files.keys()) - set(lbl_files.keys())
    orphan_labels = set(lbl_files.keys()) - set(img_files.keys())
    common_bases = set(img_files.keys()) & set(lbl_files.keys())

    invalid_boxes = 0
    total_boxes = 0
    corrupt_images = 0
    class_histogram = {0: 0, 1: 0, 2: 0, 3: 0}

    for base in common_bases:
        img_path = os.path.join(img_dir, img_files[base])
        lbl_path = os.path.join(lbl_dir, lbl_files[base])

        img = cv2.imread(img_path)
        if img is None:
            corrupt_images += 1
            continue

        with open(lbl_path, "r") as f:
            for line_no, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    invalid_boxes += 1
                    continue

                try:
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])

                    if cid not in VALID_CLASSES:
                        invalid_boxes += 1
                        continue

                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                        invalid_boxes += 1
                        continue

                    class_histogram[cid] += 1
                    total_boxes += 1
                except ValueError:
                    invalid_boxes += 1

    return {
        "valid_images": len(common_bases) - corrupt_images,
        "corrupt_images": corrupt_images,
        "orphan_images": len(orphan_images),
        "orphan_labels": len(orphan_labels),
        "total_boxes": total_boxes,
        "invalid_boxes": invalid_boxes,
        "class_histogram": class_histogram
    }

def main():
    parser = argparse.ArgumentParser(description="Dataset V2 Validation Engine")
    parser.add_argument("--img_dir", type=str, default="dataset_v2/images", help="Path to images directory")
    parser.add_argument("--lbl_dir", type=str, default="dataset_v2/labels", help="Path to labels directory")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f" DATASET V2 VALIDATION REPORT")
    print(f" Image Dir: {args.img_dir}")
    print(f" Label Dir: {args.lbl_dir}")
    print(f"=======================================================\n")

    res = validate_dataset_split(args.img_dir, args.lbl_dir)
    if "error" in res:
        print(f"[Notice] {res['error']}")
        sys.exit(0)

    print(f"  Valid Image-Label Pairs: {res['valid_images']}")
    print(f"  Corrupt Images:         {res['corrupt_images']}")
    print(f"  Orphan Images:          {res['orphan_images']}")
    print(f"  Orphan Labels:          {res['orphan_labels']}")
    print(f"  Total Bounding Boxes:   {res['total_boxes']}")
    print(f"  Invalid Bounding Boxes: {res['invalid_boxes']}")
    print(f"\n  Class Histogram:")
    for cid, count in res["class_histogram"].items():
        print(f"    Class {cid} ({CLASS_NAMES[cid]}): {count} boxes")

    print(f"\n=======================================================\n")

if __name__ == "__main__":
    main()
