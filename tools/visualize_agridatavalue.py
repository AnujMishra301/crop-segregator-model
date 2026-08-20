"""
AgriDataValue Annotation QA Visualization Engine
Renders ground-truth bounding boxes and class labels (0: Crop [Green], 1: Weed [Red])
for representative images from train, valid, and test splits.
Saves visualizations under dataset_agridatavalue/qa/initial_annotations/.
"""

import os
import sys
import random
import cv2
import numpy as np
from PIL import Image

EXTRACTED_DIR = "dataset_agridatavalue/extracted"
QA_DIR = "dataset_agridatavalue/qa/initial_annotations"

CLASS_NAMES = {0: "Crop", 1: "Weed"}
CLASS_COLORS = {
    0: (0, 255, 0),   # Green for Crop
    1: (0, 0, 255)    # Red for Weed
}

def find_split_files(split):
    split_dir = os.path.join(EXTRACTED_DIR, split)
    img_map, lbl_map = {}, {}
    for root, _, files in os.walk(split_dir):
        for f in files:
            base, ext = os.path.splitext(f)
            p = os.path.join(root, f)
            if ext.lower() in [".jpg", ".jpeg", ".png"]:
                img_map[base] = p
            elif ext.lower() == ".txt" and f.lower() != "classes.txt":
                lbl_map[base] = p
    return img_map, lbl_map

def render_visualization(img_path, lbl_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w = img.shape[:2]

    if os.path.exists(lbl_path):
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cid = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])
                        x1 = int((xc - bw / 2.0) * w)
                        y1 = int((yc - bh / 2.0) * h)
                        x2 = int((xc + bw / 2.0) * w)
                        y2 = int((yc + bh / 2.0) * h)

                        color = CLASS_COLORS.get(cid, (255, 255, 255))
                        lbl_name = CLASS_NAMES.get(cid, f"Class {cid}")

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        # Draw label text box
                        (tw, th), _ = cv2.getTextSize(lbl_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(img, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, max(y1, th + 8)), color, -1)
                        cv2.putText(img, lbl_name, (x1 + 3, max(y1 - 4, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    except ValueError:
                        continue

    cv2.imwrite(out_path, img)
    return True

def generate_qa_visualizations():
    os.makedirs(QA_DIR, exist_ok=True)
    random.seed(42)

    print("[QA Visualizer] Generating representative QA annotation visualizations...")
    total_rendered = 0

    categories = {
        "crop_only": [],
        "weed_only": [],
        "crop_plus_weed": [],
        "small_weed": [],
        "multiple_weeds": []
    }

    # Process each split
    for split in ["train", "valid", "test"]:
        img_map, lbl_map = find_split_files(split)
        common_bases = sorted(list(set(img_map.keys()) & set(lbl_map.keys())))

        # Select 10 random representative samples for split
        sample_bases = random.sample(common_bases, min(10, len(common_bases)))
        
        for idx, base in enumerate(sample_bases):
            ipath = img_map[base]
            lpath = lbl_map[base]
            opath = os.path.join(QA_DIR, f"{split}_sample_{idx+1:02d}_{base[:12]}.jpg")
            if render_visualization(ipath, lpath, opath):
                total_rendered += 1

        # Categorize images for specific category visualizations
        for base in common_bases:
            ipath = img_map[base]
            lpath = lbl_map[base]
            
            crop_c, weed_c = 0, 0
            has_small_w = False

            with open(lpath, "r") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) == 5:
                        try:
                            cid = int(p[0])
                            bw = float(p[3])
                            if cid == 0: crop_c += 1
                            elif cid == 1:
                                weed_c += 1
                                if bw * 5280 < 32.0:
                                    has_small_w = True
                        except ValueError:
                            pass

            if crop_c > 0 and weed_c == 0:
                categories["crop_only"].append((base, ipath, lpath))
            elif weed_c > 0 and crop_c == 0:
                categories["weed_only"].append((base, ipath, lpath))
            elif crop_c > 0 and weed_c > 0:
                categories["crop_plus_weed"].append((base, ipath, lpath))

            if has_small_w:
                categories["small_weed"].append((base, ipath, lpath))
            if weed_c >= 3:
                categories["multiple_weeds"].append((base, ipath, lpath))

    # Render category specific samples
    print("\n[QA Visualizer] Category Sample Breakdown:")
    for cat_name, item_list in categories.items():
        print(f"  Category '{cat_name}': {len(item_list)} matching images found in dataset.")
        if item_list:
            sample_item = random.choice(item_list)
            bname, ip, lp = sample_item
            opath = os.path.join(QA_DIR, f"category_{cat_name}_{bname[:12]}.jpg")
            render_visualization(ip, lp, opath)
            total_rendered += 1
        else:
            print(f"  [Notice] No images matching category '{cat_name}' were found in dataset.")

    print(f"\n[QA Visualizer] Generated {total_rendered} QA visual inspection images in '{QA_DIR}'.")

if __name__ == "__main__":
    generate_qa_visualizations()
