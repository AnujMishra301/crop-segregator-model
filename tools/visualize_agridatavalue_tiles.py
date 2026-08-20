"""
AgriDataValue Tiled QA Visualization Engine
Renders ground truth bounding boxes and class labels (0: Crop [Green], 1: Weed [Red])
for tiled patch images (640 x 480).
Saves visual inspection frames under dataset_agridatavalue/qa/tiles/.
"""

import os
import sys
import random
import cv2

TILED_DIR = "dataset_agridatavalue/tiled"
QA_TILES_DIR = "dataset_agridatavalue/qa/tiles"

CLASS_NAMES = {0: "Crop", 1: "Weed"}
CLASS_COLORS = {
    0: (0, 255, 0),   # Green for Crop
    1: (0, 0, 255)    # Red for Weed
}

def render_tile_visualization(img_path, lbl_path, out_path):
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
                        lbl_name = f"{CLASS_NAMES.get(cid, f'Class {cid}')} [{int(bw*w)}x{int(bh*h)}px]"

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        (tw, th), _ = cv2.getTextSize(lbl_name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), color, -1)
                        cv2.putText(img, lbl_name, (x1 + 2, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                    except ValueError:
                        continue

    cv2.imwrite(out_path, img)
    return True

def generate_tile_visualizations():
    os.makedirs(QA_TILES_DIR, exist_ok=True)
    random.seed(42)

    print("[Tile Visualizer] Generating representative tiled QA visualizations...")
    total_rendered = 0

    categories = {
        "small_weeds": [],
        "multiple_weeds": [],
        "crop_plus_weed": [],
        "boundary_objects": [],
        "empty_background": []
    }

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(TILED_DIR, split, "images")
        lbl_dir = os.path.join(TILED_DIR, split, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
        if not images:
            continue

        # Render 10 random samples per split
        sample_files = random.sample(images, min(10, len(images)))
        for idx, fname in enumerate(sample_files):
            base = os.path.splitext(fname)[0]
            ipath = os.path.join(img_dir, fname)
            lpath = os.path.join(lbl_dir, base + ".txt")
            opath = os.path.join(QA_TILES_DIR, f"{split}_tile_{idx+1:02d}_{base}.jpg")
            if render_tile_visualization(ipath, lpath, opath):
                total_rendered += 1

        # Categorize tiles
        for fname in images:
            base = os.path.splitext(fname)[0]
            ipath = os.path.join(img_dir, fname)
            lpath = os.path.join(lbl_dir, base + ".txt")

            crop_c, weed_c = 0, 0
            has_small_w = False
            has_boundary_obj = False

            if os.path.exists(lpath):
                with open(lpath, "r") as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) == 5:
                            try:
                                cid = int(p[0])
                                xc, yc, bw, bh = map(float, p[1:])
                                if cid == 0: crop_c += 1
                                elif cid == 1:
                                    weed_c += 1
                                    if bw * 640 < 32.0:
                                        has_small_w = True

                                x1 = (xc - bw / 2.0)
                                y1 = (yc - bh / 2.0)
                                x2 = (xc + bw / 2.0)
                                y2 = (yc + bh / 2.0)
                                if x1 <= 0.01 or y1 <= 0.01 or x2 >= 0.99 or y2 >= 0.99:
                                    has_boundary_obj = True
                            except ValueError:
                                pass

            if crop_c == 0 and weed_c == 0:
                categories["empty_background"].append((fname, ipath, lpath))
            else:
                if crop_c > 0 and weed_c > 0:
                    categories["crop_plus_weed"].append((fname, ipath, lpath))
                if weed_c >= 3:
                    categories["multiple_weeds"].append((fname, ipath, lpath))
                if has_small_w:
                    categories["small_weeds"].append((fname, ipath, lpath))
                if has_boundary_obj:
                    categories["boundary_objects"].append((fname, ipath, lpath))

    print("\n[Tile Visualizer] Category Specific Sample Breakdown:")
    for cat_name, item_list in categories.items():
        print(f"  Category '{cat_name}': {len(item_list)} matching tiles found.")
        if item_list:
            sample = random.choice(item_list)
            fname, ip, lp = sample
            opath = os.path.join(QA_TILES_DIR, f"category_{cat_name}_{fname}")
            render_tile_visualization(ip, lp, opath)
            total_rendered += 1

    print(f"\n[Tile Visualizer] Generated {total_rendered} QA visual inspection frames in '{QA_TILES_DIR}'.")

if __name__ == "__main__":
    generate_tile_visualizations()
