"""
AgriDataValue High-Resolution Tiling & Patching Engine
Tiles 5280 x 3956 UAV agricultural field images into 640 x 480 patches with 20% overlap.
Clips bounding box annotations, enforces minimum visible area (>=30%), re-normalizes to 640 x 480,
retains configurable empty background tiles (20%), and generates tile_metadata.csv.
"""

import os
import sys
import argparse
import random
import csv
import math
import cv2
import numpy as np
from PIL import Image

CLASS_NAMES = {0: "Crop", 1: "Weed"}

def find_extracted_split_dirs(extracted_base="dataset_agridatavalue/extracted"):
    """Locates actual image and label directories for train, valid, and test splits."""
    splits_map = {}
    for split_name, folder_name in [("train", "train"), ("val", "valid"), ("test", "test")]:
        img_dir = None
        lbl_dir = None
        
        # Search inside extracted directory
        base_split_path = os.path.join(extracted_base, folder_name)
        for root, dirs, files in os.walk(base_split_path):
            if "images" in dirs:
                img_dir = os.path.join(root, "images")
            if "labels" in dirs:
                lbl_dir = os.path.join(root, "labels")

        if img_dir and lbl_dir and os.path.exists(img_dir) and os.path.exists(lbl_dir):
            splits_map[split_name] = {"images": img_dir, "labels": lbl_dir}
        else:
            print(f"[Warning] Could not find images/labels for split '{split_name}' in '{base_split_path}'")

    return splits_map

def tile_dataset(
    extracted_base="dataset_agridatavalue/extracted",
    output_base="dataset_agridatavalue/tiled",
    tile_width=640,
    tile_height=480,
    overlap=0.20,
    min_visible_frac=0.30,
    keep_empty_ratio=0.20,
    seed=42
):
    """Processes source images one by one and generates tiled patches and annotations."""
    random.seed(seed)
    os.makedirs(output_base, exist_ok=True)
    metadata_csv_path = os.path.join(output_base, "tile_metadata.csv")

    splits_map = find_extracted_split_dirs(extracted_base)
    if not splits_map:
        print("[Error] No valid extracted splits found. Run extraction first.")
        sys.exit(1)

    stride_x = int(round(tile_width * (1.0 - overlap)))
    stride_y = int(round(tile_height * (1.0 - overlap)))

    print(f"\n=======================================================")
    print(f" AGRIDATAVALUE HIGH-RESOLUTION TILING PIPELINE")
    print(f" Tile Resolution: {tile_width} x {tile_height} px")
    print(f" Overlap: {overlap*100:.0f}% (Stride: {stride_x}x{stride_y} px)")
    print(f" Min Visible Fraction: {min_visible_frac*100:.0f}%")
    print(f" Keep Empty Ratio: {keep_empty_ratio*100:.0f}%")
    print(f"=======================================================\n")

    metadata_rows = []
    
    tiling_stats = {
        "total_source_images": 0,
        "total_tiles_generated": 0,
        "total_tiles_retained": 0,
        "object_tiles_retained": 0,
        "empty_tiles_generated": 0,
        "empty_tiles_retained": 0,
        "empty_tiles_discarded": 0,
        "annotations_discarded_low_visibility": 0,
        "annotations_invalid": 0,
        "crop_before_tiling": 0,
        "weed_before_tiling": 0,
        "crop_after_tiling": 0,
        "weed_after_tiling": 0,
        "splits": {}
    }

    csv_header = [
        "tile_filename", "source_image_filename", "source_split",
        "tile_x_origin", "tile_y_origin", "tile_width", "tile_height",
        "crop_count", "weed_count", "is_empty"
    ]

    for split_name, paths in splits_map.items():
        img_dir = paths["images"]
        lbl_dir = paths["labels"]

        dst_img_dir = os.path.join(output_base, split_name, "images")
        dst_lbl_dir = os.path.join(output_base, split_name, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"[Tiler] Processing split '{split_name}': {len(images)} source images...")

        split_stats = {
            "source_images": len(images),
            "tiles_retained": 0,
            "object_tiles": 0,
            "empty_tiles_retained": 0,
            "crop_count": 0,
            "weed_count": 0
        }

        tiling_stats["total_source_images"] += len(images)

        for src_img_name in images:
            base_name = os.path.splitext(src_img_name)[0]
            src_img_path = os.path.join(img_dir, src_img_name)
            src_lbl_path = os.path.join(lbl_dir, base_name + ".txt")

            img = cv2.imread(src_img_path)
            if img is None:
                print(f"[Warning] Could not read '{src_img_path}'")
                continue

            h_orig, w_orig = img.shape[:2]

            # Read original annotations
            orig_boxes = []
            if os.path.exists(src_lbl_path):
                with open(src_lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cid = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:])
                            x1 = (xc - bw / 2.0) * w_orig
                            y1 = (yc - bh / 2.0) * h_orig
                            x2 = (xc + bw / 2.0) * w_orig
                            y2 = (yc + bh / 2.0) * h_orig
                            area = (x2 - x1) * (y2 - y1)
                            orig_boxes.append({
                                "cid": cid, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "area": area
                            })

                            if cid == 0: tiling_stats["crop_before_tiling"] += 1
                            elif cid == 1: tiling_stats["weed_before_tiling"] += 1

            # Generate grid of tile origins
            x_origins = list(range(0, w_orig - tile_width + 1, stride_x))
            if x_origins[-1] + tile_width < w_orig:
                x_origins.append(w_orig - tile_width)  # Right boundary shift

            y_origins = list(range(0, h_orig - tile_height + 1, stride_y))
            if y_origins[-1] + tile_height < h_orig:
                y_origins.append(h_orig - tile_height)  # Bottom boundary shift

            tile_counter = 0

            for y_off in y_origins:
                for x_off in x_origins:
                    tile_counter += 1
                    tiling_stats["total_tiles_generated"] += 1

                    tx1, ty1 = x_off, y_off
                    tx2, ty2 = x_off + tile_width, y_off + tile_height

                    tile_crop = img[ty1:ty2, tx1:tx2]

                    # Filter intersecting boxes
                    tile_boxes = []
                    crop_c, weed_c = 0, 0

                    for ob in orig_boxes:
                        # Intersect box
                        ix1 = max(tx1, ob["x1"])
                        iy1 = max(ty1, ob["y1"])
                        ix2 = min(tx2, ob["x2"])
                        iy2 = min(ty2, ob["y2"])

                        if ix2 > ix1 and iy2 > iy1:
                            inter_area = (ix2 - ix1) * (iy2 - iy1)
                            vis_frac = inter_area / ob["area"] if ob["area"] > 0 else 0.0

                            if vis_frac >= min_visible_frac:
                                # Translate relative to tile
                                local_x1 = ix1 - tx1
                                local_y1 = iy1 - ty1
                                local_x2 = ix2 - tx1
                                local_y2 = iy2 - ty1

                                # Normalize to tile size (640 x 480)
                                norm_xc = round(((local_x1 + local_x2) / 2.0) / tile_width, 6)
                                norm_yc = round(((local_y1 + local_y2) / 2.0) / tile_height, 6)
                                norm_w = round((local_x2 - local_x1) / tile_width, 6)
                                norm_h = round((local_y2 - local_y1) / tile_height, 6)

                                # Assertions
                                assert 0.0 <= norm_xc <= 1.0, f"Invalid norm_xc {norm_xc} in {src_img_name}"
                                assert 0.0 <= norm_yc <= 1.0, f"Invalid norm_yc {norm_yc} in {src_img_name}"
                                assert 0.0 < norm_w <= 1.0, f"Invalid norm_w {norm_w} in {src_img_name}"
                                assert 0.0 < norm_h <= 1.0, f"Invalid norm_h {norm_h} in {src_img_name}"

                                tile_boxes.append((ob["cid"], norm_xc, norm_yc, norm_w, norm_h))
                                if ob["cid"] == 0:
                                    crop_c += 1
                                    tiling_stats["crop_after_tiling"] += 1
                                    split_stats["crop_count"] += 1
                                elif ob["cid"] == 1:
                                    weed_c += 1
                                    tiling_stats["weed_after_tiling"] += 1
                                    split_stats["weed_count"] += 1
                            else:
                                tiling_stats["annotations_discarded_low_visibility"] += 1

                    is_empty = len(tile_boxes) == 0

                    # Decide retention
                    retain_tile = False
                    if not is_empty:
                        retain_tile = True
                        tiling_stats["object_tiles_retained"] += 1
                        split_stats["object_tiles"] += 1
                    else:
                        tiling_stats["empty_tiles_generated"] += 1
                        if random.random() < keep_empty_ratio:
                            retain_tile = True
                            tiling_stats["empty_tiles_retained"] += 1
                            split_stats["empty_tiles_retained"] += 1
                        else:
                            tiling_stats["empty_tiles_discarded"] += 1

                    if retain_tile:
                        tiling_stats["total_tiles_retained"] += 1
                        split_stats["tiles_retained"] += 1

                        tile_filename = f"{base_name}_tile_{tile_counter:04d}.jpg"
                        tile_lbl_name = f"{base_name}_tile_{tile_counter:04d}.txt"

                        # Save tile image
                        cv2.imwrite(os.path.join(dst_img_dir, tile_filename), tile_crop)

                        # Save tile label file
                        with open(os.path.join(dst_lbl_dir, tile_lbl_name), "w") as lf:
                            for tb in tile_boxes:
                                lf.write(f"{tb[0]} {tb[1]:.6f} {tb[2]:.6f} {tb[3]:.6f} {tb[4]:.6f}\n")

                        # Track metadata
                        metadata_rows.append([
                            tile_filename, src_img_name, split_name,
                            tx1, ty1, tile_width, tile_height,
                            crop_c, weed_c, is_empty
                        ])

        tiling_stats["splits"][split_name] = split_stats

    # Write tile_metadata.csv
    with open(metadata_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(metadata_rows)

    print(f"\n[Tiler] Metadata saved to '{metadata_csv_path}' ({len(metadata_rows)} rows).")
    return tiling_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriDataValue High-Resolution Tiling Engine")
    parser.add_argument("--extracted-dir", type=str, default="dataset_agridatavalue/extracted")
    parser.add_argument("--output-dir", type=str, default="dataset_agridatavalue/tiled")
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=480)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--min-visible-fraction", type=float, default=0.30)
    parser.add_argument("--keep-empty-ratio", type=float, default=0.20)

    args = parser.parse_args()

    tile_dataset(
        extracted_base=args.extracted_dir,
        output_base=args.output_dir,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        overlap=args.overlap,
        min_visible_frac=args.min_visible_fraction,
        keep_empty_ratio=args.keep_empty_ratio
    )
