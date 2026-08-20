"""
AgriDataValue Tile Dataset Validation Engine
Performs strict QA verification on dataset_agridatavalue/tiled/:
- Verifies image-label file pairing and absence of orphan files
- Validates YOLO coordinate normalization [0, 1], positive dimensions, no NaN/Inf
- Ensures class IDs are strictly in {0: Crop, 1: Weed}
- Verifies metadata traceability in tile_metadata.csv
- Checks for duplicate source image content across splits (Data Leakage Audit)
"""

import os
import sys
import csv
import hashlib

TILED_DIR = "dataset_agridatavalue/tiled"
METADATA_CSV = os.path.join(TILED_DIR, "tile_metadata.csv")

def validate_tiles():
    if not os.path.exists(TILED_DIR):
        print(f"[Error] Tiled directory '{TILED_DIR}' does not exist.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f" AGRIDATAVALUE TILED DATASET VALIDATION REPORT")
    print(f"=======================================================\n")

    # 1. Verify tile_metadata.csv
    if not os.path.exists(METADATA_CSV):
        print(f"[Error] Metadata file '{METADATA_CSV}' missing.")
        sys.exit(1)

    metadata_tiles = set()
    source_to_split = {}
    
    with open(METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata_tiles.add(row["tile_filename"])
            source_to_split[row["source_image_filename"]] = row["source_split"]

    print(f"[Validation] Metadata records found: {len(metadata_tiles)} tiles.")

    total_tiles = 0
    total_boxes = 0
    crop_count = 0
    weed_count = 0
    invalid_boxes = 0
    orphan_images = 0
    orphan_labels = 0
    untraced_tiles = 0

    splits = ["train", "val", "test"]

    for split in splits:
        img_dir = os.path.join(TILED_DIR, split, "images")
        lbl_dir = os.path.join(TILED_DIR, split, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"[Warning] Missing split directory: '{img_dir}' or '{lbl_dir}'")
            continue

        images = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))}
        labels = {os.path.splitext(f)[0]: f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')}

        orph_i = set(images.keys()) - set(labels.keys())
        orph_l = set(labels.keys()) - set(images.keys())

        orphan_images += len(orph_i)
        orphan_labels += len(orph_l)
        split_tiles = len(images)
        total_tiles += split_tiles

        for base, img_fname in images.items():
            if img_fname not in metadata_tiles:
                untraced_tiles += 1

            lbl_fname = labels.get(base)
            if lbl_fname:
                lbl_path = os.path.join(lbl_dir, lbl_fname)
                with open(lbl_path, "r") as lf:
                    for line_no, line in enumerate(lf, 1):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) != 5:
                            invalid_boxes += 1
                            continue

                        try:
                            cid = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:])

                            if cid not in [0, 1]:
                                invalid_boxes += 1
                                continue

                            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                                invalid_boxes += 1
                                continue

                            total_boxes += 1
                            if cid == 0: crop_count += 1
                            elif cid == 1: weed_count += 1

                        except ValueError:
                            invalid_boxes += 1

    print(f"  Total Retained Tiles:       {total_tiles}")
    print(f"  Orphan Tile Images:         {orphan_images}")
    print(f"  Orphan Tile Labels:         {orphan_labels}")
    print(f"  Untraced Metadata Tiles:    {untraced_tiles}")
    print(f"  Total Valid Annotations:    {total_boxes} (Crop: {crop_count}, Weed: {weed_count})")
    print(f"  Invalid Bounding Boxes:     {invalid_boxes}")
    print(f"=======================================================\n")

    # 2. Check Data Leakage across splits
    print("[Validation] Checking for duplicate source image content across splits...")
    extracted_base = "dataset_agridatavalue/extracted"
    source_hashes = {}
    split_leakage = []

    for sname in ["train", "valid", "test"]:
        sdir = os.path.join(extracted_base, sname)
        for root, _, files in os.walk(sdir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(root, f)
                    with open(fpath, "rb") as fp:
                        h = hashlib.md5(fp.read()).hexdigest()

                    if h in source_hashes:
                        split_leakage.append((f, sname, source_hashes[h]))
                    else:
                        source_hashes[h] = (f, sname)

    if split_leakage:
        print(f"[Warning] Found {len(split_leakage)} duplicate source image instances across splits!")
        for f, s1, (orig_f, s2) in split_leakage[:5]:
            print(f"  - Image '{f}' in split '{s1}' is identical to '{orig_f}' in split '{s2}'")
    else:
        print("[Validation] Zero duplicate source image content detected across train/val/test splits.")

    return {
        "total_tiles": total_tiles,
        "orphan_images": orphan_images,
        "orphan_labels": orphan_labels,
        "untraced_tiles": untraced_tiles,
        "total_boxes": total_boxes,
        "crop_count": crop_count,
        "weed_count": weed_count,
        "invalid_boxes": invalid_boxes,
        "split_leakage_count": len(split_leakage)
    }

if __name__ == "__main__":
    validate_tiles()
