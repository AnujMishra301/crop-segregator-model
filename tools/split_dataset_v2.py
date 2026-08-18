"""
Sequence-Aware Dataset V2 Split Engine
Splits dataset_v2/images and dataset_v2/labels into train (70%), val (20%), test (10%).
Prevents temporal/spatial data leakage by keeping frames from the same video capture sequence together.
"""

import os
import sys
import shutil
import random
import argparse

def get_sequence_prefix(filename):
    """Extracts capture sequence prefix from image filename."""
    base = os.path.splitext(filename)[0]
    if "_f" in base:
        return base.split("_f")[0]
    parts = base.split("_")
    if len(parts) > 1:
        return parts[0]
    return base

def split_dataset(img_dir="dataset_v2/images", lbl_dir="dataset_v2/labels", output_base="dataset_v2", train_ratio=0.70, val_ratio=0.20, test_ratio=0.10, seed=42):
    """Splits images into train/val/test using sequence-level grouping."""
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"[Notice] Source directories missing: '{img_dir}' or '{lbl_dir}'.")
        return

    random.seed(seed)
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print(f"[Notice] No images found in '{img_dir}'.")
        return

    # Group images by capture sequence prefix
    sequence_groups = {}
    for img_name in images:
        seq_id = get_sequence_prefix(img_name)
        sequence_groups.setdefault(seq_id, []).append(img_name)

    seq_keys = sorted(list(sequence_groups.keys()))
    random.shuffle(seq_keys)

    total_images = len(images)
    train_target = int(total_images * train_ratio)
    val_target = int(total_images * val_ratio)

    train_files, val_files, test_files = [], [], []
    current_train, current_val = 0, 0

    for seq_id in seq_keys:
        seq_imgs = sequence_groups[seq_id]
        n_imgs = len(seq_imgs)

        if current_train + n_imgs <= train_target or (current_train == 0 and train_target > 0):
            train_files.extend(seq_imgs)
            current_train += n_imgs
        elif current_val + n_imgs <= val_target or (current_val == 0 and val_target > 0):
            val_files.extend(seq_imgs)
            current_val += n_imgs
        else:
            test_files.extend(seq_imgs)

    print(f"[SplitEngine] Split Summary Across {len(seq_keys)} Capture Sequences:")
    print(f"  Train Split: {len(train_files)} images ({len(train_files)/total_images*100:.1f}%)")
    print(f"  Val Split:   {len(val_files)} images ({len(val_files)/total_images*100:.1f}%)")
    print(f"  Test Split:  {len(test_files)} images ({len(test_files)/total_images*100:.1f}%)")

    # Copy to destination split folders
    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, file_list in split_map.items():
        dst_img = os.path.join(output_base, split_name, "images")
        dst_lbl = os.path.join(output_base, split_name, "labels")

        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)

        for fname in file_list:
            base = os.path.splitext(fname)[0]
            src_i = os.path.join(img_dir, fname)
            src_l = os.path.join(lbl_dir, base + ".txt")

            shutil.copy(src_i, os.path.join(dst_img, fname))
            if os.path.exists(src_l):
                shutil.copy(src_l, os.path.join(dst_lbl, base + ".txt"))

    print(f"[SplitEngine] Successfully distributed dataset_v2 into train/val/test splits.")

def main():
    parser = argparse.ArgumentParser(description="Sequence-Aware Dataset V2 Split Engine")
    parser.add_argument("--img_dir", type=str, default="dataset_v2/images", help="Path to raw annotated images")
    parser.add_argument("--lbl_dir", type=str, default="dataset_v2/labels", help="Path to raw annotated labels")
    parser.add_argument("--output_base", type=str, default="dataset_v2", help="Base output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sequence shuffling (default: 42)")

    args = parser.parse_args()
    split_dataset(args.img_dir, args.lbl_dir, args.output_base, seed=args.seed)

if __name__ == "__main__":
    main()
