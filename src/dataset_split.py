"""
Sequence-Grouped Dataset Splitter Module
Implements leakage-safe dataset splitting (70% train, 20% val, 10% test).
Groups images by field/video sequence to ensure near-identical consecutive drone frames
from the same field remain in the SAME split, eliminating spatial data leakage.
"""

import os
import shutil
import random
from collections import defaultdict

def extract_field_group(filename):
    """Extracts field/sequence identifier from filename.
    e.g. 'field01_frame005.jpg' -> 'field01'
         'seqA_frame10.png' -> 'seqA'
    Fallback: Uses prefix before first underscore, or whole base name.
    """
    base = os.path.splitext(filename)[0]
    if '_' in base:
        parts = base.split('_')
        # Return prefix up to last underscore or first component matching field/seq pattern
        for part in parts:
            if 'field' in part.lower() or 'seq' in part.lower():
                return part
        return parts[0]
    return base

def perform_grouped_split(raw_img_dir="dataset/raw/images", raw_lbl_dir="dataset/raw/labels",
                          output_base_dir="dataset", split_ratio=(0.70, 0.20, 0.10), seed=42):
    """Splits dataset into train/val/test splits without field data leakage."""
    random.seed(seed)
    
    if not os.path.exists(raw_img_dir) or not os.path.exists(raw_lbl_dir):
        print(f"Raw image directory '{raw_img_dir}' or label directory '{raw_lbl_dir}' missing.")
        return

    # Pair valid images and labels
    img_files = {os.path.splitext(f)[0]: f for f in os.listdir(raw_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))}
    lbl_files = {os.path.splitext(f)[0]: f for f in os.listdir(raw_lbl_dir) if f.lower().endswith('.txt')}
    valid_bases = sorted(list(set(img_files.keys()).intersection(set(lbl_files.keys()))))

    print(f"Grouped Dataset Split: Found {len(valid_bases)} valid image-label pairs.")

    # Group bases by sequence/field
    groups = defaultdict(list)
    for base in valid_bases:
        group_id = extract_field_group(base)
        groups[group_id].append(base)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    print(f"Identified {len(group_keys)} distinct sequence groups: {group_keys}")

    # Calculate group splits based on total image counts
    total_imgs = len(valid_bases)
    target_train = total_imgs * split_ratio[0]
    target_val = total_imgs * split_ratio[1]
    
    train_bases, val_bases, test_bases = [], [], []
    curr_train_cnt, curr_val_cnt = 0, 0

    for g in group_keys:
        g_bases = groups[g]
        g_size = len(g_bases)
        
        if curr_train_cnt + g_size <= target_train or len(train_bases) == 0:
            train_bases.extend(g_bases)
            curr_train_cnt += g_size
        elif curr_val_cnt + g_size <= target_val or len(val_bases) == 0:
            val_bases.extend(g_bases)
            curr_val_cnt += g_size
        else:
            test_bases.extend(g_bases)

    print(f"Split Distribution by Sequence:")
    print(f"  Train: {len(train_bases)} images ({len(train_bases)/total_imgs*100:.1f}%)")
    print(f"  Val:   {len(val_bases)} images ({len(val_bases)/total_imgs*100:.1f}%)")
    print(f"  Test:  {len(test_bases)} images ({len(test_bases)/total_imgs*100:.1f}%)")

    # Create destination directories
    splits = {
        'train': train_bases,
        'val': val_bases,
        'test': test_bases
    }

    for split_name, bases in splits.items():
        split_img_dir = os.path.join(output_base_dir, split_name, "images")
        split_lbl_dir = os.path.join(output_base_dir, split_name, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        for base in bases:
            src_img = os.path.join(raw_img_dir, img_files[base])
            src_lbl = os.path.join(raw_lbl_dir, lbl_files[base])
            dst_img = os.path.join(split_img_dir, img_files[base])
            dst_lbl = os.path.join(split_lbl_dir, lbl_files[base])

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

    print("Dataset splitting completed successfully.")

if __name__ == "__main__":
    perform_grouped_split()
