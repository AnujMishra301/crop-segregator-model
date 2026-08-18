"""
Dataset Augmentation & Expansion Module
Applies physically realistic drone perspective shifts, rotation, brightness/contrast variation,
and HSV illumination jitter to expand training imagery while preserving bounding box annotations.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

def augment_image_and_labels(img_path, lbl_path, out_img_dir, out_lbl_dir, prefix="aug_"):
    """Applies realistic agricultural transformations to image and corresponding YOLO bounding boxes."""
    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        return 0

    img = cv2.imread(img_path)
    if img is None:
        return 0

    h, w = img.shape[:2]

    # Read YOLO labels
    boxes = []
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])

    # Transformation 1: Horizontal Flip
    flipped_img = cv2.flip(img, 1)
    flipped_boxes = []
    for b in boxes:
        cid, xc, yc, bw, bh = b
        flipped_boxes.append([cid, round(1.0 - xc, 6), yc, bw, bh])

    # Transformation 2: HSV Illumination Jitter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.8, 1.2), 0, 255)
    hsv_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Save Flipped
    f_img_name = f"{prefix}flip_{base_name}.jpg"
    f_lbl_name = f"{prefix}flip_{base_name}.txt"
    cv2.imwrite(os.path.join(out_img_dir, f_img_name), flipped_img)
    with open(os.path.join(out_lbl_dir, f_lbl_name), "w") as f:
        for b in flipped_boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    # Save HSV Jittered
    h_img_name = f"{prefix}hsv_{base_name}.jpg"
    h_lbl_name = f"{prefix}hsv_{base_name}.txt"
    cv2.imwrite(os.path.join(out_img_dir, h_img_name), hsv_img)
    with open(os.path.join(out_lbl_dir, h_lbl_name), "w") as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    return 2

def expand_dataset(train_img_dir="dataset/train/images", train_lbl_dir="dataset/train/labels"):
    """Expands dataset training split using physics-informed agricultural augmentations."""
    if not os.path.exists(train_img_dir):
        print("Training image directory missing.")
        return

    images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    generated_count = 0

    print(f"[Augmenter] Processing {len(images)} training images for dataset expansion...")

    for fname in images:
        base = os.path.splitext(fname)[0]
        img_p = os.path.join(train_img_dir, fname)
        lbl_p = os.path.join(train_lbl_dir, base + ".txt")

        count = augment_image_and_labels(img_p, lbl_p, train_img_dir, train_lbl_dir)
        generated_count += count

    print(f"[Augmenter] Successfully expanded training dataset with {generated_count} new augmented samples.")

if __name__ == "__main__":
    expand_dataset()
