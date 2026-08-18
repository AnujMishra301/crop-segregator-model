"""
Dataset V2 Detailed Collection Statistics Tool
Analyzes Dataset V2 images, YOLO label files, and DATA_COLLECTION_TRACKER.csv.
Computes granular statistics across 12 collection metrics, environmental splits, and target categories.
"""

import os
import sys
import argparse
import pandas as pd
import cv2

CLASS_NAMES = {0: "weed", 1: "crop", 2: "grass_lawn", 3: "other"}

def analyze_dataset(img_dir="dataset_v2/images", lbl_dir="dataset_v2/labels", tracker_path="dataset_v2/DATA_COLLECTION_TRACKER.csv"):
    """Computes comprehensive statistics for Dataset V2."""
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        # Fallback check on dataset/ if dataset_v2/ is empty
        if os.path.exists("dataset/train/images"):
            print("[Notice] Analyzing baseline 'dataset/' directory as dataset_v2 is currently empty...")
            img_dir = "dataset/train/images"
            lbl_dir = "dataset/train/labels"
        else:
            return {"error": "Dataset image/label directories missing."}

    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(images)

    stats = {
        "total_images": total_images,
        "images_with_weed": 0,
        "images_with_crop": 0,
        "weed_only_images": 0,
        "crop_only_images": 0,
        "weed_and_crop_images": 0,
        "weed_and_grass_images": 0,
        "multiple_weed_images": 0,
        "small_weed_images": 0,
        "negative_no_weed_images": 0,
        "total_weed_boxes": 0,
        "total_crop_boxes": 0,
        "total_grass_boxes": 0,
        "total_other_boxes": 0,
        "capture_sessions": {},
        "fields": {},
        "environmental_conditions": {}
    }

    for fname in images:
        base = os.path.splitext(fname)[0]
        img_p = os.path.join(img_dir, fname)
        lbl_p = os.path.join(lbl_dir, base + ".txt")

        img = cv2.imread(img_p)
        h, w = (img.shape[0], img.shape[1]) if img is not None else (640, 640)

        weeds = 0
        crops = 0
        grass = 0
        others = 0
        has_small_weed = False

        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        bw, bh = float(parts[3]), float(parts[4])
                        box_w_px = bw * w
                        box_h_px = bh * h

                        if cid == 0:
                            weeds += 1
                            stats["total_weed_boxes"] += 1
                            if box_w_px < 32.0 or box_h_px < 32.0:
                                has_small_weed = True
                        elif cid == 1:
                            crops += 1
                            stats["total_crop_boxes"] += 1
                        elif cid == 2:
                            grass += 1
                            stats["total_grass_boxes"] += 1
                        elif cid == 3:
                            others += 1
                            stats["total_other_boxes"] += 1

        if weeds > 0:
            stats["images_with_weed"] += 1
        else:
            stats["negative_no_weed_images"] += 1

        if crops > 0:
            stats["images_with_crop"] += 1

        if weeds > 0 and crops == 0 and grass == 0:
            stats["weed_only_images"] += 1

        if crops > 0 and weeds == 0:
            stats["crop_only_images"] += 1

        if weeds > 0 and crops > 0:
            stats["weed_and_crop_images"] += 1

        if weeds > 0 and grass > 0:
            stats["weed_and_grass_images"] += 1

        if weeds >= 2:
            stats["multiple_weed_images"] += 1

        if has_small_weed:
            stats["small_weed_images"] += 1

    # Load tracker metadata if available
    if os.path.exists(tracker_path):
        try:
            df = pd.read_csv(tracker_path)
            if not df.empty:
                if "capture_session" in df.columns:
                    stats["capture_sessions"] = df["capture_session"].value_counts().to_dict()
                if "field_id" in df.columns:
                    stats["fields"] = df["field_id"].value_counts().to_dict()
                if "lighting" in df.columns:
                    stats["environmental_conditions"] = df["lighting"].value_counts().to_dict()
        except Exception:
            pass

    return stats

def print_stats_report(stats):
    """Prints clean markdown-style collection statistics report."""
    if "error" in stats:
        print(f"[Error] {stats['error']}")
        return

    tot = stats["total_images"]
    print(f"\n=======================================================")
    print(f" DATASET COLLECTION STATISTICS REPORT")
    print(f"=======================================================\n")
    print(f" Total Images Evaluated:       {tot}")
    print(f" ------------------------------------------------------")
    print(f" Images Containing Weeds:     {stats['images_with_weed']} ({(stats['images_with_weed']/tot*100 if tot else 0):.1f}%)")
    print(f" Images Containing Crops:     {stats['images_with_crop']} ({(stats['images_with_crop']/tot*100 if tot else 0):.1f}%)")
    print(f" Weed + Crop Images (Crucial):{stats['weed_and_crop_images']} ({(stats['weed_and_crop_images']/tot*100 if tot else 0):.1f}%)")
    print(f" Weed + Grass Images:         {stats['weed_and_grass_images']} ({(stats['weed_and_grass_images']/tot*100 if tot else 0):.1f}%)")
    print(f" Crop-Only Images (Negatives):{stats['crop_only_images']} ({(stats['crop_only_images']/tot*100 if tot else 0):.1f}%)")
    print(f" Weed-Only Images:            {stats['weed_only_images']} ({(stats['weed_only_images']/tot*100 if tot else 0):.1f}%)")
    print(f" Multiple-Weed Images (>=2):   {stats['multiple_weed_images']} ({(stats['multiple_weed_images']/tot*100 if tot else 0):.1f}%)")
    print(f" Small-Weed Images (<32px):   {stats['small_weed_images']} ({(stats['small_weed_images']/tot*100 if tot else 0):.1f}%)")
    print(f" No-Weed Negative Images:     {stats['negative_no_weed_images']} ({(stats['negative_no_weed_images']/tot*100 if tot else 0):.1f}%)")
    print(f" ------------------------------------------------------")
    print(f" Total Weed BBoxes:           {stats['total_weed_boxes']}")
    print(f" Total Crop BBoxes:           {stats['total_crop_boxes']}")
    print(f" Total Grass BBoxes:          {stats['total_grass_boxes']}")
    print(f" Total Other BBoxes:          {stats['total_other_boxes']}")
    print(f"=======================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Dataset V2 Collection Statistics Tool")
    parser.add_argument("--img_dir", type=str, default="dataset_v2/images", help="Path to images directory")
    parser.add_argument("--lbl_dir", type=str, default="dataset_v2/labels", help="Path to labels directory")
    parser.add_argument("--tracker", type=str, default="dataset_v2/DATA_COLLECTION_TRACKER.csv", help="Path to CSV tracker")

    args = parser.parse_args()
    stats = analyze_dataset(args.img_dir, args.lbl_dir, args.tracker)
    print_stats_report(stats)

if __name__ == "__main__":
    main()
