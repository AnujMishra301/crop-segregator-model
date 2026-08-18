"""
Dataset V2 Quality & Readiness Dashboard Engine
Evaluates Dataset V2 progress across 18 priority categories against SIH 2026 deployment targets.
Flags categories with insufficient data and reports overall dataset readiness status.
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from tools.dataset_collection_stats import analyze_dataset
except ImportError:
    from dataset_collection_stats import analyze_dataset

def evaluate_readiness(count, target_low, target_good):
    """Assigns status tag based on empirical count vs targets."""
    if count == 0:
        return "CRITICAL — INSUFFICIENT (0 Count)"
    elif count < target_low:
        return f"CRITICAL — INSUFFICIENT ({count}/{target_good})"
    elif count < target_good:
        return f"MODERATE — IN PROGRESS ({count}/{target_good})"
    else:
        return f"GOOD ({count}/{target_good})"

def generate_dashboard_report(stats):
    """Generates clean ASCII dashboard report."""
    if "error" in stats:
        print(f"[Error] {stats['error']}")
        return

    tot = stats["total_images"]
    weed_crop = stats["weed_and_crop_images"]
    small_weed = stats["small_weed_images"]
    crop_only = stats["crop_only_images"]
    no_weed = stats["negative_no_weed_images"]
    multiple_weed = stats["multiple_weed_images"]

    print(f"\n=========================================================================")
    print(f" DATASET V2 QUALITY & READINESS DASHBOARD")
    print(f"=========================================================================\n")

    print(f" OVERALL DATASET TARGET: 1,000 - 2,000 Real Field Images")
    print(f" Current Total Images:    {tot}")
    print(f" Global Status:           {evaluate_readiness(tot, 500, 1000)}\n")

    print(f" -------------------------------------------------------------------------")
    print(f" PRIORITY CATEGORY READINESS MATRIX")
    print(f" -------------------------------------------------------------------------")
    print(f" [HIGH PRIORITY CATEGORIES]")
    print(f"  1. WEED + CROP (In-Row Weeds):     {evaluate_readiness(weed_crop, 150, 350)}")
    print(f"  2. SMALL WEEDS (<32px Seedlings):  {evaluate_readiness(small_weed, 100, 200)}")
    print(f"  3. CROP ONLY (Difficult Negatives):{evaluate_readiness(crop_only, 80, 150)}")
    print(f"  4. NO WEED (Pure Soil/Crop):      {evaluate_readiness(no_weed, 100, 200)}")
    print(f"  5. MULTIPLE WEEDS (>=2 Targets):   {evaluate_readiness(multiple_weed, 100, 250)}")
    print(f"  6. WEED + GRASS:                   {evaluate_readiness(stats['weed_and_grass_images'], 50, 100)}")
    print(f" -------------------------------------------------------------------------")
    print(f" [ANNOTATION BREAKDOWN]")
    print(f"  - Weed BBoxes:  {stats['total_weed_boxes']}")
    print(f"  - Crop BBoxes:  {stats['total_crop_boxes']} (Weed-to-Crop Ratio: 1 : {(stats['total_crop_boxes']/max(1, stats['total_weed_boxes'])):.1f})")
    print(f"  - Grass BBoxes: {stats['total_grass_boxes']}")
    print(f"  - Other BBoxes: {stats['total_other_boxes']}")
    print(f"=========================================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Dataset V2 Readiness Dashboard Engine")
    parser.add_argument("--img_dir", type=str, default="dataset_v2/images", help="Path to images directory")
    parser.add_argument("--lbl_dir", type=str, default="dataset_v2/labels", help="Path to labels directory")

    args = parser.parse_args()
    stats = analyze_dataset(args.img_dir, args.lbl_dir)
    generate_dashboard_report(stats)

if __name__ == "__main__":
    main()
