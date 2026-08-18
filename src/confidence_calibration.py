"""
Confidence Calibration & Operating Threshold Evaluation Module
Sweeps confidence thresholds (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90) across test set detections.
Calculates precision, recall, false weed detection rates, missed weed rates, and spray eligibility.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.config.config_loader import load_config
from ml.utils.metrics import compute_box_iou

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

CLASS_NAMES = {0: "weed", 1: "crop", 2: "grass_lawn", 3: "other"}

def calibrate_confidence(raw_conf, temperature=1.2):
    """Applies temperature scaling calibration to raw neural network confidence logits."""
    # Convert confidence float [0, 1] to logit
    eps = 1e-7
    conf_clamped = max(eps, min(1.0 - eps, raw_conf))
    logit = np.log(conf_clamped / (1.0 - conf_clamped))
    
    # Scale logit by temperature
    calibrated_logit = logit / temperature
    
    # Sigmoid back to probability
    calibrated_conf = 1.0 / (1.0 + np.exp(-calibrated_logit))
    return float(calibrated_conf)

def evaluate_threshold_sweep(weights_path=None, test_img_dir="dataset/test/images", test_lbl_dir="dataset/test/labels"):
    """Evaluates threshold sweep across test dataset."""
    config = load_config()
    weights = weights_path or config["model_save_path"]

    if not os.path.exists(weights) or not ULTRALYTICS_AVAILABLE:
        print("Model weights or Ultralytics library missing.")
        return {}

    model = YOLO(weights)
    img_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    sweep_results = {}

    print(f"[Calibration] Running threshold sweep across {len(img_files)} test images...")

    for thresh in thresholds:
        weed_tp, weed_fp, weed_fn = 0, 0, 0
        crop_fp_as_weed = 0
        eligible_weed_count = 0
        total_weed_gt = 0

        for fname in img_files:
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(test_img_dir, fname)
            lbl_path = os.path.join(test_lbl_dir, base + ".txt")

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            # Ground truth weeds
            gt_weeds = []
            gt_crops = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:])
                            xmin = (xc - bw / 2.0) * w
                            ymin = (yc - bh / 2.0) * h
                            xmax = (xc + bw / 2.0) * w
                            ymax = (yc + bh / 2.0) * h
                            box_dict = {"bbox": [xmin, ymin, xmax, ymax], "cls_id": cls_id}
                            if cls_id == 0:
                                gt_weeds.append(box_dict)
                                total_weed_gt += 1
                            elif cls_id == 1:
                                gt_crops.append(box_dict)

            # Predict
            results = model.predict(img_path, conf=thresh, iou=0.45, verbose=False)[0]
            preds = []
            if results.boxes is not None and len(results.boxes) > 0:
                b_xyxy = results.boxes.xyxy.cpu().numpy()
                b_cls = results.boxes.cls.cpu().numpy().astype(int)
                b_conf = results.boxes.conf.cpu().numpy()
                for box, cls_id, conf in zip(b_xyxy, b_cls, b_conf):
                    calib_c = calibrate_confidence(float(conf))
                    preds.append({"bbox": box.tolist(), "cls_id": int(cls_id), "conf": calib_c})

            # Check weed predictions
            pred_weeds = [p for p in preds if p["cls_id"] == 0 and p["conf"] >= thresh]
            eligible_weed_count += len(pred_weeds)

            matched_gt = set()
            for p in pred_weeds:
                best_iou, best_gt_idx = 0.0, -1
                for g_idx, g in enumerate(gt_weeds):
                    iou = compute_box_iou(p["bbox"], g["bbox"])
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, g_idx

                if best_iou >= 0.45 and best_gt_idx != -1:
                    weed_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    weed_fp += 1
                    # Check if over crop
                    if any(compute_box_iou(p["bbox"], c["bbox"]) > 0.25 for c in gt_crops):
                        crop_fp_as_weed += 1

            weed_fn += (len(gt_weeds) - len(matched_gt))

        weed_prec = weed_tp / (weed_tp + weed_fp) if (weed_tp + weed_fp) > 0 else 0.0
        weed_rec = weed_tp / (weed_tp + weed_fn) if (weed_tp + weed_fn) > 0 else 0.0
        false_weed_det_rate = weed_fp / (weed_tp + weed_fp) if (weed_tp + weed_fp) > 0 else 0.0
        missed_weed_rate = weed_fn / total_weed_gt if total_weed_gt > 0 else 0.0

        sweep_results[thresh] = {
            "weed_precision": weed_prec,
            "weed_recall": weed_rec,
            "false_weed_det_rate": false_weed_det_rate,
            "missed_weed_rate": missed_weed_rate,
            "crop_false_sprays": crop_fp_as_weed,
            "spray_eligible_count": eligible_weed_count
        }

    return sweep_results

if __name__ == "__main__":
    res = evaluate_threshold_sweep()
    print("\nCONFIDENCE THRESHOLD SWEEP EVALUATION:")
    for t, m in res.items():
        print(f"  Thresh {t:.2f}: Precision={m['weed_precision']:.4f}, Recall={m['weed_recall']:.4f}, Spray Eligible={m['spray_eligible_count']}")
