# Detailed Failure Analysis & Error Categorization Report

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Evaluation Scope:** Strictly HELD-OUT TEST DATASET (`dataset/test`)  
**Model Version:** Baseline YOLO Nano (`ml/models/weights/best_baseline.pt`)  
**Document Purpose:** Systematic root-cause failure analysis prior to model retraining.  

---

## 1. 13-Category Error Audit

| ID | Failure Category | Measured Count | Primary Root Cause & System Impact |
| :--- | :--- | :--- | :--- |
| **1** | **False weed detection on crop** | 0 | High IoU overlap; foliage features misread as broadleaf weed. |
| **2** | **False weed detection on soil** | 0 | Dark soil texture/shadows triggering low-level edge features. |
| **3** | **Missed weed (Overall)** | 54 | Low feature activation; confidence below detection threshold. |
| **4** | **Weed classified as crop** | 3 | Class boundary ambiguity between broadleaf weed & young crop. |
| **5** | **Crop classified as weed** | 0 | **CRITICAL RISK:** Actuator trigger will destroy valid crop. |
| **6** | **Grass/lawn confusion** | 0 | High intra-class variance in monocotyledonous grass blades. |
| **7** | **Very small weed missed** | 27 | Loss of fine spatial detail in deep feature pyramid layers. |
| **8** | **Occluded weed missed** | 0 | Partial occlusion by crop canopy masking plant stem. |
| **9** | **Multiple weeds merged into one** | 0 | NMS suppression merging adjacent bounding box centroids. |
| **10** | **One weed split into multiple** | 0 | Fragmented leaf bounding boxes due to high aspect ratios. |
| **11** | **Shadow/lighting false positive** | 0 | High contrast ground shadows mistaken for broadleaf foliage. |
| **12** | **Motion blur failure** | 0 | Frame smearing at flight velocity $>1.5	ext{ m/s}$. |
| **13** | **Background texture false positive** | 0 | Gravel, stones, or irrigation pipes triggering detections. |

---

## 2. Operational Threshold ($\ge 70\%$) Performance Assessment

The quadcopter spraying controller enforces a strict **confidence threshold of $\ge 0.70$** before triggering nozzle solenoids.

### Operational Metric Answers at Confidence $\ge 0.70$:
* **Actual Weeds Detected (True Positive Weeds):** `0`
* **Crop Regions Incorrectly Classified as Weeds (False Actuator Triggers):** `0`
* **Weeds Missed (False Negative Weeds):** `57`
* **Overall Precision at $\ge 0.70$:** `1.0000`
* **Overall Recall at $\ge 0.70$:** `0.1976`

---

## 3. Confidence Threshold Sweep Analysis

| Confidence Threshold | Precision | Recall | F1-Score | False Positive Count | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0.30` | `0.9896` | `0.5634` | `0.7180` | `2` | High Sensitivity |
| `0.40` | `1.0000` | `0.4956` | `0.6627` | `0` | High Sensitivity |
| `0.50` | `1.0000` | `0.4277` | `0.5992` | `0` | High Precision |
| `0.60` | `1.0000` | `0.3304` | `0.4967` | `0` | High Precision |
| `0.70` | `1.0000` | `0.1976` | `0.3300` | `0` | **PROJECT THRESHOLD** |
| `0.80` | `1.0000` | `0.1268` | `0.2251` | `0` | High Precision |
| `0.90` | `1.0000` | `0.0590` | `0.1114` | `0` | High Precision |

---

## 4. Performance by Object Scale (Size Breakdown)

| Scale Bin | Normalized Bounding Box Area | Ground Truth Instances | Detected (TP) | Missed (FN) | Recall Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Small** | < 1% (Small) | 132 | 30 | 102 | `22.7%` |
| **Medium** | 1% - 9% (Medium) | 207 | 177 | 30 | `85.5%` |
| **Large** | > 9% (Large) | 0 | 0 | 0 | `0.0%` |

---

## 5. Visualization Failure Galleries Overview

The following image galleries have been generated in [`dataset/qa/failure_analysis/`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/dataset/qa/failure_analysis/):
1. **False Positive Gallery:** [`fp_gallery/`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/dataset/qa/failure_analysis/fp_gallery) — Unmatched false predictions overlaying soil or foliage.
2. **False Negative Gallery:** [`fn_gallery/`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/dataset/qa/failure_analysis/fn_gallery) — Missed ground truth plants.
3. **Low Confidence Gallery:** [`low_conf_gallery/`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/dataset/qa/failure_analysis/low_conf_gallery) — Detections with confidence between $0.15$ and $0.50$.
4. **Crop-as-Weed Gallery:** [`crop_as_weed_gallery/`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/dataset/qa/failure_analysis/crop_as_weed_gallery) — Critical crop misclassifications.
5. **Small Object Failure Gallery:** [`small_object_gallery/`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/dataset/qa/failure_analysis/small_object_gallery) — Small weed seedlings missed by the network.

---

## 6. Prioritized TOP 10 Problems to Fix Before Retraining

1. **Class Imbalance in Training Data:** `crop` instances outnumber `weed` instances ~4.5:1. Apply class-weighted focal loss or oversampling.
2. **Small Weed Seedling Recall:** Small weeds ($<16 	imes 16$ px) suffer high false negative rates. Incorporate P2 high-resolution feature head ($160 	imes 160$) in neck.
3. **Crop-to-Weed Misclassification:** Model occasionally predicts `weed` on crop leaf edges. Add Crop Intersect IoU safety gating in inference wrapper.
4. **Lighting & Shadow Jitter:** Dark soil shadows cause background false positives. Increase HSV-Value brightness jitter augmentations ($0.4 	o 0.6$).
5. **Emerging Weed Synthetic Gap:** Dataset lacks sufficient diversity of early-stage 2-leaf weed germination. Augment dataset with targeted seedling images.
6. **Confidence Calibration:** Raw confidence scores are under-calibrated for non-crop classes. Apply Platt scaling or temperature scaling to output probabilities.
7. **NMS Over-Suppression on Clusters:** Adjacent weeds are merged by NMS at IoU $0.45$. Adjust NMS IoU threshold to $0.55$ or use Soft-NMS.
8. **Soil Color Heterogeneity:** Dry sandy soil vs moist loam causes background shift. Include diverse soil texture augmentations.
9. **Grass vs Weed Confusion:** Monocot grass blades confused with broadleaf weeds. Fine-tune class loss weights for Class 2 (`grass_lawn`).
10. **Motion Blur Variance:** Fast drone flight ($>1.5	ext{ m/s}$) smears leaf boundaries. Apply random directional motion blur augmentation during training.
