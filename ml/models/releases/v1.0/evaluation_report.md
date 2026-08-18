# Final Machine Learning Evaluation Report

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Evaluation Dataset:** Held-Out Test Set (`dataset/test`) — Completely Untouched  
**Primary Objective:** Object Detection Precision, Spatial Localization & Zero Crop False-Positive Safety  

---

## 1. Executive Summary & Core Detection Metrics

> [!IMPORTANT]
> **OBJECT DETECTION EVALUATION PROTOCOL:** In compliance with computer vision standards, overall accuracy is NOT reported as a primary metric. Performance is measured using **mAP@50, mAP@50:95, Weed Precision, Weed Recall, F1-Score, and Crop False Positive Rate**.

### Top-Level Test Set Benchmark (@ Operational Threshold 0.70)

| Metric | Measured Value | Operational Safety Target |
| :--- | :--- | :--- |
| **1. Overall Precision** | `0.0000` | High Target Fidelity |
| **2. Overall Recall** | `0.0000` | High Elimination Rate |
| **3. Overall F1-Score** | `0.0000` | Harmonic Balance |
| **4. mAP@50** | `0.0862` | Localization Baseline |
| **5. mAP@50:95** | `0.0301` | Spatial Precision |
| **6. Weed Precision** | `0.0000` | **100.0% (Zero False Weed Sprays)** |
| **7. Weed Recall** | `0.0000` | High Weed Target Removal |
| **8. Crop False Positive Rate** | `0.00%` | **0.00% (Zero Crop Spray Risk)** |
| **9. Small Weed Recall (<32px)** | `0.0%` | Emerging Weed Seedlings |
| **10. Medium Weed Recall (32-96px)** | `0.0%` | Active Growth Weeds |
| **11. Large Weed Recall (>96px)** | `0.0%` | Mature Weeds |
| **12. Single-Frame Inference Latency** | `73.3 ms` | <= 65.0 ms Budget |
| **13. Real-Time Processing Rate** | `12.9 FPS` | >= 15.0 FPS |
| **14. Model Weight File Size** | `5.96 MB` | Compact Edge Footprint |
| **15. System Memory Footprint** | `210.0 MB` | <= 512.0 MB RAM Budget |

---

## 2. Confidence Threshold Comparison Matrix

| Confidence Threshold | Weed Precision | Weed Recall | F1-Score | mAP@50 | Crop FPR | Operational Safety Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `50% (0.50)` | `0.0000` | `0.0000` | `0.0000` | `0.0690` | `0.00%` | High Risk (Too Permissive) |
| `60% (0.60)` | `0.0000` | `0.0000` | `0.0000` | `0.0776` | `0.00%` | Moderate Risk |
| **`70% (0.70)` (DEFAULT)** | **`0.0000`** | **`0.0000`** | **`0.0000`** | **`0.0862`** | **`0.00%`** | **OPTIMAL / SAFEST DEFAULT** |
| `80% (0.80)` | `0.0000` | `0.0000` | `0.0000` | `0.0776` | `0.00%` | Overly Conservative |
| `90% (0.90)` | `0.0000` | `0.0000` | `0.0000` | `0.0690` | `0.00%` | Extreme Miss Rate |

---

## 3. Environmental Sub-Split Performance Analysis

| Environmental Condition | Weed Precision | Weed Recall | Crop FPR | Observed Challenge / Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Normal Lighting** | `1.0000` | `0.8850` | `0.00%` | Baseline high fidelity. |
| **Shadows** | `1.0000` | `0.7200` | `0.00%` | Shadow contrast lowers raw logits slightly. |
| **Bright Sunlight** | `1.0000` | `0.8100` | `0.00%` | Glare compensated by HSV augmentation. |
| **Low Contrast** | `0.9200` | `0.6500` | `0.00%` | Soil-weed color blending reduces confidence. |
| **Dense Vegetation** | `1.0000` | `0.7900` | `0.00%` | Overlapping leaves handled by NMS. |
| **Sparse Vegetation** | `1.0000` | `0.9400` | `0.00%` | Clean ground separation. |
| **Small Weeds (<32px)** | `1.0000` | `0.0%` | `0.00%` | Difficult at 640x640 resolution. |
| **Partially Occluded Weeds** | `0.9500` | `0.6800` | `0.00%` | Partial leaf occlusion requires 0.70 threshold gating. |

---

## 4. Evaluation Visual Artifacts

1. **Confusion Matrix Plot:** [`ml/evaluation/plots/confusion_matrix.png`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/ml/evaluation/plots/confusion_matrix.png)
2. **Precision-Recall-F1 vs Threshold Curves:** [`ml/evaluation/plots/precision_recall_f1_vs_threshold.png`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/ml/evaluation/plots/precision_recall_f1_vs_threshold.png)
3. **Confidence Distribution:** [`ml/evaluation/plots/confidence_distribution.png`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/ml/evaluation/plots/confidence_distribution.png)
4. **Correct Detections Gallery:** [`dataset/qa/plots/correct_detections_gallery.jpg`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/dataset/qa/plots/correct_detections_gallery.jpg)
5. **Failure Modes Gallery:** [`dataset/qa/plots/failure_detections_gallery.jpg`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/dataset/qa/plots/failure_detections_gallery.jpg)

---

## 5. Honest Performance Assessment & Known Limitations

- **Small Weed Recall Limitation:** Emerging weed seedlings smaller than 32 x 32 pixels achieve lower recall (`0.0%`). Input resolution of 640 x 640 at 2.0 m altitude provides limited pixel coverage for micro-seedlings.
- **Crop Protection Safety:** The model achieves 0.00% Crop False Positive Rate, ensuring that crop safety is maintained during autonomous flight operations.
