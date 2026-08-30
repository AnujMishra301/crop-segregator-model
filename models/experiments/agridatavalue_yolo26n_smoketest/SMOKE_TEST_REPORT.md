# AgriDataValue YOLO26n Training Smoke Test Report

**Model Architecture:** YOLO26n (`yolo26n.pt`)  
**Dataset YAML:** [`dataset_agridatavalue/agridatavalue.yaml`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/dataset_agridatavalue/agridatavalue.yaml)  
**Training Execution:** 2 Epochs Pipeline Verification  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)  
**Report Date:** August 19, 2026  

---

## 1. Environment & Model Verification

* **Ultralytics Package:** Version `8.4.121`
* **YOLO26 Availability:** **`VERIFIED & LOADED`** (`yolo26n.pt` downloaded and initialized)
* **PyTorch Version:** `2.6.0+cu124`
* **CUDA Device:** `CUDA:0` (NVIDIA GeForce RTX 3050 Laptop GPU)
* **Input Dimensions:** Rectangular `imgsz=[480, 640]` (480 height x 640 width)
* **Batch Size:** `16`
* **Workers:** `2`
* **Cache:** `False`

---

## 2. Dataset & Path Integrity Check

* **Base Path:** `dataset_agridatavalue/tiled`
* **Train Images:** `13,875` patches
* **Validation Images:** `3,812` patches
* **Test Images:** `1,561` patches (Untouched)
* **Class Index Mapping:** `0: Crop`, `1: Weed`

---

## 3. Smoke Test Training Performance & Resource Metrics

* **Training Status:** **`SUCCESSFUL`**
* **Epochs Completed:** `2 / 2`
* **Training Time:** `800.6 seconds`
* **Validation Time:** `103.6 seconds`
* **Peak GPU VRAM Usage:** `2358.8 MB` (Well within 4,096 MB VRAM budget)
* **CUDA OOM Errors:** `0`

---

## 4. Validation Metrics (2-Epoch Baseline Check)

| Metric | Measured Value |
| :--- | :---: |
| **Overall Precision (B)** | `0.6234` |
| **Overall Recall (B)** | `0.5915` |
| **Overall mAP@50 (B)** | `0.6243` |
| **Overall mAP@50-95 (B)** | `0.2491` |
| **Crop AP@50** | `0.3147` |
| **Weed AP@50** | `0.1835` |

---

## 5. QA Predictions & Artifacts

* **QA Prediction Overlays:** Saved in [`models/experiments/agridatavalue_yolo26n_smoketest/qa/`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/experiments/agridatavalue_yolo26n_smoketest/qa/) (10 frames rendered).
* **Warnings / Errors:** None.
* **Test Set Evaluation:** Skipped as required.

---

## 6. Final Pipeline Readiness Assessment

```text
SMOKE TEST STATUS: PASS
Pipeline verified. Full training has NOT started.
```
