# Full Model Training Plan — YOLO26n on AgriDataValue Dataset

**Experiment Identifier:** `agridatavalue_yolo26n_full`  
**Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)  
**Dataset Configuration:** [`dataset_agridatavalue/agridatavalue.yaml`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/dataset_agridatavalue/agridatavalue.yaml)  
**Report Date:** August 19, 2026  

---

## 1. Executive Summary & Training Objectives

The 2-epoch smoke test verified that the CUDA training environment, `yolo26n.pt` pretrained checkpoint, $640 \times 480$ rectangular image batching, and dataset YAML function properly on the target NVIDIA RTX 3050 GPU (Peak VRAM: $2.37\text{ GB}$).

This document establishes the official **50-Epoch Full Training Plan** for `MODEL V2 (YOLO26n)` to build a clean two-class experimental baseline (`0: Crop`, `1: Weed`) on the AgriDataValue tiled dataset.

---

## 2. Complete Hyperparameter & Training Configuration

```yaml
# Model & Checkpoint
model: yolo26n.pt
data: dataset_agridatavalue/agridatavalue.yaml

# Training & Resolution Controls
epochs: 50
batch: 16
imgsz: [480, 640]  # 480 height x 640 width rectangular training
rect: true         # Rectangular batching
cache: false
patience: 10       # Early stopping patience

# Hardware & System
device: 0          # CUDA:0 (NVIDIA RTX 3050)
workers: 2
seed: 42           # Fixed seed for 100% reproducibility

# Optimization & Learning Rate
optimizer: auto    # Ultralytics AdamW / SGD auto-selection
lr0: 0.01          # Initial learning rate
lrf: 0.01          # Final learning rate fraction
weight_decay: 0.0005

# Data Augmentations
mosaic: 1.0        # Mosaic data augmentation ratio
mixup: 0.0         # Mixup augmentation disabled for clear bounding box boundaries
fliplr: 0.5        # Horizontal flip
degrees: 0.0       # Rotation degrees disabled (nadir drone flight orientation)
scale: 0.5         # Gain scale ratio

# Output Location
project: models/experiments
name: agridatavalue_yolo26n_full
```

---

## 3. Resource & VRAM Headroom Verification

* **Available GPU Memory:** $4,096\text{ MB}$ ($4.00\text{ GB}$)
* **Smoke Test Measured VRAM (Batch 16):** $2,367.6\text{ MB}$ ($2.37\text{ GB}$)
* **Headroom Safety Buffer:** $1,728.4\text{ MB}$ ($42.2\%$ safety headroom)
* **Batch Size Recommendation:** Keep **`batch=16`**. This maximizes CUDA tensor throughput while maintaining complete VRAM stability against out-of-memory errors.

---

## 4. Class Imbalance Analysis & Recommendation

* **Annotation Distribution:** Crop: `161,868` ($82.3\%$), Weed: `34,782` ($17.7\%$). Ratio: **$4.65 : 1$** Crop-to-Weed imbalance.
* **Analysis:** While crops outnumber weeds $4.65:1$, there are `34,782` weed annotations in the training split, providing ample positive weed instances. Standard YOLO CIoU box loss and Task-Aligned Assigner (TAL) effectively handle this level of class distribution.
* **Recommendation:** Proceed with standard Ultralytics loss functions for the baseline run. Do not add artificial loss re-weighting until baseline metrics are established.

---

## 5. Strict Test Set Protection Guarantee

> [!IMPORTANT]
> **TEST SET ISOLATION:**  
> The test set (`dataset_agridatavalue/tiled/test/`) is **STRICTLY EXCLUDED** from the training loop, validation checkpoints, and early-stopping selection. Model selection will rely exclusively on the `val` split (`3,812` patches). The test set (`1,561` patches) will be evaluated **EXACTLY ONCE** after full training concludes.

---

## 6. Execution Command

When full training is authorized, run:

```bash
.venv\Scripts\python.exe -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); model.train(data='dataset_agridatavalue/agridatavalue.yaml', epochs=50, batch=16, imgsz=[480, 640], rect=True, device=0, workers=2, seed=42, cache=False, patience=10, project='models/experiments', name='agridatavalue_yolo26n_full')"
```
