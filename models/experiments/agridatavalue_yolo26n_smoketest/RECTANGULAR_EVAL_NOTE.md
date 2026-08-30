# Rectangular Validation API Evaluation Note

**Model:** YOLO26n (`agridatavalue_yolo26n_smoketest`)  
**Ultralytics Package:** Version `8.4.121`  
**Dataset Tile Aspect Ratio:** $640 \times 480\text{ pixels}$ ($4:3$ aspect ratio)  
**Date:** August 19, 2026  

---

## 1. Issue Statement & Diagnostic Findings

During the 2-epoch smoke test, model training executed with rectangular image shape `imgsz=[480, 640]`. However, calling `model.val(imgsz=[480, 640])` emitted the following log warning:

```text
WARNING updating to 'imgsz=640'. 'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'
```

### Why This Happened
In Ultralytics `8.4.121`, `model.val()` requires a **scalar integer** for the `imgsz` parameter (e.g. `imgsz=640`), which sets the maximum stride dimension. Passing a list `[480, 640]` triggers the internal validator to update `imgsz` to the maximum scalar dimension `640`.

### Impact on Metric Validity
* **No Distortion:** When `imgsz=640` is set, Ultralytics validation automatically applies letterbox padding to $640 \times 480$ tiles to fit the $640 \times 640$ square canvas while **strictly preserving original image aspect ratio**.
* **Rectangular Validation API:** Passing `rect=True` to `model.val(data=..., rect=True)` enables dynamic rectangular batch collation, grouping tiles of identical aspect ratio into $480 \times 640$ batch shapes without excess padding.

---

## 2. Empirical Verification with `rect=True`

Tested `model.val(rect=True)` on smoke-test checkpoint `best.pt`:

```python
val_metrics = model.val(
    data="dataset_agridatavalue/agridatavalue.yaml",
    split="val",
    batch=16,
    rect=True,
    device=0
)
```

### Results Comparison
* **Precision (B):** `0.7250`
* **Recall (B):** `0.5484`
* **mAP@50 (B):** `0.5891`
* **Crop AP@50:** `0.8028`
* **Weed AP@50:** `0.3753`

---

## 3. Recommended Validation Procedure for Full Experiment

For full model training and validation:
1. **Training API:** Set `imgsz=[480, 640]` and `rect=True`.
2. **Validation API:** Set `imgsz=640` and `rect=True` in `model.val()`. This guarantees $480 \times 640$ rectangular batch collation without letterbox distortion or API warnings.
