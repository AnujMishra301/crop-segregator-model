# Edge AI Inference Module API Specification

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Document Scope:** Hardware-Independent Onboard Computer API Interface Contract  
**Implementation File:** [`tflite_inference.py`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/tflite_inference.py)  

---

## 1. Interface Overview & Architecture

> [!IMPORTANT]
> **DECOUPLING GUARANTEE:** The `TFLiteEdgeInferenceEngine` operates completely independently of physical spray pumps, flight controllers, or MAVLink telemetry streams. It consumes camera frame arrays and outputs structured target detections.

```
Downward Camera Frame ──> TFLiteEdgeInferenceEngine.infer_frame(frame) ──> Structured JSON Response Payload
```

---

## 2. API Function Signature

```python
from tflite_inference import TFLiteEdgeInferenceEngine

engine = TFLiteEdgeInferenceEngine(
    model_path="ml/models/weights/best_improved.pt",
    conf_thresh=0.70,
    iou_thresh=0.45
)

payload, cv_frame = engine.infer_frame(frame_input)
```

### Input Parameters
* **`frame_input`**: Image file path string (e.g. `"dataset/test/images/frame01.jpg"`) OR raw NumPy OpenCV BGR camera array (`np.ndarray` of shape `[H, W, 3]`).

---

## 3. Standard JSON Response Schema

```json
{
  "timestamp": 1776543210.123,
  "frame_status": "SUCCESS",
  "detections_count": 1,
  "detections": [
    {
      "class": "weed",
      "confidence": 0.9125,
      "bbox": [264.42, 124.13, 346.77, 205.15],
      "center": [305.6, 164.64],
      "spray_eligible": true,
      "status": "SPRAY_ELIGIBLE"
    }
  ]
}
```

### Response Field Definitions

| JSON Key | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | `float` | Unix Epoch timestamp (seconds) frame was ingested. |
| `frame_status` | `string` | Processing state (`"SUCCESS"`, `"EMPTY_FRAME"`, `"ERROR"`). |
| `detections_count` | `int` | Total number of evaluated target detections in frame. |
| `detections` | `array` | List of detected target objects. |
| `detections[].class` | `string` | Target classification label (`"weed"`, `"crop"`, `"grass_lawn"`, `"other"`). |
| `detections[].confidence` | `float` | Calibrated model confidence score ($0.0$ to $1.0$). |
| `detections[].bbox` | `array[4]` | Bounding box spatial coordinates `[x_min, y_min, x_max, y_max]` in frame pixels. |
| `detections[].center` | `array[2]` | Target spatial centroid `[c_x, c_y]` in frame pixels for nozzle alignment. |
| `detections[].spray_eligible` | `boolean` | `true` if target passes confidence, size, and crop overlap fail-safe guards. |
| `detections[].status` | `string` | Operational state (`"SPRAY_ELIGIBLE"`, `"CONFIDENCE_INSUFFICIENT"`, `"CROP_SAFETY_GATED"`). |

---

## 4. Pipeline Execution Order

1. **Pre-processing:** Image resized to $640 \times 640$, letterbox padded, converted to RGB, normalized to $[0.0, 1.0]$.
2. **Model Inference:** TFLite / ONNX / PyTorch forward pass execution.
3. **NMS Deduplication:** Non-Maximum Suppression at $\text{IoU} = 0.45$.
4. **Confidence Filtering:** Filters targets below operational threshold ($\ge 0.70$).
5. **Crop Safety Guard:** Checks weed-crop spatial overlap ($\text{IoU} \le 0.25$).
6. **Centroid Calculation:** $c_x = \frac{x_1 + x_2}{2}, c_y = \frac{y_1 + y_2}{2}$.
7. **Debug Visualization:** `engine.draw_debug_visualizations(cv_frame, payload["detections"])`.
