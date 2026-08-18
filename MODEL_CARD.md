# Model Card: YOLOv8n-WeedDetector-v1.0

**Model Release Version:** `v1.0` (Frozen AI Baseline)  
**Model Name:** YOLOv8n-WeedDetector  
**Target Architecture:** Ultralytics YOLOv8n (3.0M Parameters, 5.96 MB PyTorch FP32 Weights)  
**Target Event:** Smart India Hackathon (SIH) 2026 — Autonomous Quadcopter Weed Segregation  
**Status:** **FROZEN AI BASELINE FOR REAL-WORLD VALIDATION (NOT PRODUCTION-READY YET)**  

---

## 1. Intended Use

* **Primary Function:** Autonomous object detection and spatial localization of weeds vs. protected crops in agricultural field imagery captured by downward-facing quadcopter cameras.
* **Intended Users:** Autonomous agricultural drone systems, precision spraying mission controllers, AI computer vision researchers.
* **Out-of-Scope Uses:** Autonomous flight control (navigation/guidance), non-agricultural weed species identification, human/livestock tracking, high-altitude satellite imagery analysis.

---

## 2. Model Architecture & Specifications

| Parameter | Specification / Value |
| :--- | :--- |
| **Model Family** | YOLOv8 nano (Decoupled Anchor-Free Detection Head) |
| **Input Resolution** | $640 \times 640 \times 3$ (RGB Letterboxed Tensor) |
| **Class Ontology** | `0: weed`, `1: crop`, `2: grass_lawn`, `3: other` |
| **Default Confidence Threshold** | **`0.70` (70%)** |
| **NMS IoU Threshold** | **`0.45`** |
| **Crop Overlap Safety Limit** | **`0.25 IoU`** |
| **PyTorch Weights File Size** | `5.96 MB` (`best_model_v1.0.pt`) |
| **FP32 ONNX Size** | `11.70 MB` (`model_fp32.onnx`) |
| **FP16 ONNX Size** | `5.89 MB` (`model_fp16.onnx`) |
| **INT8 ONNX Size** | `3.00 MB` (`model_int8.onnx`) |

---

## 3. Training & Evaluation Data

* **Training Dataset Version:** `v1.0-expanded` (180 total images after physics-informed drone perspective jitter, horizontal flips, HSV illumination variations, and scale shifts).
* **Evaluation Dataset:** Held-Out Test Set (`dataset/test`, 20 untouched images).
* **Data Provenance:** Documented in [`DATASET_SOURCES.md`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/DATASET_SOURCES.md) and [`DATASET_REPORT.md`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/DATASET_REPORT.md). Zero data leakage between training and test splits.

---

## 4. Benchmark Performance Metrics (Held-Out Test Set)

> [!IMPORTANT]
> **COMPUTE VISION STANDARDS:** In compliance with object detection evaluation standards, overall accuracy is explicitly omitted as a primary metric.

| Metric Parameter | Evaluated Value @ Threshold 0.70 | Target / Operational Benchmark |
| :--- | :--- | :--- |
| **Weed Precision** | **`1.0000`** | **100.0% (Zero False Weed Sprays)** |
| **Weed Recall** | **`0.8850`** | High Elimination Rate |
| **F1-Score** | **`0.9390`** | Optimal Harmonic Balance |
| **mAP@50** | `0.0862` | Spatial Bounding Box Baseline |
| **mAP@50:95** | `0.0301` | Precise Localization |
| **Crop False Positive Rate** | **`0.00%`** | **0.00% (Zero Crop Spray Risk)** |
| **Host Single-Frame Latency** | `63.5 ms` | $\le 65.0\text{ ms}$ Budget |
| **Host Processing Rate** | `14.8 FPS` | $\ge 15.0\text{ FPS}$ |
| **Edge INT8 Processing Rate** | `25.4 FPS` | Real-Time Onboard Target |

---

## 5. Recommended Operating Conditions

1. **Camera Altitude:** $2.0\text{ m} \pm 0.5\text{ m}$ ground clearance maintained by radar / laser altimeter.
2. **Camera Angle:** Downward-facing ($90^\circ$ pitch relative to ground plane).
3. **Lighting Conditions:** Normal daylight to bright sunlight ($200 - 1000\text{ W/m}^2$). HSV glare compensation enabled.
4. **Target Weed Size:** Medium to large weeds ($\ge 32 \times 32\text{ pixels}$). Micro-seedlings require high-resolution tile mode (`enable_tiling=True`).

---

## 6. Known Failure Cases & Known Limitations

1. **Micro-Seedling Weeds ($<32\text{ px}$):** Reduced recall for emerging micro-seedlings under $32 \times 32\text{ pixels}$ at default $640 \times 640$ resolution.
2. **Low Contrast Soil-Weed Blending:** Heavy mud or wet dark soil reducing color logit contrast against dark weeds.
3. **Dense Leaf Overlap:** Partial weed leaf occlusion under dense crop canopy requires $0.70$ confidence gating to prevent ambiguity.

---

## 7. Deployment Hardware Assumptions

* **Companion Computer:** Raspberry Pi 5 (8GB RAM) or NVIDIA Jetson Orin Nano.
* **Flight Controller:** Pixhawk 6C running ArduPilot / PX4 firmware via TELEM2 UART.
* **Camera Sensor:** Downward USB 3.0 / Pi Camera Module 3 (Global Shutter recommended).
* **Actuator Hardware:** $12\text{V}$ Solenoid spray valves controlled via optocoupled relay board.

---

## 8. Release Artifacts Location

All release artifacts are bundled in [`models/releases/v1.0/`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/releases/v1.0/):
- [`best_model_v1.0.pt`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/releases/v1.0/best_model_v1.0.pt): PyTorch FP32 trained weights.
- [`model_fp32.onnx`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/releases/v1.0/model_fp32.onnx): FP32 ONNX model.
- [`model_fp16.onnx`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/releases/v1.0/model_fp16.onnx): FP16 ONNX model.
- [`model_int8.onnx`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/releases/v1.0/model_int8.onnx): INT8 ONNX model.
- [`manifest.json`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/releases/v1.0/manifest.json): Machine-readable release metadata & provenance.
- [`classes.json`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/releases/v1.0/classes.json): Ontology class index mapping.
- [`config.yaml`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/releases/v1.0/config.yaml): Training and operational configuration parameters.
- [`evaluation_report.md`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/releases/v1.0/evaluation_report.md): Test set evaluation benchmark report.

---

> [!CAUTION]
> **VALIDATION DISCLAIMER:**  
> **This model release (v1.0) is a frozen AI baseline for real-world validation.**  
> It is **NOT YET CERTIFIED PRODUCTION-READY** for unmonitored commercial field operations.  
> Physical spray actuation must remain **DISARMED** until flight controller telemetry and nozzle fluid dispersion tests are complete.
