# Computer Vision & AI Architecture Document

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Document Status:** Technical Architecture Specification & Design Proposal  

---

## 1. System Context & Environment Audit

### 1.1 Repository & Codebase Status
- **Repository Path:** `crop-seggregation-model`
- **Current State:** Empty workspace initialization. No legacy ML code, data loaders, or model weights currently exist.
- **Programming Language:** Python 3.x

### 1.2 Development & Edge Host Environments
- **Local Host Hardware:** Windows System equipped with NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), Python 3.13.5, PyTorch 2.9.1 (CPU mode on host; CUDA PyTorch to be provisioned for local model training).
- **Target Edge Platform:**
  - **Single Board Computer (SBC):** Raspberry Pi 4 Model B (4GB/8GB) or Raspberry Pi 5 (4GB/8GB) running 64-bit Raspberry Pi OS (Debian Bookworm / Python 3.10 or 3.11).
  - **Hardware Accelerators (Optional/Supported):** Coral USB Accelerator (Edge TPU) OR Hailo-8L AI HAT for RPi 5 OR ARM NEON optimized TFLite CPU execution.
  - **Camera Unit:** Downward-facing RGB camera (Global shutter or high-frame-rate USB/CSI camera, e.g., Raspberry Pi Camera Module 3 / Arducam Global Shutter) mounted on vibration dampeners.
  - **Flight Controller (FC):** ArduPilot / Pixhawk flight controller running ArduCopter firmware communicating via MAVLink over UART (`/dev/ttyAMA0` or USB Telemetry).
  - **Sensors:** RTK GNSS (centimeter-level horizontal position hold), Radar / Altimeter / TFmini LiDAR for precise altitude hold ($\approx 2.0\text{ m}$ Above Ground Level).
  - **Actuators:** Relays / Solenoid valves driving a multi-nozzle spray bar array aligned with camera Field of View (FOV).

---

## 2. Computer Vision Paradigm: Object Detection vs. Image Classification

For an autonomous targeted crop spraying system, **Object Detection is strictly mandatory** over Image Classification due to fundamental operational requirements:

1. **Spatial Localization ($x, y$ coordinates):**  
   Image classification outputs only a single label per image (e.g., "weed present"). It provides no spatial coordinates. An autonomous quadcopter moving over a field requires exact spatial bounding box centroids $(x_c, y_c)$ relative to the frame to calculate ground offsets and actuate the corresponding spray nozzle at the precise time.

2. **Multi-Object Field Scene Handling:**  
   A single camera frame captured from 2 meters altitude typically contains multiple plants simultaneously (crops, weeds, soil, grass). Classification would flag an entire multi-plant frame as "weed", leading to indiscriminate blanket-spraying over adjacent crops. Object detection isolates individual weed instances while preserving crop locations.

3. **Nozzle Mapping & Temporal Timing:**  
   The bounding box centroid $(x_c, y_c)$ combined with drone flight velocity $(v_x, v_y)$ and altitude $h$ enables calculating the exact delay time $\Delta t$ required for the drone to move from frame capture position to nozzle spraying position.

4. **False Positive & Crop Damage Prevention:**  
   Multi-class detection explicitly locates `crop` bounding boxes alongside `weed` bounding boxes. Spatial overlap analysis (Intersection over Union) ensures that if a weed is immediately adjacent to a crop, spraying can be dynamically modulated or gated to protect crop health.

---

## 3. Model Architecture Comparison

A comparative evaluation of edge-compatible object detection architectures suitable for low-power drone hardware is detailed below:

| Metric / Criteria | EfficientDet-Lite (Lite0 / Lite1) | SSD MobileNet (v2 / v3) | Lightweight YOLO (YOLOv8n / YOLOv11n / YOLOv5n-TFLite) |
| :--- | :--- | :--- | :--- |
| **Backbone & Neck** | EfficientNet-Lite + BiFPN | MobileNetV2/V3 + SSDLite Head | CSPDarknet / HGNet + C2f/C3k + Decoupled Head |
| **Small Object Sensitivity (Weeds at 2m AGL)** | High (BiFPN multi-scale feature fusion excels at small targets) | Moderate to Low (Standard SSD pyramid loses fine spatial detail) | **Very High** (Multi-scale feature pyramid P3/P4/P5 with specialized anchor-free heads) |
| **Inference Latency on RPi 4/5 (CPU TFLite)** | ~40–60 ms (16–25 FPS) | ~30–45 ms (22–33 FPS) | **~20–35 ms (28–50 FPS on RPi 5 CPU; <10 ms on Hailo/EdgeTPU)** |
| **TensorFlow / TFLite Export Path** | Native (TFLite Model Maker native) | Native (TF Object Detection API) | **Fully Exportable** (PyTorch $\to$ ONNX $\to$ TFLite FP16/INT8 via Ultralytics / `tflite-runtime`) |
| **Quantized Model Size (INT8 TFLite)** | ~4.4 MB – ~5.8 MB | ~5.5 MB – ~12 MB | **~3.0 MB – ~6.0 MB (YOLOv8n INT8 TFLite ~6 MB)** |
| **Training Pipeline Ecosystem** | Moderate (Legacy TF1/TF2 model garden dependencies) | Complex (TF OD API XML configs, fragile dependency graph) | **Extremely Streamlined** (Modern PyTorch ecosystem, rich augmentations, native TFLite export) |
| **Baseline COCO Performance** | ~30.4% mAP | ~22.0% – 28.0% mAP | **~37.3% mAP (YOLOv8n) / ~39.0% mAP (YOLOv11n)** |

---

## 4. Architecture Recommendation & Selection

**RECOMMENDED ARCHITECTURE: Lightweight YOLO Nano (YOLOv8n / YOLOv11n exported to TFLite INT8)**

### Technical Justification:
1. **Superior Feature Extraction for Small Weeds:** Downward drone cameras at 2m height capture young/emerging weeds occupying small pixel areas (e.g., $20 \times 20$ pixels in a $640 \times 640$ frame). YOLO's Feature Pyramid Network (FPN) with Path Aggregation (PAN) preserves fine spatial features from early backbone layers, delivering higher recall on small targets than SSD MobileNet.
2. **Real-time Latency Budget Compliance:** Flight speeds of 1.5–2.0 m/s require detection processing latency $\le 35\text{ ms}$ ($\ge 30\text{ FPS}$). YOLO Nano INT8 TFLite achieves 30–45 FPS on Raspberry Pi 5 CPU (and $>100\text{ FPS}$ with NPU/Hailo accelerator), minimizing motion blur artifacts and positioning lag.
3. **Robust Export & Deployment Pipeline:** Model training is conducted in PyTorch using local NVIDIA GPU capabilities, leveraging robust augmentations (Mosaic, MixUp, HSV jitter, random crop/scale). The trained checkpoint is exported directly to ONNX and quantized to TensorFlow Lite (INT8/FP16) using full integer calibration.
4. **Reliability on Edge Runtimes:** TFLite INT8 execution via `tflite_runtime.interpreter` provides low CPU load and stable memory consumption during continuous quadcopter flight operations.

---

## 5. Input / Output Format Specification

### 5.1 Model Input Specification
- **Spatial Resolution:** $640 \times 640 \times 3$ (RGB color format)
- **Tensor Format:** `NHWC` format for TFLite `[1, 640, 640, 3]`
- **Data Type:** `uint8` $[0, 255]$ for full INT8 quantized TFLite model, or `float32` normalized $[0.0, 1.0]$ for FP32/FP16 models.
- **Preprocessing Pipeline:**
  1. Capture raw video frame from camera stream (e.g., $1920 \times 1080$ or $1280 \times 720$).
  2. Perform letterbox resize to $640 \times 640$ with stride 32 (maintaining aspect ratio with gray padding).
  3. Convert color space from BGR to RGB.
  4. Perform tensor reshape to `[1, 640, 640, 3]`.

### 5.2 Model Output Specification (Post-NMS TFLite Interpreter)
- **Bounding Boxes Tensor (`boxes`):** Shape `[1, N, 4]` normalized coordinates $[y_{min}, x_{min}, y_{max}, x_{max}]$ or $[x_c, y_c, w, h]$ relative to $640 \times 640$ input frame.
- **Class Labels Tensor (`classes`):** Shape `[1, N]` integer class indices corresponding to:
  - `0`: `weed`
  - `1`: `crop`
  - `2`: `grass_lawn`
  - `3`: `other`
- **Confidence Scores Tensor (`scores`):** Shape `[1, N]` float confidence values in range $[0.0, 1.0]$.
- **Valid Detections Count (`num_detections`):** Shape `[1]` scalar count of valid detections $N$ after Non-Maximum Suppression (NMS).

---

## 6. Confidence Threshold ($\ge 70\%$) Logic & Safety Protocol

The operational protocol enforces a strict confidence threshold of **$\ge 70\%$ ($0.70$)** for weed detection before spray activation.

```
       [Raw Bounding Box Prediction]
                     │
         Is Class == 0 ('weed')?
          ├── No  ──> [IGNORE / LOG]
          └── Yes
                     │
          Is Confidence >= 0.70?
          ├── No  ──> [DO NOT SPRAY] (Confidence insufficient)
          └── Yes
                     │
   Compute IoU with 'crop' Bounding Boxes
          ├── IoU > 0.30 ──> [SPRAY GATED / WARN] (Prevents crop damage)
          └── IoU <= 0.30
                     │
         [TARGET ACCEPTED FOR SPRAYING]
  Calculate (x_c, y_c) Centroid & Velocity Delay Δt
                     │
         [TRIGGER SPRAY SOLENOID]
```

### Protocol Steps:
1. **Confidence Gating:** Filter all predicted bounding boxes where `class_id == 0` (`weed`) and `confidence < 0.70`. Unconfirmed predictions are logged for telemetry but do not activate actuators.
2. **Crop Intersect Guard (Safety Gating):** Compute Intersection over Union (IoU) between accepted weed boxes and any detected `crop` boxes (`class_id == 1`). If $\text{IoU}(\text{weed\_box}, \text{crop\_box}) > 0.30$, spraying is suppressed or pulse-modulated to avoid crop damage.
3. **Centroid Extraction:** For accepted weed boxes, compute the bounding box centroid in frame coordinates:
   $$x_c = \frac{x_{min} + x_{max}}{2}, \quad y_c = \frac{y_{min} + y_{max}}{2}$$

---

## 7. Drone Hardware Integration & Controller Interface

### 7.1 Hardware & Subsystem Communication Architecture
- **Flight Altitude Lock:** Radar / Altimeter maintains fixed altitude $h \approx 2.0\text{ m} \pm 0.1\text{ m}$ AGL. This maintains constant Ground Sampling Distance (GSD):
  $$\text{GSD} = \frac{2 \cdot h \cdot \tan(\text{FOV}/2)}{\text{Frame Height (pixels)}}$$
- **Velocity Telemetry:** MAVLink protocol reads quadcopter speed vector $(v_x, v_y)$ from ArduPilot over UART serial interface (`/dev/ttyAMA0`).
- **Physical Nozzle Offset:** Camera is mounted at longitudinal distance $L_{offset}$ ahead of the spray nozzle bar.

### 7.2 Timing & Delay Calculation
For a detected weed centroid at vertical frame coordinate $y_c$ (where $y=0$ is top of frame, $y=1.0$ is bottom):
$$\text{Physical Distance to Nozzle } D_{target} = L_{offset} + (1.0 - y_c) \cdot \text{FOV}_{length\_meters}$$
$$\text{Spraying Delay } \Delta t = \frac{D_{target}}{\sqrt{v_x^2 + v_y^2}}$$

When time elapsed equals $\Delta t$, the Raspberry Pi sends a High digital pulse ($\tau_{pulse} \approx 50-100\text{ ms}$) to the GPIO pin controlling the corresponding solenoid spray valve.

---

## 8. Training Strategy & Pipeline

### 8.1 Dataset Organization (YOLO Format)
```
dataset/
├── dataset.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 8.2 Dataset Configuration (`dataset.yaml`)
```yaml
path: ./dataset
train: train/images
val: val/images
test: test/images

names:
  0: weed
  1: crop
  2: grass_lawn
  3: other
```

### 8.3 Augmentation Strategy (Tailored for Downward Drone Footage)
- **Rotational Invariance:** Random 90-degree rotations, horizontal/vertical flips.
- **Lighting Variability:** HSV color channel jitter (Hue $\pm 0.015$, Saturation $\pm 0.7$, Value $\pm 0.4$) to simulate field shadows and changing sunlight.
- **Scale & Density:** Mosaic (4-image blending) and MixUp augmentations to enhance small object detection performance.
- **Drone Motion Simulation:** Random perspective transformation and minor motion blur.

### 8.4 Hyperparameter Configuration
- **Base Architecture:** YOLOv8n / YOLOv11n Nano
- **Optimizer:** AdamW (lr0=0.01, momentum=0.937, weight_decay=0.0005)
- **Learning Rate Schedule:** Cosine Annealing decay
- **Epochs:** 100 – 150 with early stopping (patience = 15)
- **Batch Size:** 16 or 32 (optimized for GPU memory)
- **Image Resolution:** $640 \times 640$

### 8.5 Quantization & TFLite Export Flow
1. Train PyTorch model to convergence $\to$ `best.pt`.
2. Export checkpoint to ONNX format $\to$ `best.onnx`.
3. Convert to TFLite INT8 format using TensorFlow Lite Converter with full integer quantization calibrated using a representative sample of 100 validation field images $\to$ `model_int8.tflite`.

---

## 9. Evaluation Metrics Definition

Model evaluation will be strictly conducted on an independent test dataset using standard object detection metrics:

1. **Intersection over Union (IoU):**
   $$\text{IoU} = \frac{|\mathcal{B}_{pred} \cap \mathcal{B}_{gt}|}{|\mathcal{B}_{pred} \cup \mathcal{B}_{gt}|}$$
   Evaluates spatial overlap between predicted bounding box $\mathcal{B}_{pred}$ and ground truth box $\mathcal{B}_{gt}$. An IoU threshold of $0.50$ defines a True Positive match.

2. **Precision ($P$):**
   $$P = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   Measures proportion of correct weed detections against all positive predictions (critical for minimizing false sprays on crops/soil).

3. **Recall ($R$):**
   $$R = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   Measures proportion of actual ground truth weeds correctly detected by the model.

4. **F1-Score:**
   $$F1 = 2 \cdot \frac{P \cdot R}{P + R}$$
   Harmonic mean balancing precision and recall.

5. **mAP@50 (Mean Average Precision at IoU 0.50):**
   Average Precision evaluated across all 4 classes at $\text{IoU threshold} = 0.50$:
   $$\text{mAP}@50 = \frac{1}{4} \sum_{c=0}^{3} \text{AP}_{c, 0.50}$$

6. **mAP@50:95 (Mean Average Precision across IoU 0.50 to 0.95):**
   Average Precision evaluated across 10 IoU thresholds ($0.50, 0.55, \dots, 0.95$ in steps of 0.05). Indicates exact bounding box localization precision.

---

## 10. Blueprint of Repository Structure & Files to Create

The implementation phase will establish the following modular directory structure:

```
crop-seggregation-model/
├── AI_ARCHITECTURE.md              # Computer vision architecture specification (this file)
├── requirements.txt                # Dependencies for training and edge inference
├── config/
│   └── dataset.yaml                # Class mapping and dataset path configurations
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Image acquisition, letterboxing, normalization
│   ├── detector.py                 # TFLite / ONNX inference engine, NMS & confidence gating
│   ├── spray_controller.py         # MAVLink telemetry sync, coordinate conversion & GPIO pulse
│   ├── train.py                    # PyTorch training wrapper
│   ├── export_tflite.py            # Model quantization & TFLite export script
│   └── evaluate.py                 # Computation of Precision, Recall, F1, IoU, mAP metrics
└── main_edge.py                    # Main real-time operational loop running on Raspberry Pi
```
