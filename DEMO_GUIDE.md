# AI Quadcopter Weed Segregation Demo Application Guide

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Document Purpose:** Demonstration Guide & Architectural Component Separation Document  
**Application Script:** [`demo_app.py`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/demo_app.py)  

---

## 1. Quick Start Guide: Running the Demo Application

### Step 1: Launch the Interactive Web Dashboard
Run the Streamlit web application from your terminal:

```bash
streamlit run demo_app.py
```

The application will launch automatically in your web browser at `http://localhost:8501`.

### Step 2: Select Input Source
In the left sidebar, choose one of the available input modes:
- **Upload Image:** Select a local field image file (`.jpg`, `.png`).
- **Upload Video:** Stream an MP4/AVI field video recorded during flight testing.
- **Live Camera:** Ingest live downward-facing USB / Pi camera frames.
- **Run Pre-Loaded Test Batch:** Replay a sequence of held-out test field images.

---

## 2. Dashboard Features & Interactive Controls

1. **Operational Confidence Threshold Slider:** Adjust between $50\%$ and $95\%$ (Default: **$70\%$**). Detections below this threshold are tagged `NO SPRAY (Reason: LOW CONFIDENCE)`.
2. **Crop Safety IoU Overlap Slider:** Adjust max allowed weed-crop spatial overlap (Default: **$0.25\text{ IoU}$**). Weed boxes overlapping crop boxes beyond this limit are tagged `NO SPRAY (Reason: CROP SAFETY)`.
3. **Real-Time Telemetry Cards:** Displays FPS, Single-Frame Inference Latency (ms), Total Weeds Detected, Spray Candidates, and Simulated Spray Pulse Counts.
4. **Target Side-Panel:** Displays per-object confidence score, pixel centroid $(c_x, c_y)$, ground offset $[dx_{\text{cam}}, dy_{\text{cam}}]\text{ m}$, nozzle target offset $[dx_{\text{noz}}, dy_{\text{noz}}]\text{ m}$, and decision reason.

---

## 3. Strict Architectural Component Separation

To ensure scientific rigor and clear technical boundaries during SIH 2026 evaluations, the system maintains strict functional separation between software layers:

```
[1. AI Detection Layer] ──> [2. Target Localization] ──> [3. Decision Logic] ──> [4. Physical Actuation]
(Neural Net Inference)     (Pinhole Ground Map)        (Safety Gate)            (Disarmed / Simulated)
```

| Layer Component | Responsibility & Scope | Verification Status |
| :--- | :--- | :--- |
| **1. AI Detection Layer** | YOLOv8 object detection model identifies spatial bounding boxes and class probabilities for `weed`, `crop`, `grass_lawn`, and `other`. | **Evaluated on Held-Out Test Set** |
| **2. Target Localization** | Pinhole camera projector transforms 2D pixel centroids $(c_x, c_y)$ to camera ground offsets $(\Delta x_{\text{cam}}, \Delta y_{\text{cam}})$ and subtracts camera-to-nozzle mechanical offset vector. | **Verified via Synthetic Geometry Tests** |
| **3. Decision Logic** | Multi-condition safety gate boolean evaluator verifies confidence $\ge 0.70$, minimum bounding box area, and zero crop spatial overlap ($\text{IoU} \le 0.25$). | **Validated in Protocol Unit Tests** |
| **4. Physical Actuation** | Hardware relay/solenoid spray pump triggering. | **DISARMED IN SIMULATION MODE** |

---

## 4. Important Evaluation Disclaimer

> [!WARNING]
> **SIMULATION vs. PHYSICAL SPRAYING DISCLAIMER:**  
> Simulated spray triggers and virtual pump pulses demonstrate **software decision logic, target localization mathematics, and dataflow protocol integration**.  
> Simulated spray execution **DOES NOT PROVE physical spraying accuracy, nozzle pressure calibration, or aerodynamic fluid spray dispersion** under actual quadcopter downwash flight conditions.  
> Physical spray accuracy requires field calibration with the physical nozzle hardware and flight controller.
