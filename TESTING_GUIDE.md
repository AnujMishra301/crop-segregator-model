# AI Quadcopter Weed Detection Testing & Verification Guide

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Test Runner Script:** [`run_demo.py`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/run_demo.py)  

---

## 1. Environment Setup & Dependency Installation

### Step 1: Create Virtual Environment
Open a terminal in the project root directory:

```bash
# Windows (PowerShell / CMD)
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Core Dependencies
Install the required computer vision and machine learning libraries:

```bash
pip install -r requirements.txt
```

---

## 2. Model Checkpoint Locations

The test runner automatically checks and loads trained model weights from the following paths:
1. Primary Weights: [`ml/models/weights/best_baseline.pt`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/ml/models/weights/best_baseline.pt) (5.96 MB PyTorch FP32)
2. Release Weights: [`models/releases/v1.0/best_model_v1.0.pt`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/releases/v1.0/best_model_v1.0.pt)
3. ONNX Edge Models: [`models/releases/v1.0/model_fp32.onnx`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/models/releases/v1.0/model_fp32.onnx)

---

## 3. Running AI Inference Tests

### Option A: Single Image Inference
Run inference on a local agricultural image file:

```bash
python run_demo.py --image dataset/test/images/field01_frame001.jpg
```

**Custom Output File & Custom Threshold:**
```bash
python run_demo.py --image dataset/test/images/field01_frame002.jpg --conf 0.70 --save output_custom.jpg
```

### Option B: Video File Inference
Process an MP4/AVI flight test video frame by frame:

```bash
python run_demo.py --video path/to/field_video.mp4
```

### Option C: Live Webcam / USB Camera Stream
Run real-time inference on a connected downward camera feed:

```bash
python run_demo.py --camera 0
```
*(Press **`q`** to exit the camera window).*

---

## 4. Interpreting Output Format & Decision Telemetry

For every frame, the test runner evaluates class confidence against the configured operational threshold (**`0.70` / 70%**):

### Example Output 1: Valid Weed Target
```text
[1] WEED: 0.87
    Target: (423, 281)
    Decision: SPRAY ELIGIBLE
```
* **Meaning:** Weed detected with $87\%$ confidence ($\ge 70\%$). Target centroid $(423, 281)$ localized and marked eligible for simulated spraying.

### Example Output 2: Protected Crop
```text
[2] CROP: 0.94
    Decision: NO SPRAY
    Reason: PROTECTED CROP
```
* **Meaning:** Crop detected with $94\%$ confidence. Spray trigger disarmed to protect crop safety.

### Example Output 3: Low-Confidence Weed Target
```text
[3] WEED: 0.54
    Decision: NO SPRAY
    Reason: LOW CONFIDENCE
```
* **Meaning:** Weed detected with $54\%$ confidence. Below operational threshold ($70\%$), so target is rejected.

---

## 5. Troubleshooting Common Issues

1. **`[Error] Image file not found`**  
   - *Fix:* Ensure the file path passed to `--image` exists and use relative slashes (e.g. `dataset/test/images/field01_frame001.jpg`).

2. **`[Error] Could not open camera index 0`**  
   - *Fix:* Verify USB camera connection or test index `1` (`python run_demo.py --camera 1`). On Windows, ensure privacy settings allow camera access.

3. **`OpenCV cv2.imshow GUI Error (Headless Server)`**  
   - *Fix:* On headless Linux servers or CI/CD pipelines without X11 display, use `--image` or `--video` with `--save` output instead of `--camera`.

---

> [!NOTE]
> **SAFETY DISARM GUARANTEE:**  
> `run_demo.py` is purely a software inference test. Physical spray pumps, MAVLink hardware commands, and GPIO actuation outputs are **DISARMED**.
