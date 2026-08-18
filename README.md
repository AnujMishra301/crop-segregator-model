# AI-Powered Precision Weed Detection & Targeted Spraying

An autonomous agricultural quadcopter system that uses computer vision to detect weeds and spray only the identified targets instead of blanket-spraying the entire field.

## Overview

Conventional agricultural spraying treats the entire field uniformly, leading to unnecessary chemical consumption on healthy crops and soil. This project aims to solve that problem through an autonomous, low-cost precision-spraying system.

The quadcopter uses a downward-facing camera to capture field images. A custom-trained computer vision model detects weeds and determines their location in the camera frame. A spray is considered eligible only when the weed detection confidence reaches the configured **70% threshold**.

The system is designed around:

* AI-powered weed detection
* Real-time computer vision
* Autonomous flight
* Targeted spraying
* RTK-based positioning
* Radar-based ground clearance
* ArduPilot and MAVLink
* Edge AI deployment

## Problem Statement

**Smart India Hackathon 2026 — Agriculture**

> An autonomous quadcopter that identifies weeds using computer vision and sprays only the detected target instead of blanket-spraying the entire field.

Traditional spraying can result in:

* Chemical wastage on healthy crops
* Excessive chemical application
* Soil and environmental damage
* High operational costs
* Reduced precision in weed control

Our objective is to develop a scalable and cost-effective precision-agriculture system that identifies individual weeds and targets them selectively.

## Proposed Solution

```text
Downward-Facing Camera
          ↓
    Image Acquisition
          ↓
     Preprocessing
          ↓
    Weed Detection AI
          ↓
 Bounding Box + Confidence
          ↓
    Confidence Check
          ↓
    Target Localization
          ↓
     Safety Validation
          ↓
    Spray Decision
          ↓
   Targeted Spray System
```

The AI model performs object detection, while a separate deterministic decision layer handles spray eligibility and safety checks.

## AI Model

The current system uses a YOLOv8n-based object-detection model trained for four classes:

| Class ID | Class      |
| -------: | ---------- |
|        0 | Weed       |
|        1 | Crop       |
|        2 | Grass/Lawn |
|        3 | Other      |

The model produces:

* Object class
* Confidence score
* Bounding box
* Target center coordinates

### Spray Decision

A weed is considered spray-eligible only when:

```text
class = weed
AND
confidence >= 0.70
```

Examples:

```text
WEED + 0.65 → NO SPRAY
WEED + 0.70 → SPRAY ELIGIBLE
WEED + 0.91 → SPRAY ELIGIBLE
CROP + 0.95 → NO SPRAY
```

The 70% threshold is enforced as a safety floor in the decision layer.

## Dataset

The initial baseline dataset contains:

* **221 images**
* **101 original real-field images**
* **120 augmented images**
* **3,752 bounding-box annotations**

The project is currently developing **Dataset V2**, focused on real deployment conditions.

Dataset V2 prioritizes:

* Weed growing among crops
* Small weeds
* Multiple weeds
* Weed + grass
* Crop-only negative examples
* Bare soil
* Dense vegetation
* Sparse vegetation
* Different lighting conditions
* Shadows
* Different crop growth stages
* Different weed growth stages
* Motion blur
* Downward-facing camera imagery

### Dataset V2 Targets

| Category           |      Target |
| ------------------ | ----------: |
| Weed + Crop        | ≥350 images |
| Small Weeds        | ≥200 images |
| Crop Only          | ≥150 images |
| No Weed / Negative | ≥200 images |
| Multiple Weeds     | ≥250 images |
| Weed + Grass       | ≥100 images |

The overall target is at least **1,000 real field images**, with 1,500–2,000 preferred where sufficient diversity can be collected.

## Dataset Workflow

```text
Field Video / Images
        ↓
Frame Extraction
        ↓
Frame Selection
        ↓
YOLO Annotation
        ↓
Annotation QA
        ↓
Dataset Statistics
        ↓
Sequence-Aware Split
        ↓
Model Training
        ↓
Evaluation
```

The dataset is split into training, validation, and test sets while preventing frames from the same capture sequence from appearing across different splits.

## Technology Stack

### AI / Computer Vision

* Python
* YOLOv8
* PyTorch
* OpenCV
* NumPy
* Pillow

### Edge Deployment

* TensorFlow Lite
* Raspberry Pi / onboard edge computer

### Autonomous System

* ArduPilot
* MAVLink
* RTK GPS
* Radar / altitude sensing
* Mission Planner

### Hardware

The proposed system includes:

* Autonomous quadcopter
* Downward-facing camera
* Raspberry Pi or equivalent edge computer
* RTK positioning
* Radar/altitude sensor
* Spray pump
* Targeted misting nozzle

## Repository Structure

```text
├── dataset/
├── dataset_v2/
│   ├── raw/
│   ├── extracted_frames/
│   ├── images/
│   ├── labels/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── qa/
│
├── ml/
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── visualization/
│
├── src/
├── tools/
├── tests/
│
├── run_demo.py
├── requirements.txt
├── DATASET_V2_PLAN.md
├── DATA_COLLECTION_GUIDE.md
├── ANNOTATION_WORKFLOW_V2.md
└── README.md
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset V2 Commands

Extract frames from field videos:

```bash
python tools/extract_frames.py --raw_dir dataset_v2/raw --output_dir dataset_v2/extracted_frames --interval 1.0 --diff_thresh 12.0
```

Validate annotations:

```bash
python tools/validate_dataset_v2.py --img_dir dataset_v2/images --lbl_dir dataset_v2/labels
```

Generate dataset statistics:

```bash
python tools/dataset_collection_stats.py --img_dir dataset_v2/images --lbl_dir dataset_v2/labels
```

Generate annotation visualizations:

```bash
python tools/visualize_dataset_v2.py --img_dir dataset_v2/images --lbl_dir dataset_v2/labels --output_dir dataset_v2/qa --samples 10
```

Generate the dataset readiness dashboard:

```bash
python tools/dataset_v2_dashboard.py --img_dir dataset_v2/images --lbl_dir dataset_v2/labels
```

## Running the AI Demo

Test the model on an image:

```bash
python run_demo.py --image path/to/image.jpg
```

Test on a video:

```bash
python run_demo.py --video path/to/video.mp4
```

Test using a camera:

```bash
python run_demo.py --camera 0
```

The demo displays:

* Detected class
* Confidence
* Bounding box
* Target center
* Spray eligibility
* Reason for rejection when applicable

Physical spray hardware is not activated during AI-only testing.

## Safety

The neural network must **never directly activate the physical spray system**.

The intended architecture separates detection from actuation:

```text
AI Detection
     ↓
Confidence Check
     ↓
Target Validation
     ↓
Safety Checks
     ↓
Spray Request
     ↓
Physical Actuation
```

Low-confidence, ambiguous, invalid, or crop-conflicting detections should result in:

```text
NO SPRAY
```

Initial hardware testing should use water or another safe test liquid rather than agricultural chemicals.

## Current Status

**Current Phase: Dataset V2 Collection**

The baseline AI pipeline and decision layer have been implemented and tested.

The current baseline model is **not yet deployment-ready**. Dataset V2 is being developed to improve real-world weed detection, particularly weed-vs-crop discrimination and detection of small or partially occluded weeds.

Model V2 training will begin only after sufficient real-world data has been collected, annotated, validated, and reviewed.

## Roadmap

* [x] Initial object-detection pipeline
* [x] Four-class model architecture
* [x] Spray confidence decision layer
* [x] 70% confidence safety threshold
* [x] Dataset V2 framework
* [x] Annotation QA tools
* [x] Dataset statistics and readiness dashboard
* [ ] Collect Dataset V2
* [ ] Annotate and validate Dataset V2
* [ ] Train Model V2
* [ ] Perform model error analysis
* [ ] Validate on real-world field imagery
* [ ] Optimize edge inference
* [ ] Camera calibration
* [ ] Target localization
* [ ] Spray calibration
* [ ] Stationary closed-loop testing
* [ ] Motion and latency compensation
* [ ] ArduPilot/MAVLink integration
* [ ] Hardware-in-the-loop testing
* [ ] Controlled field testing
* [ ] Final model release

## Expected Impact

The proposed system aims to provide:

* Reduced chemical wastage
* Lower operating costs
* Reduced unnecessary chemical application
* Reduced dependence on heavy tractor-based spraying
* Precise weed control
* Autonomous and repeatable field missions
* A low-cost alternative to expensive commercial precision-agriculture systems

## Team

**The Unscripted Six**

Developed for **Smart India Hackathon 2026 — Agriculture**.

## Disclaimer

This project is an experimental engineering prototype. Model accuracy, targeting precision, autonomous flight behavior, and spraying effectiveness must be validated through controlled testing before any real agricultural chemical application.

## License

Add the appropriate project license before public distribution.
