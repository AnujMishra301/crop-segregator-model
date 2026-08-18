# Dataset V2 Engineering Plan & Architecture

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Document Purpose:** Dataset V2 Specification, Target Metrics, Class Balance, and Sequence-Aware Leakage Prevention  

---

## 1. Motivation & Real-World Field Deployment Requirements

The baseline model (`best_baseline.pt` trained on initial 180 expanded images) achieved $100\%$ precision at the operational threshold $0.70$, but suffered from low logit confidence ($0.21 - 0.24$) on complex field imagery, resulting in near-zero recall. 

**Dataset V2** is engineered to eliminate synthetic/small-data dependence by collecting **$\ge 1000$ real agricultural field images** under physical quadcopter flight conditions:

* **Camera Clearance:** Downward-facing $90^\circ$ pitch at $2.0\text{ m} \pm 0.5\text{ m}$ ground clearance.
* **Agricultural Contexts:** Weeds among active crop rows, weeds embedded in lawn/grass, and bare soil.
* **Environmental Variance:** Bright sunlight glare, shadow occlusions, dark/wet soil, dry light soil, motion blur, and micro-seedlings ($<32\text{ px}$).

---

## 2. Quantitative Targets for Dataset V2

| Metric / Parameter | Target Goal | Engineering Rationale |
| :--- | :---: | :--- |
| **Total Real Field Images** | **$\ge 1,000$ images** | Eliminates small-sample logit degradation. |
| **Synthetic Imagery Ratio** | **$0.0\%$** | Real field footage strictly required. |
| **Train Split Ratio** | **$70\%$ ($\approx 700+$ images)** | Robust feature extraction learning. |
| **Validation Split Ratio** | **$20\%$ ($\approx 200+$ images)** | Hyperparameter tuning and early stopping. |
| **Test Split Ratio** | **$10\%$ ($\approx 100+$ images)** | Touchless held-out sequence evaluation. |
| **Class Ontology Mapping** | `0: weed`, `1: crop`, `2: grass_lawn`, `3: other` | Preserves canonical system API contract. |

---

## 3. Class Balance & Difficult Negative Engineering

To prevent the dataset from becoming mostly "weed + soil" images, Dataset V2 enforces strict negative sampling and crop-resemblance balance:

```
[Target Class Distribution]
├── Weed Candidates (Class 0): ~40% of annotations (Targeted Weed Targets)
├── Crop Candidates (Class 1): ~45% of annotations (Protected Crop Rows)
├── Grass / Lawn (Class 2): ~10% of annotations (Vegetation Background)
└── Other / Soil (Class 3): ~5% of annotations (Non-Target Agricultural Objects)
```

### Mandatory Negative Image Inclusion:
1. **Difficult Crop Negatives:** Images containing crop rows only (no weeds) to teach the model zero-weed confidence.
2. **Grass / Lawn Backgrounds:** Turf grass without crop or weed targets.
3. **Bare Soil & Shadows:** Field ground without vegetation, including heavy tractor tire tracks and shadow lines.
4. **Ambiguous Weed-Crop Overlaps:** Leaf contact between weed seedlings and crop leaves to train the $0.25\text{ IoU}$ crop fail-safe guard.

---

## 4. Sequence-Level Leakage Prevention Strategy

Standard random image-level splitting causes **temporal data leakage** when consecutive frames from the same drone flight video are split across train and test sets.

### Sequence Grouping Protocol:
* All frames extracted from a single video session or field pass share a **sequence ID** (e.g., `field_session_01_f00010.jpg`).
* [`tools/split_dataset_v2.py`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/tools/split_dataset_v2.py) groups all frames by `sequence_id` **before** performing the $70:20:10$ split.
* Entire flight passes are assigned atomically to either `train`, `val`, or `test`.
