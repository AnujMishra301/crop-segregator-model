# Dataset Statistical & Validation Report

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Document Status:** Comprehensive Dataset Audit & Validation  

---

## 1. Executive Summary & Suitability Assessment

- **Total Usable Images:** 100
- **Total Bounding Box Annotations:** 1704
- **Dataset Suitability Status:** **SUITABLE FOR MODEL TRAINING**
- **Data Leakage Risk:** **Mitigated** via Sequence-Grouped Field Splitting.

---

## 2. Dataset Split Breakdown

| Split Name | Image Count | Image % | Annotation Count | Annotation % |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 60 | 60.0% | 1024 | 60.1% |
| **Validation** | 20 | 20.0% | 341 | 20.0% |
| **Test** | 20 | 20.0% | 339 | 19.9% |
| **Total** | **100** | **100.0%** | **1704** | **100.0%** |

---

## 3. Canonical Class Distribution

| Class ID | Class Name | Total Annotations | Annotation % | Images Containing Class |
| :--- | :--- | :--- | :--- | :--- |
| `0` | **weed** | 289 | 17.0% | 96 |
| `1` | **crop** | 1297 | 76.1% | 100 |
| `2` | **grass_lawn** | 75 | 4.4% | 53 |
| `3` | **other** | 43 | 2.5% | 36 |

---

## 4. Bounding Box & Resolution Distributions

### 4.1 Bounding Box Scale Analysis
- **Small Objects (Area < 1% frame):** 666 (39.1%) — *Emerging weeds & grass blades*
- **Medium Objects (1% - 9% area):** 1038 (60.9%) — *Mature weeds & small crop plants*
- **Large Objects (Area > 9% frame):** 0 (0.0%) — *Established crop canopy*

### 4.2 Resolution Distribution
- `640x640`: 60 images

---

## 5. Identified Dataset Weaknesses & Mitigations

1. **Class Imbalance:** Higher proportion of `crop` vs `weed` instances.  
   *Mitigation:* Applied focal loss in YOLO training and Mosaic/MixUp data augmentations during epoch training.
2. **Small Object Size:** Significant portion of weed bounding boxes are small ($<32 	imes 32$ pixels).  
   *Mitigation:* Input image size fixed at $640 	imes 640$ with multi-scale feature pyramid (P3/P4/P5).
3. **Lighting & Shadow Jitter:** Natural sunlight causes high dynamic contrast.  
   *Mitigation:* Applied HSV jitter (Hue, Saturation, Brightness adjustment) during preprocessing.
