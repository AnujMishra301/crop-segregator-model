# Annotation Quality Assurance (QA) Report

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Document Status:** Complete Annotation Quality Inspection  

---

## 1. QA Executive Summary

- **Total Annotations Evaluated:** 1704
- **Structurally Invalid Annotations:** 0
- **Suspicious / Flagged Annotations:** 7
- **Review List File:** [`dataset/qa/flagged_review_list.json`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/dataset/qa/flagged_review_list.json)
- **Automatic Deletions:** **0** (All flagged items are retained and queued for manual verification).

---

## 2. Validation & Suspicious Annotation Summary

### 2.1 Structural Validation (Task 1)
| Error Type | Count | Description |
| :--- | :--- | :--- |
| **Malformed Line Structure** | 0 | Line elements $\neq 5$ |
| **NaN / Inf Values** | 0 | Non-numeric or infinity values |
| **Out-of-Bounds Coords** | 0 | Center or dimensions outside $[0.0, 1.0]$ |
| **Invalid Class ID** | 0 | Class ID not in $[0..3]$ |
| **Total Invalid** | **0** | Structural clean pass |

### 2.2 Suspicious Detection Summary (Task 2)
| Flag Category | Count | Primary Impact |
| :--- | :--- | :--- |
| **Extremely Tiny Box** ($<0.04\%$ frame) | 0 | Edge camera feature extraction limit |
| **Massive Box** ($>75\%$ frame) | 0 | Background confusion |
| **Abnormal Aspect Ratio** ($>8:1$ or $<1:8$) | 0 | Line/artifact labeling |
| **Duplicate Bounding Boxes** ($	ext{IoU} > 0.92$) | 0 | Redundant loss weighting |
| **Weed vs Crop Overlap** ($	ext{IoU} > 0.45$) | 7 | Spray trigger risk |
| **Total Flagged for Review** | **7** | Queued in `flagged_review_list.json` |

---

## 3. Geometric Box Dimension Statistics

| Metric | Normalized Width ($w$) | Normalized Height ($h$) | Normalized Area ($w \times h$) | Equivalent Pixel Area ($640 \times 640$) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean (Average)** | 0.1074 | 0.1074 | 0.012215 | 5003.4 pixels |
| **Median** | 0.1083 | 0.1083 | 0.011725 | 4802.5 pixels |

---

## 4. Extreme Bounding Box Samples

### 4.1 Smallest Annotations (Top 3)
- **Sample #1:** `field01_frame007.jpg` (Class `weed`) — Size: $0.0413 \times 0.0413$ (Area: 0.001702, ~697.0 px²)
- **Sample #2:** `field01_frame010.jpg` (Class `weed`) — Size: $0.0413 \times 0.0413$ (Area: 0.001702, ~697.0 px²)
- **Sample #3:** `field01_frame011.jpg` (Class `weed`) — Size: $0.0413 \times 0.0413$ (Area: 0.001702, ~697.0 px²)

### 4.2 Largest Annotations (Top 3)
- **Sample #1:** `field05_frame020.jpg` (Class `crop`) — Size: $0.1477 \times 0.1477$ (Area: 0.021802, ~8930.2 px²)
- **Sample #2:** `field05_frame017.jpg` (Class `crop`) — Size: $0.1477 \times 0.1477$ (Area: 0.021802, ~8930.2 px²)
- **Sample #3:** `field05_frame016.jpg` (Class `crop`) — Size: $0.1477 \times 0.1477$ (Area: 0.021802, ~8930.2 px²)

---

## 5. QA Action Plan & Guidelines

1. **Review Flagged Items:** Human annotator to inspect entries in `dataset/qa/flagged_review_list.json`.
2. **Adhere to Standards:** Refer to [`ANNOTATION_GUIDELINES.md`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/ANNOTATION_GUIDELINES.md) for plant occlusion and tiny weed bounding rules.
