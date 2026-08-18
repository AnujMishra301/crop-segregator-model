# Dataset V2 Annotation Workflow & Standard Operating Procedure

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Document Purpose:** Dataset V2 Annotation Protocol, Class Rules, & QA Review Flow  

---

## 1. End-to-End Dataset V2 Pipeline Flowchart

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw Video      │ ──> │ Frame Extraction │ ──> │ Frame Selection │
│  (dataset_v2/   │     │ (extract_frames. │     │ (Filter near-   │
│   raw/*.mp4)    │     │  py interval=1s) │     │  duplicates)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Dataset Stats & │ <── │ Review & QA      │ <── │ BBox Annotation │
│ Readiness       │     │ (validate_dataset│     │ (YOLO format:   │
│ Dashboard       │     │  _v2.py)         │     │  0, 1, 2, 3)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐
│ Sequence-Aware  │
│ Train/Val/Test  │
│ Split (70/20/10)│
└─────────────────┘
```

---

## 2. Canonical Class Ontology Index

| Class Index | Class Name | Color Code | Description |
| :---: | :--- | :---: | :--- |
| **`0`** | **`weed`** | **Red** | Unwanted target weed species requiring precision herbicide spray. |
| **`1`** | **`crop`** | **Green** | Protected agricultural crop plant (maize, cotton, soybean, wheat, etc.). |
| **`2`** | **`grass_lawn`** | **Yellow** | Turf grass or lawn background vegetation. |
| **`3`** | **`other`** | **Cyan** | Non-target objects (pipes, rocks, soil mounds, farming tools). |

---

## 3. Detailed Bounding Box Annotation Guidelines

### 3.1 Weeds Touching or Overlapping Crops (CRITICAL FOR CROP SAFETY)
* **Rule:** Draw separate bounding boxes for the weed plant and the crop plant.
* **Tightness:** Draw tight boxes around the distinct outer leaf boundaries of both plants.
* **Rationle:** Overlapping bounding boxes allow the 5-stage decision engine to calculate exact $\text{IoU}$ spatial overlap and trigger the **Crop Safety Guard** ($\text{IoU} > 0.25$) when necessary.

### 3.2 Micro-Seedling Weeds ($<32 \times 32\text{ pixels}$)
* **Rule:** Annotate tiny weed seedlings even if they occupy only a few pixels ($15 - 30\text{ px}$).
* **Zoom Requirement:** Zoom in to $200\% - 400\%$ to draw precise tight boundaries around micro-sprouts.
* **Do NOT Group:** Do not merge multiple separated micro-seedlings into one large box. Draw individual tight boxes.

### 3.3 Multiple & Dense Weed Clustering
* **Rule:** Draw separate individual boxes for distinct weed roots/stem centers.
* **Cluster Exception:** If leaves are so tightly entangled that individual root centers cannot be distinguished, draw a single bounding box enclosing the contiguous weed mass.

### 3.4 Partially Occluded Weeds
* **Rule:** Draw the bounding box around the **visible portion** of the weed leaf canopy.
* **Edge Cutoff:** For weeds partially cut off at image frame boundaries, extend the box flush to the image border ($x=0$, $y=0$, $x=W$, or $y=H$).

### 3.5 Ambiguous Plants & Unknown Vegetation

> [!WARNING]
> **SAFETY GUARD AGAINST AMBIGUOUS LABELS:**  
> Do **NOT** automatically label an ambiguous plant as a `weed`.  
> If an annotator is uncertain whether a plant is a crop seedling or a weed, tag the image for **Review Status = NEEDS_EXPERT_REVIEW** in `DATA_COLLECTION_TRACKER.csv`.  
> False weed labels on crops introduce dangerous crop-damage risks.

### 3.6 Crop Rows & Grass Backgrounds
* **Crop Rows:** Draw individual bounding boxes for each crop plant rather than one continuous box across the whole row.
* **Turf Grass:** Label areas of turf grass as `2: grass_lawn` to train ground background separation.
