# Dataset V2 Quality Assurance & Validation Report

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Document Purpose:** Dataset V2 QA Protocol, Validation Rules, and Tool Execution Logs  

---

## 1. Class Ontology & Annotation Guidelines

All Dataset V2 annotations must strictly adhere to YOLO bounding box format (`class_id x_center y_center width height` with normalized $[0.0, 1.0]$ coordinates):

| Class ID | Class Name | Description / Identification Criteria |
| :---: | :--- | :--- |
| **`0`** | **`weed`** | Target unwanted vegetation requiring precision herbicide spray. |
| **`1`** | **`crop`** | Protected agricultural crop plant (e.g. maize, soybean, cotton, wheat). |
| **`2`** | **`grass_lawn`** | Turf grass or lawn background vegetation. |
| **`3`** | **`other`** | Non-target field objects (pipes, rocks, soil mounds, farming tools). |

---

## 2. Automated Quality Assurance Verification Rules

Every image-label pair in Dataset V2 must pass 5 automated verification checks enforced by [`tools/validate_dataset_v2.py`](file:///c:/Users/mishr/OneDrive/Documents/crop-seggregation-model/tools/validate_dataset_v2.py):

1. **Bounding Box Bounds Check:** Center coordinates $0.0 \le x_c, y_c \le 1.0$ and dimensions $0.0 < w, h \le 1.0$.
2. **Class ID Integrity:** Class index strictly $\in \{0, 1, 2, 3\}$.
3. **Image Readability:** OpenCV image decoding check for corrupt `.jpg`/`.png` files.
4. **Orphan File Detection:** Flags images without label files or label files without images.
5. **Minimum Box Dimension:** Flags zero-area or corrupted negative dimension boxes.

---

## 3. Dataset V2 Validation Execution Results

Executed validation tool on `dataset_v2/images` and `dataset_v2/labels`:

```bash
python tools/validate_dataset_v2.py --img_dir dataset_v2/images --lbl_dir dataset_v2/labels
```

### Initial Workspace Scan Output:
```text
=======================================================
 DATASET V2 VALIDATION REPORT
 Image Dir: dataset_v2/images
 Label Dir: dataset_v2/labels
=======================================================

  Valid Image-Label Pairs: 0 (Awaiting raw field video frame annotation)
  Corrupt Images:         0
  Orphan Images:          0
  Orphan Labels:          0
  Total Bounding Boxes:   0
  Invalid Bounding Boxes: 0

  Class Histogram:
    Class 0 (weed): 0 boxes
    Class 1 (crop): 0 boxes
    Class 2 (grass_lawn): 0 boxes
    Class 3 (other): 0 boxes
=======================================================
```

---

## 4. Visual Verification Commands

To generate QA visualization overlays on annotated samples:

```bash
python tools/visualize_dataset_v2.py --img_dir dataset_v2/images --lbl_dir dataset_v2/labels --output_dir dataset_v2/qa --samples 10
```

Annotated sample images will be rendered with color-coded bounding boxes and saved to `dataset_v2/qa/`.
