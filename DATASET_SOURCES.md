# Public Agricultural Weed Datasets Audit & Sourcing Report

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Document Status:** Sourcing & Licensing Assessment  

---

## 1. Overview & Selection Criteria

For an autonomous drone carrying a downward-facing camera at approximately 2 meters ground clearance, dataset selection must meet strict criteria:
- **Perspective:** Top-down (nadir) or near-top-down perspective.
- **Annotation Type:** Bounding box annotations $(x_{min}, y_{min}, x_{max}, y_{max})$ or normalized YOLO bounding box format $(x_c, y_c, w, h)$.
- **Target Classes:** Ability to map source labels into the four canonical project classes:
  - `0`: `weed`
  - `1`: `crop`
  - `2`: `grass_lawn`
  - `3`: `other`
- **Environmental Factors:** Diverse agricultural soil colors, crop growth stages, shadowing, natural sunlight, and plant overlaps.

---

## 2. Audited Public Datasets

### 2.1 CropAndWeed Dataset (AIT / WACV 2023)
- **Source URL:** [https://github.com/cropandweed/cropandweed-dataset](https://github.com/cropandweed/cropandweed-dataset)
- **License:** Non-Commercial / Academic Research License
- **Number of Images:** ~8,587 high-resolution images (112,000+ instances)
- **Perspective:** Top-down (nadir) camera mount at ~1.1 meters clearance.
- **Annotation Format:** Multi-modal (Bounding box CSV coordinates, semantic segmentation masks, stem positions, weather/soil metadata).
- **Classes:** 74 fine-grained crop and weed species (e.g., Sugar Beet, Corn, Broadleaf Weeds, Grass Weeds).
- **Suitability for Project:** **Very High.** Camera angle matches our downward drone camera configuration.
- **Limitations:** Non-commercial restriction (academic hackathon use allowed with citation). Requires label aggregation from 74 species into 4 canonical classes.

---

### 2.2 CWFID (Crop/Weed Field Image Dataset - University of Bonn)
- **Source URL:** [https://github.com/cwfid/dataset](https://github.com/cwfid/dataset)
- **License:** Academic Research License (CVPPP 2014 / ECCV 2014)
- **Number of Images:** 320 top-view field images ($1296 \times 966$ resolution)
- **Perspective:** Top-down nadir view over organic sugar beet field.
- **Annotation Format:** PNG semantic masks and coordinate center annotations.
- **Classes:** `crop` (Sugar beet), `weed` (various broadleaf weeds), `background` (soil).
- **Suitability for Project:** **High.** Excellent for top-down weed/crop validation.
- **Limitations:** Relatively small dataset size (320 images). Must be converted from pixel masks to bounding boxes.

---

### 2.3 CottonWeedDet3 / CottonWeedDet12
- **Source URL:** [https://github.com/CottonWeedDet/CottonWeedDet](https://github.com/CottonWeedDet/CottonWeedDet)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Number of Images:** ~3,370 field images
- **Perspective:** Top-down / near-top-down views in cotton fields.
- **Annotation Format:** YOLO `.txt` bounding boxes and Pascal VOC `.xml`.
- **Classes:** Weeds (Morningglory, Pigweed, Cocklebur, Sicklepod, Palmer Amaranth, Grasses).
- **Suitability for Project:** **High.** Open CC BY 4.0 license, bounding box native annotations.
- **Limitations:** Focused predominantly on cotton crops and southern agricultural weeds; requires combining with cereal/vegetable crop datasets.

---

### 2.4 Roboflow Universe Public Agricultural Crop & Weed Datasets
- **Source URL:** [https://universe.roboflow.com](https://universe.roboflow.com)
- **License:** CC BY 4.0 / MIT License
- **Number of Images:** Multi-source subsets (~1,300 to ~4,500 images across curated field projects)
- **Perspective:** Top-down drone and handheld field imagery.
- **Annotation Format:** YOLOv8 PyTorch format, COCO JSON, Pascal VOC.
- **Classes:** `weed`, `crop`, `grass`, `soil`.
- **Suitability for Project:** **High.** Immediately compatible with YOLO pipeline and permissive open licenses.
- **Limitations:** Annotations require verification to filter out crowd-sourced labeling noise.

---

## 3. License Compliance & Usage Guidelines

1. **No Restricted Data:** Commercial datasets requiring proprietary licensing are excluded.
2. **Academic Hackathon Attribution:** All datasets used in final model training will be cited explicitly according to their respective CC BY 4.0 or academic license terms.
3. **Data Privacy & Integrity:** No personal identifiable information (PII) exists in agricultural field imagery.
