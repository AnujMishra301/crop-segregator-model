# Drone Downward Camera Annotation Guidelines

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Perspective:** Downward Nadir View ($90^\circ$ pitch angle) at $\approx 2.0\text{ m}$ Ground Clearance  
**Document Status:** Operational Annotation Standard  

---

## 1. Primary Objective & Class Taxonomy

The accurate operation of the targeted spray actuator depends on precise class distinction. Annotators must assign bounding boxes according to the four canonical class IDs:

| Class ID | Class Name | Description & Visual Indicators | Target Spray Action |
| :--- | :--- | :--- | :--- |
| `0` | **`weed`** | Broadleaf unwanted plants, wild flora, invasive vegetation growing between or within crop rows. Features irregular leaf patterns, jagged serrations, or non-uniform symmetry. | **SPRAY** (Trigger solenoid if confidence $\ge 70\%$) |
| `1` | **`crop`** | Intended cultivated plants (e.g., sugar beet, corn, cotton, soybean). Displays row alignment, uniform plant spacing, symmetric leaf structures, and healthier foliage color. | **DO NOT SPRAY** (Protected target) |
| `2` | **`grass_lawn`** | Monocotyledonous narrow blade grasses, turf, or dense lawn patches growing on field boundaries or inter-rows. | **DO NOT SPRAY** (Unless broadleaf targeted herbicide is specified) |
| `3` | **`other`** | Non-plant objects including stones, rocks, bare soil patches, plastic mulch, irrigation pipes, shadows, and agricultural debris. | **DO NOT SPRAY** |

---

## 2. Bounding Box Rules for Downward Camera Imagery

### 2.1 Tight Bounding Box Rule
- Every bounding box must tightly enclose the visible boundaries of the plant canopy.
- Do NOT include excessive bare soil or background surrounding the plant.
- The box must span from the leftmost visible leaf tip to the rightmost visible leaf tip ($x_{min}, x_{max}$) and top-most to bottom-most leaf tip ($y_{min}, y_{max}$).

---

### 2.2 Small & Emerging Seedling Weeds
- **Minimum Size Threshold:** Annotate all visible weeds down to a minimum size of $8 \times 8$ pixels in a $640 \times 640$ frame.
- **Micro-Weeds:** If a weed seedling is smaller than $6 \times 6$ pixels and indistinguishable from soil noise, do NOT annotate it.
- **Center Focus:** Ensure the bounding box centroid $(x_c, y_c)$ corresponds to the physical plant center so that nozzle alignment remains accurate.

---

### 2.3 Individual vs. Grouped Box Policy (Multiple Weeds in a Region)
- **Default Rule (Separate Boxes Required):** Each distinct weed plant MUST receive its own individual bounding box, even if growing closely adjacent to another weed.
- **Clustered Dense Weed Mats:** Only when multiple tiny weed seedlings form an inseparable, overlapping dense carpet where individual stems cannot be visually distinguished should a single bounding box labeled `0: weed` cover the cluster.
- **Crop Proximity:** Never group a weed and a crop into the same bounding box under any circumstances.

---

### 2.4 Overlapping Plants (Weed Overlapping Crop)
- **Dual Annotation:** When a weed leaf overlaps a crop leaf, draw separate bounding boxes for BOTH plants:
  - Box A: Labeled `1: crop` enclosing the full visible boundaries of the crop plant.
  - Box B: Labeled `0: weed` enclosing the full visible boundaries of the weed plant.
- **Spatial Overlap Integrity:** Annotate the estimated spatial extent of each plant even if partially occluded by an overlapping leaf.

---

### 2.5 Edge Truncation (Partially Visible Plants at Frame Boundaries)
- **Partial Boundary Plants:** If a weed or crop is cut off by the border of the camera frame, draw the bounding box up to the frame edge.
- **Inclusion Threshold:** Annotate truncated plants if at least **$20\%$** of the plant canopy is visible within the frame.
- **Exclusion Threshold:** If less than $20\%$ of a plant is visible at the extreme edge (e.g. just a tiny leaf tip), omit the annotation to prevent training bounding box regression instability.

---

### 2.6 Shadows, Lighting & Camera Motion Artifacts
- **Shadow Inclusions:** If a plant casts a strong shadow on the soil, enclose ONLY the green plant structure. Do NOT extend the bounding box to include the dark shadow region on the ground.
- **Motion Blur:** In cases of mild quadcopter motion blur, enclose the full blurred boundary of the plant foliage.

---

## 3. Annotation Verification Checklist

Before marking an annotated image frame as approved, verify:
- [ ] All broadleaf weeds are assigned Class `0` (`weed`).
- [ ] No crops are mislabeled as weeds.
- [ ] Bounding boxes tightly wrap leaf extremities without excess soil.
- [ ] No duplicate overlapping boxes exist for the same physical plant.
- [ ] Truncated plants at image edges meet the $20\%$ visibility rule.
