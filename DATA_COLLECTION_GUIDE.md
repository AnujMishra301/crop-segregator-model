# Data Collection Guide for Dataset V2

**Project:** Autonomous Agricultural Quadcopter for Targeted Weed Detection & Precision Spraying  
**Target Event:** Smart India Hackathon (SIH) 2026  
**Document Purpose:** Field Capture Standard Operating Procedure (SOP) & Video Extraction Protocol  

---

> [!IMPORTANT]
> **REAL FIELD DATA MANDATE:**  
> Synthetic image generation, stock photos, or non-agricultural outdoor images are **STRICTLY PROHIBITED** as a replacement for real field data.  
> Dataset V2 requires **$\ge 1,000$ real field images** collected under authentic quadcopter flight conditions.

---

## 1. Hardware Setup & Camera Mounting Specifications

* **Camera Pitch Angle:** Exactly $90^\circ$ downward-facing (nadir orientation perpendicular to ground).
* **Ground Clearance:** $2.0\text{ m} \pm 0.5\text{ m}$ maintained via altimeter / lidar rangefinder.
* **Camera Sensor:** Global shutter RGB sensor or high frame rate ($\ge 60\text{ FPS}$) USB 3.0 / CSI camera (e.g., Raspberry Pi Camera Module 3 / Arducam Global Shutter).
* **Focal Length & FOV:** Fixed focal length lens with $70^\circ - 85^\circ$ horizontal FOV.
* **Vibration Isolation:** Camera must be mounted on silicon rubber dampers to minimize motor vibration blur.

---

## 2. Field Imagery Collection Checklist (Environmental Splits)

Collect video footage across the following 12 distinct field environmental conditions:

1. **Normal Daylight:** Sun at $45^\circ - 60^\circ$ elevation, clear sky, uniform lighting.
2. **Bright Overhead Sunlight:** Direct noon sunlight with high ground specular reflection.
3. **Cloudy / Shadowed Ground:** Soft lighting under cloud cover or tree/propeller shadow lines.
4. **Wet / Dark Soil:** Rich dark organic soil or damp ground after irrigation.
5. **Dry / Light Soil:** Sandy or dry light clay soil with high color contrast.
6. **Micro-Weed Seedlings:** Emerging weed sprouts ($<32 \times 32\text{ pixels}$) in early growth stage.
7. **Mature Weeds:** Large weed canopies overlapping crop rows.
8. **Dense Vegetation Canopy:** Overlapping leaves where weeds and crops touch.
9. **Sparse Vegetation Canopy:** Isolated weed sprouts on open soil.
10. **Difficult Negative Crops:** Crop-only rows (maize, wheat, cotton, soybean) with ZERO weeds.
11. **Bare Soil & Tractor Tracks:** Unplanted soil with tire grooves and zero vegetation.
12. **Motion Blur Pass:** Low-altitude camera sweep at $2.5\text{ m/s}$ flight speed.

---

## 3. Frame Extraction Instructions

1. **Place Raw Video Files:** Copy field video files (`.mp4`, `.avi`, `.mov`) into `dataset_v2/raw/`.
2. **Run Frame Extractor Tool:** Execute the automated perceptual difference extraction tool:

```bash
python tools/extract_frames.py --raw_dir dataset_v2/raw --output_dir dataset_v2/extracted_frames --interval 1.0 --diff_thresh 12.0
```

This tool automatically skips near-duplicate frames and saves extracted candidates into `dataset_v2/extracted_frames/`.
