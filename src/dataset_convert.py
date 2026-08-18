"""
Dataset Converter Module
Converts raw external annotations (COCO JSON, Pascal VOC XML, or non-standard CSV)
into canonical YOLO format labels (class_id x_center y_center width height).
Applies canonical class mapping:
  0: weed
  1: crop
  2: grass_lawn
  3: other
"""

import os
import json
import xml.etree.ElementTree as ET

# Canonical Class Mapping dictionary for external dataset mapping
EXTERNAL_LABEL_MAP = {
    # Weed variants
    "weed": 0,
    "broadleaf_weed": 0,
    "pigweed": 0,
    "morningglory": 0,
    "cocklebur": 0,
    "sicklepod": 0,
    "taraxacum": 0,
    "cirsium": 0,
    
    # Crop variants
    "crop": 1,
    "sugarbeet": 1,
    "sugar_beet": 1,
    "cotton": 1,
    "soybean": 1,
    "maize": 1,
    "corn": 1,
    "paddy": 1,
    
    # Grass variants
    "grass": 2,
    "grass_lawn": 2,
    "lawn": 2,
    "poa": 2,
    "crabgrass": 2,
    
    # Other/Background objects
    "other": 3,
    "stone": 3,
    "soil": 3,
    "debris": 3,
    "shadow": 3
}

def map_label_to_canonical(raw_label):
    """Maps arbitrary label string or integer to canonical class ID (0, 1, 2, 3)."""
    if isinstance(raw_label, int):
        if raw_label in [0, 1, 2, 3]:
            return raw_label
        return 3 # Default fallback to 'other'

    clean_label = str(raw_label).strip().lower().replace(" ", "_")
    return EXTERNAL_LABEL_MAP.get(clean_label, 3)

def convert_voc_xml_to_yolo(xml_file, output_txt_file, img_width, img_height):
    """Converts Pascal VOC XML format to YOLO txt format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        cls_id = map_label_to_canonical(label)
        
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        # Convert to normalized YOLO format
        xc = ((xmin + xmax) / 2.0) / img_width
        yc = ((ymin + ymax) / 2.0) / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height
        
        yolo_boxes.append((cls_id, xc, yc, w, h))
        
    with open(output_txt_file, "w") as f:
        for box in yolo_boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

def convert_coco_json_to_yolo(json_file, output_dir):
    """Converts COCO JSON annotations file to individual YOLO txt label files."""
    os.makedirs(output_dir, exist_ok=True)
    with open(json_file, "r") as f:
        data = json.load(f)
        
    categories = {cat["id"]: map_label_to_canonical(cat["name"]) for cat in data.get("categories", [])}
    images = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in data.get("images", [])}
    
    # Map image_id -> list of yolo boxes
    image_boxes = {img_id: [] for img_id in images.keys()}
    
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in images:
            continue
            
        file_name, img_w, img_h = images[img_id]
        cls_id = categories.get(ann["category_id"], 3)
        
        # COCO bbox: [xmin, ymin, width, height]
        xmin, ymin, w_abs, h_abs = ann["bbox"]
        xc = (xmin + w_abs / 2.0) / img_w
        yc = (ymin + h_abs / 2.0) / img_h
        w = w_abs / img_w
        h = h_abs / img_h
        
        image_boxes[img_id].append((cls_id, xc, yc, w, h))
        
    for img_id, boxes in image_boxes.items():
        file_name = images[img_id][0]
        base_name = os.path.splitext(file_name)[0]
        out_txt = os.path.join(output_dir, f"{base_name}.txt")
        with open(out_txt, "w") as f:
            for box in boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

if __name__ == "__main__":
    print("Dataset converter module ready. Canonical Class Mapping:")
    for k, v in sorted(EXTERNAL_LABEL_MAP.items()):
        print(f"  {k} -> Class {v}")
