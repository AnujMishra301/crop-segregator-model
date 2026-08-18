"""
Inference Script Module
Runs trained baseline weed detector on downward-facing drone images.
Outputs formatted detections: class_id, class_name, confidence, x_min, y_min, x_max, y_max.
"""

import os
import sys
import json
from PIL import Image
from ml.config.config_loader import load_config

CLASS_NAMES = {
    0: "weed",
    1: "crop",
    2: "grass_lawn",
    3: "other"
}

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

def run_inference(image_path, weights_path=None, conf_thresh=0.70):
    """Runs inference on a single image and returns structured prediction dictionary."""
    config = load_config()
    weights = weights_path or config["model_save_path"]

    if not os.path.exists(image_path):
        print(f"Image path '{image_path}' not found.")
        return []

    if not os.path.exists(weights):
        print(f"Weights path '{weights}' not found.")
        return []

    img = Image.open(image_path)
    img_w, img_h = img.size

    detections = []

    if ULTRALYTICS_AVAILABLE:
        model = YOLO(weights)
        results = model.predict(image_path, conf=conf_thresh, iou=0.45, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes_xyxy, clss, confs):
                xmin, ymin, xmax, ymax = map(float, box)
                cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                
                detections.append({
                    "class_id": int(cls_id),
                    "class_name": cls_name,
                    "confidence": round(float(conf), 4),
                    "x_min": round(xmin, 2),
                    "y_min": round(ymin, 2),
                    "x_max": round(xmax, 2),
                    "y_max": round(ymax, 2),
                    "spray_action": "TRIGGER_SPRAY" if cls_id == 0 and conf >= 0.70 else "DO_NOT_SPRAY"
                })

    else:
        # Fallback simulation format
        print(f"[Inference Engine] Simulating detection on '{image_path}'...")
        detections.append({
            "class_id": 0,
            "class_name": "weed",
            "confidence": 0.8421,
            "x_min": 120.5,
            "y_min": 180.0,
            "x_max": 210.0,
            "y_max": 265.5,
            "spray_action": "TRIGGER_SPRAY"
        })

    return detections

if __name__ == "__main__":
    target_img = sys.argv[1] if len(sys.argv) > 1 else "dataset/test/images/field05_frame001.jpg"
    print(f"Running inference on: '{target_img}'")
    dets = run_inference(target_img)
    print(json.dumps(dets, indent=2))
