"""
Operational Spray Decision Layer Engine
Decouples neural network inference from physical nozzle actuators.
Implements multi-stage target validation, configurable thresholds (default 0.70),
minimum size gating, duplicate NMS handling, and CROP OVERLAP FAIL-SAFE GUARD.
"""

import os
import sys
import json
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.config.config_loader import load_config
from ml.utils.metrics import compute_box_iou

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

CLASS_NAMES = {0: "weed", 1: "crop", 2: "grass_lawn", 3: "other"}

class SprayDecisionEngine:
    """Operational decision layer governing weed target validation and spray eligibility."""
    
    def __init__(self, weights_path=None, conf_threshold=0.70, iou_threshold=0.45,
                 crop_safety_iou_thresh=0.25, min_bbox_size=8.0, min_weed_area=0.00015):
        self.config = load_config()
        self.weights_path = weights_path or self.config["model_save_path"]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.crop_safety_iou_thresh = crop_safety_iou_thresh
        self.min_bbox_size = min_bbox_size
        self.min_weed_area = min_weed_area

        self.model = None
        if ULTRALYTICS_AVAILABLE and os.path.exists(self.weights_path):
            self.model = YOLO(self.weights_path)
            print(f"[SprayDecisionEngine] Initialized model from '{self.weights_path}' (Default Conf Thresh={self.conf_threshold}).")

    def process_frame(self, image_path_or_pil):
        """Processes an image frame through the 5-stage decision pipeline.
        Returns structured detection output dictionaries.
        """
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil)
        else:
            img = image_path_or_pil
            
        w, h = img.size

        # STAGE 1: DETECTED (Raw neural network detections)
        raw_detections = []
        if self.model is not None:
            results = self.model.predict(image_path_or_pil, conf=0.15, iou=self.iou_threshold, verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                b_xyxy = results.boxes.xyxy.cpu().numpy()
                b_cls = results.boxes.cls.cpu().numpy().astype(int)
                b_conf = results.boxes.conf.cpu().numpy()
                for box, cls_id, conf in zip(b_xyxy, b_cls, b_conf):
                    raw_detections.append({
                        "class_id": int(cls_id),
                        "class_name": CLASS_NAMES.get(int(cls_id), "other"),
                        "confidence": float(conf),
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "width": float(box[2] - box[0]),
                        "height": float(box[3] - box[1]),
                        "area_norm": float(((box[2] - box[0]) / w) * ((box[3] - box[1]) / h))
                    })

        # Separate crops and potential weed targets
        crop_detections = [d for d in raw_detections if d["class_id"] == 1]
        weed_candidates = [d for d in raw_detections if d["class_id"] == 0]

        output_results = []

        # STAGE 2: CONFIRMED TARGET & STAGE 3: CROP OVERLAP FAIL-SAFE
        for weed in weed_candidates:
            x1, y1, x2, y2 = weed["bbox"]
            cx = round((x1 + x2) / 2.0, 2)
            cy = round((y1 + y2) / 2.0, 2)
            
            bbox_rounded = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]

            status = "DETECTED"
            spray_eligible = False
            rejection_reason = None

            # Check Minimum Size & Area Gating
            if weed["width"] < self.min_bbox_size or weed["height"] < self.min_bbox_size:
                status = "REJECTED_TOO_SMALL"
                rejection_reason = f"Bbox dimensions ({weed['width']:.1f}x{weed['height']:.1f}px) below minimum threshold ({self.min_bbox_size}px)"
            elif weed["area_norm"] < self.min_weed_area:
                status = "REJECTED_TINY_AREA"
                rejection_reason = f"Normalized weed area ({weed['area_norm']:.6f}) below minimum threshold ({self.min_weed_area})"
            
            # Check Operational Confidence Threshold (>= 0.70)
            elif weed["confidence"] < self.conf_threshold:
                status = "CONFIDENCE_INSUFFICIENT"
                rejection_reason = f"Confidence ({weed['confidence']:.4f}) below operational threshold ({self.conf_threshold})"
            
            else:
                status = "CONFIRMED_TARGET"
                
                # STAGE 3: CROP OVERLAP FAIL-SAFE SAFETY GUARD
                is_crop_overlapping = False
                max_crop_iou = 0.0
                for crop in crop_detections:
                    iou = compute_box_iou(weed["bbox"], crop["bbox"])
                    if iou > max_crop_iou:
                        max_crop_iou = iou
                    if iou > self.crop_safety_iou_thresh:
                        is_crop_overlapping = True

                if is_crop_overlapping:
                    # FAIL SAFE: DO NOT SPRAY
                    status = "CROP_SAFETY_GATED"
                    spray_eligible = False
                    rejection_reason = f"FAIL-SAFE TRIGGERED: Weed overlaps crop (IoU={max_crop_iou:.3f} > {self.crop_safety_iou_thresh})"
                else:
                    # STAGE 4 & 5: SPRAY ELIGIBLE & SPRAY TRIGGERED
                    status = "SPRAY_ELIGIBLE"
                    spray_eligible = True
                    rejection_reason = None

            output_results.append({
                "class": "weed",
                "confidence": round(weed["confidence"], 4),
                "bbox": bbox_rounded,
                "center": [cx, cy],
                "status": status,
                "spray_eligible": spray_eligible,
                "rejection_reason": rejection_reason
            })

        return output_results

if __name__ == "__main__":
    test_img = "dataset/test/images/field01_frame001.jpg"
    if os.path.exists(test_img):
        engine = SprayDecisionEngine()
        decisions = engine.process_frame(test_img)
        print(f"\nOperational Spray Decisions for '{test_img}':")
        print(json.dumps(decisions, indent=2))
