"""
TFLite Edge Inference Engine Module
Independent edge inference module processing camera frames.
Decodes model outputs, applies NMS, filters confidence, produces structured JSON payloads,
and renders debug visualizations.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.config.config_loader import load_config
from src.spray_decision import SprayDecisionEngine

CLASS_NAMES = {0: "weed", 1: "crop", 2: "grass_lawn", 3: "other"}
CLASS_COLORS = {
    0: (0, 0, 255),      # Weed: Red
    1: (0, 255, 0),      # Crop: Green
    2: (255, 255, 0),    # Grass/Lawn: Yellow
    3: (255, 128, 0)     # Other: Orange
}

class TFLiteEdgeInferenceEngine:
    """Independent edge inference module running on Raspberry Pi / onboard computer."""

    def __init__(self, model_path=None, conf_thresh=0.70, iou_thresh=0.45, enable_tiling=False):
        self.config = load_config()
        self.model_path = model_path or self.config["model_save_path"]
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.enable_tiling = enable_tiling
        
        self.decision_engine = SprayDecisionEngine(
            weights_path=self.model_path,
            conf_threshold=self.conf_thresh,
            iou_threshold=self.iou_thresh
        )
        print(f"[TFLiteInferenceEngine] Ready for edge frame processing using '{self.model_path}' (High-Res Tiling={self.enable_tiling}).")

    def infer_frame(self, frame_input):
        """Processes a single frame or image file path.
        Returns structured detection output matching AI_INFERENCE_API.md schema.
        """
        timestamp = round(time.time(), 3)
        
        if isinstance(frame_input, str):
            image_path = frame_input
            image_cv = cv2.imread(image_path)
        elif isinstance(frame_input, np.ndarray):
            image_cv = frame_input
            image_path = "temp_frame.jpg"
            cv2.imwrite(image_path, frame_input)
        else:
            raise ValueError("Unsupported input format. Pass image file path or numpy array frame.")

        # Run decision pipeline
        decisions = self.decision_engine.process_frame(image_path)

        structured_detections = []
        for d in decisions:
            # Format API output
            structured_detections.append({
                "class": d["class"],
                "confidence": d["confidence"],
                "bbox": d["bbox"],
                "center": d["center"],
                "spray_eligible": d["spray_eligible"],
                "status": d["status"]
            })

        output_payload = {
            "timestamp": timestamp,
            "frame_status": "SUCCESS",
            "detections_count": len(structured_detections),
            "detections": structured_detections
        }

        return output_payload, image_cv

    def draw_debug_visualizations(self, image_cv, detections, output_path="dataset/qa/debug_tflite_output.jpg"):
        """Draws bounding boxes, labels, confidence scores, and spray eligibility on debug frames."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_out = image_cv.copy()

        for d in detections:
            bbox = [int(v) for v in d["bbox"]]
            x1, y1, x2, y2 = bbox
            cx, cy = int(d["center"][0]), int(d["center"][1])
            cls_name = d["class"]
            conf = d["confidence"]
            eligible = d.get("spray_eligible", False)

            color = (0, 0, 255) if eligible else (255, 165, 0)
            cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
            cv2.circle(img_out, (cx, cy), 4, (0, 255, 255), -1)

            label = f"{cls_name.upper()} {conf:.2f} [{'SPRAY' if eligible else 'NO_SPRAY'}]"
            cv2.putText(img_out, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(output_path, img_out)
        print(f"[TFLiteInferenceEngine] Saved debug visualization to '{output_path}'.")
        return output_path

if __name__ == "__main__":
    sample_img = "dataset/test/images/field01_frame001.jpg"
    if os.path.exists(sample_img):
        engine = TFLiteEdgeInferenceEngine()
        payload, cv_img = engine.infer_frame(sample_img)
        print("\nSTRUCTURED INFERENCE API RESPONSE:")
        print(json.dumps(payload, indent=2))
        engine.draw_debug_visualizations(cv_img, payload["detections"])
