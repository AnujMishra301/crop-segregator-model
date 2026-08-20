"""
Real-Time Inference Engine Module
Executes model inference, decodes bounding boxes, class names, and confidence scores.
Calculates centroid [cx, cy], box width, box height, applies NMS, filters operational confidence,
and measures inference latency.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.config.config_loader import load_config
from src.spray_decision import SprayDecisionEngine

CLASS_NAMES = {
    0: "crop",
    1: "weed"
}

class RealTimeInferenceEngine:
    """High-performance inference execution unit for real-time drone vision."""

    def __init__(self, weights_path=None, conf_thresh=0.70, iou_thresh=0.45):
        self.config = load_config()
        self.weights_path = weights_path or self.config["model_save_path"]
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.decision_engine = SprayDecisionEngine(
            weights_path=self.weights_path,
            conf_threshold=self.conf_thresh,
            iou_threshold=self.iou_thresh
        )
        print(f"[RealTimeInferenceEngine] Initialized inference engine with weights '{self.weights_path}'.")

    def run_inference(self, frame_bgr, timestamp=None):
        """Runs model inference and target localization on a single frame.
        Returns:
            detections: list of dicts with bbox, center, width, height, confidence, class, spray_eligible
            latency_ms: measured inference latency in milliseconds
        """
        t_start = time.perf_counter()
        
        # Temporary frame path for decision engine
        temp_path = "temp_camera_frame.jpg"
        import cv2
        cv2.imwrite(temp_path, frame_bgr)

        # Run multi-stage decision engine
        raw_decisions = self.decision_engine.process_frame(temp_path)
        t_end = time.perf_counter()
        
        latency_ms = (t_end - t_start) * 1000.0

        formatted_detections = []
        for d in raw_decisions:
            x1, y1, x2, y2 = d["bbox"]
            width = round(x2 - x1, 2)
            height = round(y2 - y1, 2)
            cx, cy = d["center"]

            formatted_detections.append({
                "class_name": d["class"],
                "class_display": f"{d['class'].upper()} {d['confidence']:.2f}",
                "confidence": d["confidence"],
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
                "width": width,
                "height": height,
                "status": d["status"],
                "spray_eligible": d["spray_eligible"],
                "rejection_reason": d.get("rejection_reason")
            })

        return formatted_detections, latency_ms, timestamp or time.time()

if __name__ == "__main__":
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    engine = RealTimeInferenceEngine()
    dets, lat, ts = engine.run_inference(dummy_bgr)
    print(f"[Inference Test] Detections count: {len(dets)}, Latency: {lat:.2f} ms")
