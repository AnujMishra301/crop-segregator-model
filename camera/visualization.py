"""
Real-Time Debug Visualization Module
Renders bounding boxes, center pixel coordinates [cx, cy], dimensions (W x H),
class names, confidence scores, performance telemetry (FPS, Latency), and spray eligibility tags.
Saves debug visualization frames to dataset/qa/realtime_debug/.
"""

import os
import cv2
import numpy as np

class PipelineVisualizer:
    """Debug visualization engine for real-time quadcopter camera stream."""

    def __init__(self, output_dir="dataset/qa/realtime_debug"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def draw_detections(self, frame_bgr, detections, fps=0.0, latency_ms=0.0, target_data=None, save_debug=False, frame_id=0):
        """Draws bounding boxes, target crosshairs, HUD overlay, and spray state directly onto frame_bgr."""
        annotated = frame_bgr  # In-place drawing optimization
        h_frame, w_frame = frame_bgr.shape[:2]

        spray_count = 0

        for d in detections:
            bbox = [int(v) for v in d["bbox"]]
            x1, y1, x2, y2 = bbox
            cx, cy = int(d["center"][0]), int(d["center"][1])
            cls_name = d["class_name"].upper()
            conf = d["confidence"]
            width_px = d["width"]
            height_px = d["height"]
            eligible = d.get("spray_eligible", False)

            if eligible:
                spray_count += 1
                color = (0, 0, 255)      # Bright Red for Weed Spray Target
                tag = "SPRAY_ELIGIBLE"
            elif cls_name == "CROP":
                color = (0, 255, 0)      # Bright Green for Protected Crop
                tag = "PROTECTED_CROP"
            else:
                color = (0, 255, 255)    # Yellow for Grass/Other
                tag = "NO_SPRAY"

            # Draw Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw Centroid Crosshair
            cv2.circle(annotated, (cx, cy), 4, (255, 255, 0), -1)
            cv2.drawMarker(annotated, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

            # Format Display Label: e.g. "WEED 0.91 [120x85px]"
            display_str = f"{cls_name} {conf:.2f} [{int(width_px)}x{int(height_px)}px] - {tag}"
            
            # Draw background text box
            (text_w, text_h), baseline = cv2.getTextSize(display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(y1 - text_h - 6, 0)), (x1 + text_w + 4, max(y1, text_h + 6)), color, -1)
            cv2.putText(annotated, display_str, (x1 + 2, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Overlay Telemetry HUD Banner
        hud_bg = np.zeros((45, w_frame, 3), dtype=np.uint8)
        annotated[0:45, 0:w_frame] = cv2.addWeighted(annotated[0:45, 0:w_frame], 0.3, hud_bg, 0.7, 0)

        hud_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms | Spray Eligible Targets: {spray_count} | Mode: AUTONOMOUS WEED SEGREGATION"
        cv2.putText(annotated, hud_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        if save_debug:
            out_file = os.path.join(self.output_dir, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(out_file, annotated)

        return annotated

if __name__ == "__main__":
    vis = PipelineVisualizer()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = [{
        "class_name": "weed", "confidence": 0.91, "bbox": [100, 100, 200, 200],
        "center": [150, 150], "width": 100, "height": 100, "spray_eligible": True
    }]
    res = vis.draw_detections(dummy, dets, fps=25.0, latency_ms=35.0, save_debug=True)
    print(f"[Visualization Test] Rendered HUD frame successfully. Shape: {res.shape}")
