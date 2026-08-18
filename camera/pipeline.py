"""
Real-Time Computer Vision Pipeline Orchestrator
Connects Threaded Camera Ingestion -> Preprocessing -> Inference -> NMS -> Confidence Gating ->
Target Centroid Localization -> Operational Spray Decision -> Debug Visualization -> Telemetry Logging.
"""

import os
import sys
import time
import json
import logging
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from camera.capture import ThreadedCamera
from camera.preprocessing import FramePreprocessor
from camera.inference import RealTimeInferenceEngine
from camera.visualization import PipelineVisualizer

class RealTimeCVPipeline:
    """Quadcopter onboard computer real-time vision processing pipeline."""

    def __init__(self, camera_index=0, cap_width=1280, cap_height=720,
                 img_size=640, conf_threshold=0.70, iou_threshold=0.45,
                 save_debug=True, max_frames=50):
        self.camera_index = camera_index
        self.cap_width = cap_width
        self.cap_height = cap_height
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.save_debug = save_debug
        self.max_frames = max_frames

        # Initialize Pipeline Components
        self.camera = ThreadedCamera(camera_index, cap_width, cap_height)
        self.preprocessor = FramePreprocessor(img_size, img_size)
        self.inference_engine = RealTimeInferenceEngine(conf_thresh=conf_threshold, iou_thresh=iou_threshold)
        self.visualizer = PipelineVisualizer()

        # Telemetry stats
        self.fps_history = deque(maxlen=30)
        self.latency_history = deque(maxlen=30)
        
        # Detection logger
        os.makedirs("dataset/qa", exist_ok=True)
        self.log_file = "dataset/qa/realtime_detections.log"
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

        print(f"[CVPipeline] Real-Time Vision Pipeline initialized (Conf={conf_threshold}, ImgSize={img_size}).")

    def run(self):
        """Executes real-time vision loop."""
        self.camera.start()
        time.sleep(0.5)

        frame_count = 0
        total_eligible_targets = 0

        print(f"[CVPipeline] Starting autonomous real-time processing loop (Target max_frames={self.max_frames})...")

        try:
            while frame_count < self.max_frames:
                t_loop_start = time.perf_counter()

                # 1. Non-blocking Frame Capture
                ret, frame_bgr, timestamp = self.camera.read()
                if not ret or frame_bgr is None:
                    time.sleep(0.01)
                    continue

                # 2. Resize & Normalize
                batch_tensor, scale_info, rgb_canvas = self.preprocessor.preprocess(frame_bgr)

                # 3. Model Inference, NMS & Target Localization
                detections, inf_latency_ms, ts = self.inference_engine.run_inference(frame_bgr, timestamp)

                t_loop_end = time.perf_counter()
                total_latency_ms = (t_loop_end - t_loop_start) * 1000.0
                fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

                self.fps_history.append(fps)
                self.latency_history.append(total_latency_ms)

                avg_fps = sum(self.fps_history) / len(self.fps_history)
                avg_latency = sum(self.latency_history) / len(self.latency_history)

                # Count spray candidates
                spray_candidates = [d for d in detections if d.get("spray_eligible", False)]
                total_eligible_targets += len(spray_candidates)

                # Log structured detections
                if detections:
                    log_entry = {
                        "frame_id": frame_count,
                        "timestamp": timestamp,
                        "latency_ms": round(total_latency_ms, 2),
                        "fps": round(fps, 1),
                        "detections_count": len(detections),
                        "spray_eligible_count": len(spray_candidates),
                        "detections": detections
                    }
                    logging.info(json.dumps(log_entry))

                # 4. Debug Visualization & Telemetry HUD Overlay
                annotated_frame = self.visualizer.draw_frame(
                    frame_bgr, detections, fps=avg_fps, latency_ms=avg_latency,
                    frame_id=frame_count, save_debug=self.save_debug
                )

                frame_count += 1

                if frame_count % 10 == 0:
                    print(f"  [Frame {frame_count}/{self.max_frames}] FPS: {avg_fps:.1f} | Latency: {avg_latency:.1f} ms | Detections: {len(detections)} | Spray Targets: {len(spray_candidates)}")

        except KeyboardInterrupt:
            print("[CVPipeline] User requested pipeline shutdown.")
        finally:
            self.camera.stop()
            print(f"[CVPipeline] Processing complete. Evaluated {frame_count} frames. Total Spray Eligible Targets: {total_eligible_targets}.")
            return {
                "total_frames": frame_count,
                "avg_fps": sum(self.fps_history)/len(self.fps_history) if self.fps_history else 0.0,
                "avg_latency_ms": sum(self.latency_history)/len(self.latency_history) if self.latency_history else 0.0,
                "total_eligible_targets": total_eligible_targets
            }

if __name__ == "__main__":
    pipeline = RealTimeCVPipeline(max_frames=20)
    summary = pipeline.run()
    print("\nPIPELINE EXECUTION SUMMARY:")
    print(json.dumps(summary, indent=2))
