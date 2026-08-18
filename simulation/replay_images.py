"""
Image Sequence Simulation Replay Module
Replays sequences of agricultural field images through the complete 9-stage AI vision pipeline:
Frame Stream -> Preprocessing -> Model Inference -> Detection -> Confidence Check -> Target Localization ->
Safety Checks -> Spray Decision -> Simulated Spray Trigger.
"""

import os
import sys
import time
import json
import logging
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from camera.preprocessing import FramePreprocessor
from camera.inference import RealTimeInferenceEngine
from camera.visualization import PipelineVisualizer
from targeting.target_selector import TargetSelector
from communication.protocol import MultiConditionSafetyEvaluator
from simulation.simulated_pump import SimulatedSprayPump
from simulation.simulated_drone import SimulatedDrone
from simulation.metrics import SimulationMetricsTracker

# Frame Logger setup
os.makedirs("dataset/qa", exist_ok=True)
frame_logger = logging.getLogger("SimulationFrameLogger")
frame_logger.setLevel(logging.INFO)
fh = logging.FileHandler("dataset/qa/simulation_frames.log", mode="w")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
frame_logger.addHandler(fh)

def run_image_simulation(img_dir="dataset/test/images", max_frames=20, conf_threshold=0.70):
    """Executes full simulation pipeline across image dataset."""
    img_files = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])[:max_frames] if os.path.exists(img_dir) else []

    print(f"[Simulation] Starting Image Sequence Simulation on {len(img_files)} field frames...")

    # Initialize Pipeline Components
    preprocessor = FramePreprocessor(640, 640)
    inference_engine = RealTimeInferenceEngine(conf_thresh=conf_threshold)
    target_selector = TargetSelector(img_width=640, img_height=640, default_altitude=2.0)
    safety_evaluator = MultiConditionSafetyEvaluator(conf_threshold=conf_threshold, hardware_trigger_enabled=False)
    visualizer = PipelineVisualizer(output_dir="dataset/qa/simulation_annotated_video")
    
    drone = SimulatedDrone()
    pump = SimulatedSprayPump()
    metrics = SimulationMetricsTracker()

    video_out_path = "dataset/qa/simulation_annotated_video.mp4"
    video_writer = None

    for frame_id, img_path in enumerate(img_files):
        t_start = time.perf_counter()
        t_timestamp = round(time.time(), 3)

        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            continue

        h_orig, w_orig = frame_bgr.shape[:2]

        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_out_path, fourcc, 10.0, (w_orig, h_orig))

        # 1. Preprocessing
        batch_tensor, scale_info, rgb_canvas = preprocessor.preprocess(frame_bgr)

        # 2. Model Inference & Detection
        detections, inf_latency, ts = inference_engine.run_inference(frame_bgr, t_timestamp)

        # 3. Telemetry & Target Localization
        drone_telemetry = drone.get_telemetry()
        current_alt = drone_telemetry["altitude_m"]

        target_coords = []
        highest_conf = 0.0
        weed_count = 0
        spray_candidates_count = 0
        sprays_triggered_count = 0
        reasons_not_triggered = []

        conf_list = []

        for d in detections:
            if d["class_name"] == "weed":
                weed_count += 1
                conf = d["confidence"]
                conf_list.append(conf)
                if conf > highest_conf:
                    highest_conf = conf

                # Target Localization
                target_data = target_selector.process_detection(d, altitude=current_alt)
                target_coords.append({
                    "pixel_center": target_data["pixel_center"],
                    "ground_offset": target_data["ground_offset"],
                    "nozzle_offset": target_data["nozzle_offset"]
                })

                if d.get("spray_eligible", False):
                    spray_candidates_count += 1

                    # 4. Multi-Condition Safety Checks & Simulated Spray Trigger
                    msg_payload = {
                        "sequence_id": 1000 + frame_id,
                        "target_detected": True,
                        "class": "weed",
                        "confidence": conf,
                        "spray_eligible": True,
                        "crop_conflict": False
                    }

                    approved, status, action, conds = safety_evaluator.evaluate_spray_request(
                        msg_payload, drone_state_valid=drone_telemetry["valid_flight_state"],
                        spray_armed=drone_telemetry["is_armed"], comm_healthy=True
                    )

                    if approved:
                        sprays_triggered_count += 1
                        pump.trigger_spray_pulse(target_data, frame_id=frame_id)
                    else:
                        reasons_not_triggered.append(f"Safety gate: {status}")
                else:
                    reasons_not_triggered.append(d.get("rejection_reason", "Confidence or size below threshold"))

        t_end = time.perf_counter()
        total_latency = (t_end - t_start) * 1000.0
        fps = 1000.0 / total_latency if total_latency > 0 else 0.0

        # Record metrics
        metrics.record_frame(
            frame_id, weed_count, highest_conf, spray_candidates_count,
            sprays_triggered_count, total_latency, fps, conf_list
        )

        # Log Frame Entry
        frame_log_entry = {
            "timestamp": t_timestamp,
            "frame_id": frame_id,
            "weed_count": weed_count,
            "highest_weed_confidence": round(highest_conf, 4),
            "target_coordinates": target_coords,
            "spray_eligible": (spray_candidates_count > 0),
            "spray_triggered": (sprays_triggered_count > 0),
            "reason_if_not_triggered": reasons_not_triggered if sprays_triggered_count == 0 else None
        }
        frame_logger.info(json.dumps(frame_log_entry))

        # 6. Render Visualization HUD
        annotated_frame = visualizer.draw_detections(
            frame_bgr=frame_bgr,
            detections=detections,
            fps=fps,
            latency_ms=total_latency,
            target_data=target_coords
        )
        if video_writer is not None:
            video_writer.write(annotated_frame)

        if frame_id % 5 == 0:
            print(f"  [Frame {frame_id}/{len(img_files)}] Weeds: {weed_count} | Candidates: {spray_candidates_count} | Sprays: {sprays_triggered_count} | FPS: {fps:.1f}")

    if video_writer is not None:
        video_writer.release()

    print(f"\n[Simulation] Annotated video saved to '{video_out_path}'.")
    metrics.generate_report("SIMULATION_REPORT.md")

if __name__ == "__main__":
    run_image_simulation(max_frames=20)
