"""
Standalone AI Weed Detection End-to-End Test Runner
Verifies trained model inference, target centroid localization, and spray eligibility decision logic.
Purely software test — physical pumps, MAVLink commands, and hardware triggers are DISARMED.
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from ml.config.config_loader import load_config
from camera.inference import RealTimeInferenceEngine

def run_image_inference(engine, image_path, conf_threshold, save_path=None):
    """Runs inference on a single image and prints structured detection & decision telemetry."""
    if not os.path.exists(image_path):
        print(f"[Error] Image file '{image_path}' not found.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f" AI INFERENCE TEST — IMAGE: {image_path}")
    print(f" Active Confidence Threshold: {conf_threshold*100:.0f}%")
    print(f"=======================================================\n")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[Error] Failed to load image '{image_path}'.")
        sys.exit(1)

    t_start = time.perf_counter()
    detections, inf_latency, _ = engine.run_inference(img_bgr, timestamp=time.time())
    t_end = time.perf_counter()

    total_latency_ms = (t_end - t_start) * 1000.0

    print(f"Single-Frame Latency: {total_latency_ms:.1f} ms | Detections Found: {len(detections)}\n")

    annotated = img_bgr.copy()
    weed_spray_count = 0

    for idx, d in enumerate(detections):
        cls_name = d["class_name"].upper()
        conf = d["confidence"]
        cx, cy = int(d["center"][0]), int(d["center"][1])
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]

        # HARD SAFETY RULE: Spray eligibility strictly requires class == "weed" AND confidence >= 0.70
        is_spray_eligible = (cls_name == "WEED") and (conf >= 0.70) and d.get("spray_eligible", False)

        if cls_name == "WEED":
            if is_spray_eligible:
                weed_spray_count += 1
                decision_str = "SPRAY ELIGIBLE"
                reason_str = None
                color = (0, 0, 255)  # Bright Red
            elif conf < 0.70:
                decision_str = "NO SPRAY"
                reason_str = "LOW CONFIDENCE"
                color = (0, 255, 255)  # Yellow
            else:
                decision_str = "NO SPRAY"
                reason_str = d.get("rejection_reason", "CROP SAFETY GATED")
                color = (0, 255, 255)
        elif cls_name == "CROP":
            decision_str = "NO SPRAY"
            reason_str = "PROTECTED CROP"
            color = (0, 255, 0)  # Bright Green
        else:
            decision_str = "NO SPRAY"
            reason_str = "NON-TARGET CLASS"
            color = (0, 255, 255)

        # Print Terminal Output matching spec
        print(f"[{idx+1}] {cls_name}: {conf:.2f}")
        if is_spray_eligible:
            print(f"    Target: ({cx}, {cy})")
            print(f"    Decision: {decision_str}")
        else:
            print(f"    Decision: {decision_str}")
            if reason_str:
                print(f"    Reason: {reason_str}")
        print("-------------------------------------------------------")

        # Visual Bounding Box & Target Crosshair
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, (cx, cy), 5, (255, 255, 0), -1)
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

        disp_lbl = f"{cls_name}: {conf:.2f} -> {decision_str}"
        cv2.putText(annotated, disp_lbl, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save Output Frame
    out_file = save_path or "output_demo.jpg"
    cv2.imwrite(out_file, annotated)
    print(f"\n[Success] Annotated frame saved to '{out_file}'.")
    print(f"Summary: {len(detections)} targets detected, {weed_spray_count} spray eligible weed targets.\n")

def run_video_inference(engine, video_path, conf_threshold, save_path=None):
    """Runs inference on a video stream frame by frame."""
    if not os.path.exists(video_path):
        print(f"[Error] Video file '{video_path}' not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video file '{video_path}'.")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_file = save_path or "output_demo_video.mp4"
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h))

    print(f"[Demo] Processing video '{video_path}' at {conf_threshold*100:.0f}% confidence threshold...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, lat, _ = engine.run_inference(frame, timestamp=time.time())
        
        for d in detections:
            cls_name = d["class_name"].upper()
            conf = d["confidence"]
            cx, cy = int(d["center"][0]), int(d["center"][1])
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]

            is_spray_eligible = (cls_name == "WEED") and (conf >= 0.70) and d.get("spray_eligible", False)

            if is_spray_eligible:
                color = (0, 0, 255)
                lbl = f"WEED: {conf:.2f} -> SPRAY ELIGIBLE"
            elif cls_name == "WEED" and conf < 0.70:
                color = (0, 255, 255)
                lbl = f"WEED: {conf:.2f} -> NO SPRAY (LOW CONF)"
            elif cls_name == "CROP":
                color = (0, 255, 0)
                lbl = f"CROP: {conf:.2f} -> NO SPRAY"
            else:
                color = (0, 255, 255)
                lbl = f"{cls_name}: {conf:.2f} -> NO SPRAY"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, lbl, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[Success] Processed {frame_idx} video frames. Output saved to '{out_file}'.")

def run_webcam_inference(engine, camera_idx, conf_threshold):
    """Runs inference on live webcam feed."""
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[Error] Could not open camera index {camera_idx}.")
        sys.exit(1)

    print(f"[Demo] Starting live camera stream on index {camera_idx}. Press 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to capture camera frame.")
            break

        t0 = time.perf_counter()
        detections, lat, _ = engine.run_inference(frame, timestamp=time.time())
        t1 = time.perf_counter()
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0

        for d in detections:
            cls_name = d["class_name"].upper()
            conf = d["confidence"]
            cx, cy = int(d["center"][0]), int(d["center"][1])
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]

            if cls_name == "WEED" and conf >= conf_threshold:
                color = (0, 0, 255)
                lbl = f"WEED: {conf:.2f} -> SPRAY ELIGIBLE"
            elif cls_name == "CROP":
                color = (0, 255, 0)
                lbl = f"CROP: {conf:.2f} -> NO SPRAY"
            else:
                color = (0, 255, 255)
                lbl = f"{cls_name}: {conf:.2f} -> NO SPRAY"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, lbl, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {lat:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Quadcopter AI Weed Detection Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="End-to-End AI Weed Detection Inference Test Runner")
    parser.add_argument("--image", type=str, help="Path to input image file (e.g., path/to/image.jpg)")
    parser.add_argument("--video", type=str, help="Path to input video file (e.g., path/to/video.mp4)")
    parser.add_argument("--camera", type=int, help="Webcam device index (e.g., 0)")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override (e.g., 0.70)")
    parser.add_argument("--save", type=str, default=None, help="Output annotated file path")

    args = parser.parse_args()

    config = load_config()
    conf_threshold = args.conf if args.conf is not None else float(config.get("confidence_threshold", 0.70))
    weights_path = config.get("model_save_path", "ml/models/weights/best_baseline.pt")

    if not os.path.exists(weights_path):
        weights_path = "models/releases/v1.0/best_model_v1.0.pt"

    engine = RealTimeInferenceEngine(weights_path=weights_path, conf_thresh=conf_threshold)

    if args.image:
        run_image_inference(engine, args.image, conf_threshold, save_path=args.save)
    elif args.video:
        run_video_inference(engine, args.video, conf_threshold, save_path=args.save)
    elif args.camera is not None:
        run_webcam_inference(engine, args.camera, conf_threshold)
    else:
        print("[Notice] No input stream specified. Running default image test on 'dataset/test/images/field01_frame001.jpg'...")
        sample_img = "dataset/test/images/field01_frame001.jpg"
        if os.path.exists(sample_img):
            run_image_inference(engine, sample_img, conf_threshold, save_path=args.save)
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
