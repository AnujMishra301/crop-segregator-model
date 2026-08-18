"""
Agricultural Video Frame Extraction Tool for Dataset V2
Extracts high-quality candidate frames from raw agricultural field videos.
Filters out temporal near-duplicates using frame skipping and perceptual image difference thresholding.
"""

import os
import sys
import argparse
import cv2
import numpy as np

def compute_frame_difference(img1, img2):
    """Computes mean absolute pixel difference between two downsampled grayscale frames."""
    g1 = cv2.cvtColor(cv2.resize(img1, (160, 160)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(img2, (160, 160)), cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(g1, g2)))

def extract_frames_from_video(video_path, output_dir="dataset_v2/extracted_frames", interval_sec=1.0, diff_thresh=12.0):
    """Extracts non-duplicate frames from a single video file."""
    if not os.path.exists(video_path):
        print(f"[Error] Video file '{video_path}' not found.")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video file '{video_path}'.")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(fps * interval_sec))

    extracted_count = 0
    frame_idx = 0
    last_saved_frame = None

    print(f"[FrameExtractor] Processing '{video_name}' (FPS: {fps:.1f}, Step: {frame_step} frames)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            if last_saved_frame is not None:
                diff = compute_frame_difference(frame, last_saved_frame)
                if diff < diff_thresh:
                    frame_idx += 1
                    continue  # Skip near-duplicate frame

            out_filename = f"{video_name}_f{frame_idx:06d}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, frame)
            
            last_saved_frame = frame.copy()
            extracted_count += 1

        frame_idx += 1

    cap.release()
    print(f"[FrameExtractor] Extracted {extracted_count} candidate frames from '{video_name}' -> '{output_dir}'.")
    return extracted_count

def process_raw_video_directory(raw_dir="dataset_v2/raw", output_dir="dataset_v2/extracted_frames", interval_sec=1.0, diff_thresh=12.0):
    """Processes all raw video files in the dataset_v2/raw directory."""
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        print(f"[FrameExtractor] Raw video directory '{raw_dir}' created. Place raw field videos (.mp4, .avi, .mov) here.")
        return 0

    video_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        print(f"[FrameExtractor] No video files found in '{raw_dir}'. Place raw drone videos (.mp4, .avi) here.")
        return 0

    total_extracted = 0
    for vfile in video_files:
        total_extracted += extract_frames_from_video(vfile, output_dir, interval_sec, diff_thresh)

    print(f"[FrameExtractor] Total frames extracted across all raw videos: {total_extracted}")
    return total_extracted

def main():
    parser = argparse.ArgumentParser(description="Agricultural Video Frame Extraction Tool for Dataset V2")
    parser.add_argument("--video", type=str, help="Path to single raw video file")
    parser.add_argument("--raw_dir", type=str, default="dataset_v2/raw", help="Directory containing raw video files")
    parser.add_argument("--output_dir", type=str, default="dataset_v2/extracted_frames", help="Output directory for extracted frames")
    parser.add_argument("--interval", type=float, default=1.0, help="Frame extraction sampling interval in seconds (default: 1.0s)")
    parser.add_argument("--diff_thresh", type=float, default=12.0, help="Perceptual frame difference threshold (default: 12.0)")

    args = parser.parse_args()

    if args.video:
        extract_frames_from_video(args.video, args.output_dir, args.interval, args.diff_thresh)
    else:
        process_raw_video_directory(args.raw_dir, args.output_dir, args.interval, args.diff_thresh)

if __name__ == "__main__":
    main()
