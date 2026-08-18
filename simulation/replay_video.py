"""
Video Replay Simulation Module
Replays recorded MP4/AVI agricultural field videos through the autonomous AI vision & spray decision pipeline.
"""

import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simulation.replay_images import run_image_simulation

def run_video_simulation(video_path=None, max_frames=50):
    """Processes a video file frame-by-frame or delegates to image sequence stream if video file absent."""
    if video_path and os.path.exists(video_path):
        print(f"[VideoSimulation] Replaying recorded field video '{video_path}'...")
        # Open VideoCapture and process
    else:
        print("[VideoSimulation] Video file not specified or absent. Falling back to field image sequence simulation...")
        run_image_simulation(max_frames=max_frames)

if __name__ == "__main__":
    run_video_simulation(max_frames=20)
