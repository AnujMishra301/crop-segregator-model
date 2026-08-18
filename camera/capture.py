"""
Threaded Camera Capture Module
Provides non-blocking, multi-threaded camera capture for USB webcams, Pi Camera,
or synthetic drone image fallbacks. Supports timestamping and graceful failure recovery.
"""

import os
import sys
import time
import threading
import cv2
import numpy as np

class ThreadedCamera:
    """Non-blocking multi-threaded camera reader for real-time drone vision."""

    def __init__(self, camera_index=0, cap_width=1280, cap_height=720, fps_target=30, fallback_img_dir="dataset/test/images"):
        self.camera_index = camera_index
        self.cap_width = cap_width
        self.cap_height = cap_height
        self.fps_target = fps_target
        self.fallback_img_dir = fallback_img_dir

        self.cap = None
        self.is_running = False
        self.frame = None
        self.timestamp = None
        self.lock = threading.Lock()
        self.thread = None
        self.is_fallback = False

        self._initialize_camera()

    def _initialize_camera(self):
        """Attempts to open physical USB/Pi camera; falls back to synthetic video stream if offline."""
        try:
            print(f"[CameraCapture] Opening camera index {self.camera_index} ({self.cap_width}x{self.cap_height})...")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_height)
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print(f"[CameraCapture] Camera device {self.camera_index} online.")
                    self.is_fallback = False
                    return
        except Exception as e:
            print(f"[CameraCapture] Camera hardware error: {e}")

        # Fallback to test image dataset sequence for simulated drone flight
        print(f"[CameraCapture] Physical camera unavailable. Initializing synthetic drone camera stream from '{self.fallback_img_dir}'...")
        self.is_fallback = True
        self.fallback_files = sorted([
            os.path.join(self.fallback_img_dir, f)
            for f in os.listdir(self.fallback_img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]) if os.path.exists(self.fallback_img_dir) else []
        self.fallback_idx = 0

    def start(self):
        """Starts background frame capture thread."""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print("[CameraCapture] Frame capture thread started.")

    def _update_loop(self):
        """Background thread updating the latest frame without blocking main inference loop."""
        while self.is_running:
            t_now = time.time()
            if not self.is_fallback and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.timestamp = t_now
                else:
                    print("[CameraCapture] Frame read failed. Attempting camera re-initialization...")
                    time.sleep(0.5)
                    self._initialize_camera()
            else:
                # Synthetic stream loop
                if self.fallback_files:
                    path = self.fallback_files[self.fallback_idx % len(self.fallback_files)]
                    frame = cv2.imread(path)
                    if frame is not None:
                        frame = cv2.resize(frame, (self.cap_width, self.cap_height))
                        with self.lock:
                            self.frame = frame
                            self.timestamp = t_now
                    self.fallback_idx += 1
                time.sleep(1.0 / float(self.fps_target))

    def read(self):
        """Returns the latest captured frame and timestamp."""
        with self.lock:
            if self.frame is None:
                return False, None, 0.0
            return True, self.frame.copy(), self.timestamp

    def stop(self):
        """Stops background capture thread and releases hardware resources."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("[CameraCapture] Camera hardware released.")

if __name__ == "__main__":
    cam = ThreadedCamera(cap_width=640, cap_height=480)
    cam.start()
    time.sleep(1.0)
    ret, frame, ts = cam.read()
    if ret:
        print(f"[CameraCapture Test] Success! Frame shape: {frame.shape}, Timestamp: {ts:.3f}")
    cam.stop()
