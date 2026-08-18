"""
AI Companion Computer Socket Client Module
Transmits structured weed detection telemetry payloads from AI computer to Mission Controller.
Includes auto-incrementing sequence tracking, checksum calculation, timeout handling, and retry logic.
"""

import os
import sys
import time
import json
import socket
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from communication.messages import DetectionMessage
from communication.protocol import CommunicationProtocol

class AIDetectionClient:
    """Socket client running on AI companion computer transmitting target telemetry."""

    def __init__(self, host="127.0.0.1", port=8888, timeout_sec=2.0, max_retries=3):
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

        self.sequence_counter = 1000
        self.client_socket = None
        self.protocol = CommunicationProtocol()
        self.is_connected = False

    def connect(self):
        """Connects socket to Mission Controller server."""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"[AIDetectionClient] Connecting to Mission Controller at {self.host}:{self.port} (Attempt {attempt}/{self.max_retries})...")
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.settimeout(self.timeout_sec)
                self.client_socket.connect((self.host, self.port))
                self.is_connected = True
                print(f"[AIDetectionClient] Connected successfully to {self.host}:{self.port}.")
                return True
            except Exception as e:
                print(f"[AIDetectionClient] Connection attempt {attempt} failed: {e}")
                time.sleep(0.5)

        self.is_connected = False
        return False

    def send_target_detection(self, target_detected, class_name, confidence,
                               bbox, pixel_center, ground_offset, nozzle_offset,
                               spray_eligible=False, crop_conflict=False):
        """Constructs DetectionMessage payload and sends to Mission Controller.
        Returns server ACK response.
        """
        if not self.is_connected or self.client_socket is None:
            print("[AIDetectionClient] Socket not connected. Attempting reconnection...")
            if not self.connect():
                return None, "Reconnection failed"

        self.sequence_counter += 1
        msg = DetectionMessage(
            sequence_id=self.sequence_counter,
            target_detected=target_detected,
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
            pixel_center=pixel_center,
            ground_offset=ground_offset,
            nozzle_offset=nozzle_offset,
            spray_eligible=spray_eligible,
            crop_conflict=crop_conflict
        )

        payload_dict = msg.to_dict()
        byte_data = self.protocol.encode_message(payload_dict)

        try:
            self.client_socket.sendall(byte_data)
            
            # Receive ACK response
            ack_raw = self.client_socket.recv(1024).decode("utf-8").strip()
            if ack_raw:
                ack_json = json.loads(ack_raw)
                return ack_json, "SUCCESS"
            return None, "No ACK response received"

        except Exception as e:
            print(f"[AIDetectionClient] Send failure for sequence {self.sequence_counter}: {e}")
            self.is_connected = False
            return None, f"Send error: {e}"

    def close(self):
        """Closes socket connection."""
        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
        self.is_connected = False
        print("[AIDetectionClient] Socket client closed.")

if __name__ == "__main__":
    client = AIDetectionClient()
    # Test standalone connection behavior
    client.connect()
    client.close()
