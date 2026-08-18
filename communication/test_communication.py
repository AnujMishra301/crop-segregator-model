"""
End-to-End Socket Communication & Safety Integration Test Runner
Spawns MissionControllerServer, connects AIDetectionClient, transmits target telemetry messages,
evaluates multi-condition safety truth tables, and verifies SIMULATION mode spray logs.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from communication.server import MissionControllerServer
from communication.client import AIDetectionClient

class TestSocketCommunicationAndSafety(unittest.TestCase):
    """Integration test suite for socket communication protocol and multi-condition safety evaluator."""

    @classmethod
    def setUpClass(cls):
        cls.server = MissionControllerServer(port=8889, conf_threshold=0.70, hardware_trigger_enabled=False)
        cls.server.start()
        time.sleep(0.5)

        cls.client = AIDetectionClient(port=8889)
        cls.assertTrue(cls.client.connect(), "Client failed to connect to server.")

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.server.stop()

    def test_valid_weed_target_triggers_simulated_spray(self):
        """Test that a valid weed target (conf=0.91, eligible=True, crop_conflict=False) is approved for simulated spray."""
        ack, status = self.client.send_target_detection(
            target_detected=True, class_name="weed", confidence=0.91,
            bbox=[100, 100, 200, 200], pixel_center=[150, 150],
            ground_offset=[0.10, 0.20], nozzle_offset=[-0.05, 0.0],
            spray_eligible=True, crop_conflict=False
        )
        self.assertEqual(status, "SUCCESS")
        self.assertIsNotNone(ack)
        self.assertEqual(ack["status"], "SPRAY_REQUEST_APPROVED")
        self.assertIn("SIMULATED_SPRAY_LOGGED", ack["action"])

    def test_low_confidence_weed_target_denied(self):
        """Test that a low confidence weed target (conf=0.55 < 0.70) is denied spray trigger."""
        ack, status = self.client.send_target_detection(
            target_detected=True, class_name="weed", confidence=0.55,
            bbox=[100, 100, 200, 200], pixel_center=[150, 150],
            ground_offset=[0.10, 0.20], nozzle_offset=[-0.05, 0.0],
            spray_eligible=True, crop_conflict=False
        )
        self.assertEqual(status, "SUCCESS")
        self.assertIsNotNone(ack)
        self.assertEqual(ack["status"], "SPRAY_REQUEST_DENIED")
        self.assertIn("DO_NOT_SPRAY", ack["action"])

    def test_crop_conflict_target_denied(self):
        """Test that a weed target overlapping crop (crop_conflict=True) is denied spray trigger by safety gate."""
        ack, status = self.client.send_target_detection(
            target_detected=True, class_name="weed", confidence=0.88,
            bbox=[100, 100, 200, 200], pixel_center=[150, 150],
            ground_offset=[0.10, 0.20], nozzle_offset=[-0.05, 0.0],
            spray_eligible=False, crop_conflict=True
        )
        self.assertEqual(status, "SUCCESS")
        self.assertIsNotNone(ack)
        self.assertEqual(ack["status"], "SPRAY_REQUEST_DENIED")
        self.assertIn("DO_NOT_SPRAY", ack["action"])

if __name__ == "__main__":
    print("[TestCommunication] Running socket communication and safety gate unit tests...")
    unittest.main(verbosity=2)
