"""
Target Selector & Geometry Validation Engine
Orchestrates target selection, converts 2D weed detections into physical nozzle-relative coordinates,
evaluates target validity, and executes synthetic geometry validation unit tests.
"""

import os
import sys
import json
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from targeting.pixel_to_ground import PixelToGroundProjector
from targeting.nozzle_calibration import NozzleCalibration
from targeting.coordinate_transform import CoordinateTransformer

class TargetSelector:
    """Target selection and localization engine for autonomous weed spraying quadcopter."""

    def __init__(self, img_width=640, img_height=640, h_fov_deg=80.0, v_fov_deg=80.0,
                 offset_x=0.15, offset_y=0.20, default_altitude=2.0):
        self.projector = PixelToGroundProjector(
            img_width, img_height, h_fov_deg, v_fov_deg, default_altitude=default_altitude
        )
        self.calibration = NozzleCalibration(offset_x, offset_y)
        self.transformer = CoordinateTransformer(self.calibration)

    def process_detection(self, detection, altitude=2.0):
        """Processes a single detection dictionary and computes ground and nozzle target offsets.
        Returns target structure matching specified schema:
        {
            "weed_confidence": float,
            "pixel_center": [cx, cy],
            "ground_offset": [dx, dy],
            "nozzle_offset": [dx, dy],
            "target_valid": bool
        }
        """
        cx, cy = detection["center"]
        conf = float(detection.get("confidence", 0.0))
        spray_eligible = detection.get("spray_eligible", False)

        # 1. Pixel to Camera Ground Offset
        dx_cam, dy_cam = self.projector.pixel_to_ground(cx, cy, altitude)

        # 2. Camera Ground to Nozzle Offset
        dx_noz, dy_noz = self.transformer.camera_to_nozzle(dx_cam, dy_cam)

        # 3. Reachability and Safety Validation
        reachable = self.transformer.is_target_reachable(dx_noz, dy_noz)
        target_valid = spray_eligible and reachable

        return {
            "weed_confidence": round(conf, 4),
            "pixel_center": [round(cx, 2), round(cy, 2)],
            "ground_offset": [dx_cam, dy_cam],
            "nozzle_offset": [dx_noz, dy_noz],
            "target_valid": target_valid
        }

class TestTargetLocalizationGeometry(unittest.TestCase):
    """Synthetic geometry validation unit tests."""

    def setUp(self):
        self.selector = TargetSelector(img_width=640, img_height=640, h_fov_deg=80.0, v_fov_deg=80.0,
                                       offset_x=0.15, offset_y=0.20, default_altitude=2.0)

    def test_image_center_maps_to_zero_ground_offset(self):
        """Test that image center (320, 320) projects to (0.0, 0.0) camera ground offset."""
        det = {"center": [320.0, 320.0], "confidence": 0.91, "spray_eligible": True}
        res = self.selector.process_detection(det, altitude=2.0)
        
        self.assertEqual(res["ground_offset"], [0.0, 0.0])
        # Nozzle offset should be (0.0 - 0.15, 0.0 - 0.20) = [-0.15, -0.20]
        self.assertEqual(res["nozzle_offset"], [-0.15, -0.20])
        self.assertTrue(res["target_valid"])

    def test_altitude_scaling_proportionality(self):
        """Test that doubling altitude from 2.0m to 4.0m doubles ground offset linearly."""
        det = {"center": [480.0, 160.0], "confidence": 0.85, "spray_eligible": True}
        res_2m = self.selector.process_detection(det, altitude=2.0)
        res_4m = self.selector.process_detection(det, altitude=4.0)

        # Ground offsets at 4m should be exactly 2x ground offsets at 2m
        self.assertAlmostEqual(res_4m["ground_offset"][0], res_2m["ground_offset"][0] * 2.0, places=3)
        self.assertAlmostEqual(res_4m["ground_offset"][1], res_2m["ground_offset"][1] * 2.0, places=3)

    def test_nozzle_offset_subtraction(self):
        """Test that nozzle offset equals camera ground offset minus physical offset vector."""
        det = {"center": [400.0, 200.0], "confidence": 0.88, "spray_eligible": True}
        res = self.selector.process_detection(det, altitude=2.0)
        
        dx_c, dy_c = res["ground_offset"]
        dx_n, dy_n = res["nozzle_offset"]
        
        self.assertAlmostEqual(dx_n, dx_c - 0.15, places=4)
        self.assertAlmostEqual(dy_n, dy_c - 0.20, places=4)

if __name__ == "__main__":
    print("[TargetSelector] Running synthetic geometry validation unit tests...")
    unittest.main(verbosity=2)
