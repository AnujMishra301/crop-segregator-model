"""
Target Coordinate Transformation Module
Converts camera-relative ground offsets (dx_cam, dy_cam) into nozzle-relative target offsets (dx_noz, dy_noz).
Enforces physical frame conventions and target reachability checks.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from targeting.nozzle_calibration import NozzleCalibration

class CoordinateTransformer:
    """Coordinate transformation engine converting camera ground coordinates to nozzle target coordinates."""

    def __init__(self, nozzle_calibration=None, frame_convention="FORWARD_RIGHT_DOWN"):
        self.calibration = nozzle_calibration or NozzleCalibration()
        self.frame_convention = frame_convention

    def camera_to_nozzle(self, dx_cam, dy_cam):
        """Transforms camera-relative ground displacement (dx_cam, dy_cam)
        to nozzle-relative target displacement (dx_noz, dy_noz) in meters.
        
        Formula:
            dx_noz = dx_cam - offset_x
            dy_noz = dy_cam - offset_y
        """
        off_x, off_y, _ = self.calibration.get_offset_vector()
        
        dx_noz = dx_cam - off_x
        dy_noz = dy_cam - off_y

        return round(dx_noz, 4), round(dy_noz, 4)

    def is_target_reachable(self, dx_noz, dy_noz, max_longitudinal_m=1.5, max_lateral_m=0.8):
        """Validates if nozzle-relative target is within maximum physical boom reach and timing window."""
        in_lateral_reach = abs(dx_noz) <= max_lateral_m
        in_longitudinal_reach = abs(dy_noz) <= max_longitudinal_m
        return in_lateral_reach and in_longitudinal_reach

if __name__ == "__main__":
    calib = NozzleCalibration(offset_x=0.15, offset_y=0.20)
    tf = CoordinateTransformer(calib)
    dx_n, dy_n = tf.camera_to_nozzle(0.50, 0.60)
    print(f"[CoordinateTransform Test] Camera ground (0.50m, 0.60m) -> Nozzle target = ({dx_n}m, {dy_n}m)")
