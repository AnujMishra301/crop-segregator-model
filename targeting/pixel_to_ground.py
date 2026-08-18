"""
Pixel-to-Ground Coordinate Mapping Module
Converts 2D pixel coordinates (cx, cy) from downward-facing camera frames into
2D ground displacements (dx_cam, dy_cam) in meters using a pinhole camera model.
Accounts for camera resolution, FOV, mounting pitch/roll angles, and drone altitude.
"""

import math
import numpy as np

class PixelToGroundProjector:
    """Projector mapping 2D image pixel coordinates to physical ground coordinates in meters."""

    def __init__(self, img_width=640, img_height=640, h_fov_deg=80.0, v_fov_deg=80.0,
                 pitch_deg=0.0, roll_deg=0.0, default_altitude=2.0):
        self.img_width = float(img_width)
        self.img_height = float(img_height)
        self.h_fov_rad = math.radians(h_fov_deg)
        self.v_fov_rad = math.radians(v_fov_deg)
        self.pitch_rad = math.radians(pitch_deg)
        self.roll_rad = math.radians(roll_deg)
        self.default_altitude = float(default_altitude)

        # Optical center
        self.u0 = self.img_width / 2.0
        self.v0 = self.img_height / 2.0

        # Focal lengths in pixels
        self.fx = self.img_width / (2.0 * math.tan(self.h_fov_rad / 2.0))
        self.fy = self.img_height / (2.0 * math.tan(self.v_fov_rad / 2.0))

    def pixel_to_ground(self, cx, cy, altitude=None):
        """Converts pixel coordinate (cx, cy) to camera-relative ground displacement (dx_cam, dy_cam) in meters.
        
        Coordinate Frame Convention:
        - +X_cam: Right
        - +Y_cam: Forward (Top of image)
        - Altitude Z: Distance to ground plane (meters)
        """
        alt = float(altitude) if altitude is not None else self.default_altitude
        if alt <= 0.0:
            alt = 2.0

        # Pixel displacement from optical center
        du = float(cx) - self.u0
        dv = self.v0 - float(cy)  # Invert image Y axis so top is +Y (forward)

        # Flat ground pinhole projection
        dx_cam = (du / self.fx) * alt
        dy_cam = (dv / self.fy) * alt

        # Apply pitch/roll correction if non-zero
        if self.pitch_rad != 0.0 or self.roll_rad != 0.0:
            dx_cam += alt * math.tan(self.roll_rad)
            dy_cam += alt * math.tan(self.pitch_rad)

        return round(dx_cam, 4), round(dy_cam, 4)

    def get_ground_fov_footprint(self, altitude=None):
        """Returns the physical ground width and ground height (meters) covered by camera FOV at given altitude."""
        alt = float(altitude) if altitude is not None else self.default_altitude
        ground_w = 2.0 * alt * math.tan(self.h_fov_rad / 2.0)
        ground_h = 2.0 * alt * math.tan(self.v_fov_rad / 2.0)
        return round(ground_w, 3), round(ground_h, 3)

if __name__ == "__main__":
    proj = PixelToGroundProjector(640, 640, h_fov_deg=80.0, v_fov_deg=80.0, default_altitude=2.0)
    gw, gh = proj.get_ground_fov_footprint(2.0)
    dx, dy = proj.pixel_to_ground(320, 320, 2.0)
    print(f"[PixelToGround Test] At 2.0m altitude: Ground Footprint = {gw}m x {gh}m. Center pixel (320,320) ground offset = ({dx}m, {dy}m)")
