"""
Nozzle Calibration & Physical Mounting Parameters Module
Defines camera-to-nozzle mechanical offset parameters, spray pattern footprint radius,
and multi-nozzle configuration settings.
"""

class NozzleCalibration:
    """Mechanical calibration profile mapping camera optical frame to physical nozzle position."""

    def __init__(self, offset_x=0.15, offset_y=0.20, offset_z=0.0,
                 spray_radius_m=0.08, nozzle_id="PRIMARY_NOZZLE_01"):
        """
        Parameters:
            offset_x: Camera-to-nozzle lateral displacement (meters). +X = Nozzle is to the right of camera.
            offset_y: Camera-to-nozzle longitudinal displacement (meters). +Y = Nozzle is ahead of camera.
            offset_z: Vertical offset between camera and nozzle outlet (meters).
            spray_radius_m: Effective spray cone coverage radius on ground (meters).
            nozzle_id: Hardware identifier for trigger routing.
        """
        self.offset_x = float(offset_x)
        self.offset_y = float(offset_y)
        self.offset_z = float(offset_z)
        self.spray_radius_m = float(spray_radius_m)
        self.nozzle_id = nozzle_id

    def get_offset_vector(self):
        """Returns physical offset vector [dx, dy, dz] in meters."""
        return [self.offset_x, self.offset_y, self.offset_z]

    def set_calibration(self, offset_x, offset_y, offset_z=0.0):
        """Updates camera-to-nozzle mechanical offset values."""
        self.offset_x = float(offset_x)
        self.offset_y = float(offset_y)
        self.offset_z = float(offset_z)

if __name__ == "__main__":
    calib = NozzleCalibration(offset_x=0.15, offset_y=0.20)
    print(f"[NozzleCalibration Test] Nozzle '{calib.nozzle_id}' offset vector: {calib.get_offset_vector()} m, Spray radius: {calib.spray_radius_m} m")
