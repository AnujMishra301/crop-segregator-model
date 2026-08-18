"""
Simulated Quadcopter Telemetry Module
Simulates onboard drone flight parameters: altitude (H = 2.0m), ground speed (2.5 m/s),
flight mode (AUTO), and arming status.
"""

import time
import math
import random

class SimulatedDrone:
    """Virtual quadcopter flight dynamics and telemetry provider."""

    def __init__(self, base_altitude=2.0, cruising_speed=2.5):
        self.base_altitude = base_altitude
        self.cruising_speed = cruising_speed
        self.flight_mode = "AUTO"
        self.is_armed = True
        self.start_time = time.time()
        self.x_pos = 0.0
        self.y_pos = 0.0

    def get_telemetry(self):
        """Returns current flight telemetry state."""
        elapsed = time.time() - self.start_time
        
        # Simulate slight altitude disturbance (+/- 0.05m)
        alt_noise = math.sin(elapsed * 0.5) * 0.04
        current_altitude = round(self.base_altitude + alt_noise, 3)

        # Update forward flight position
        self.y_pos = round(elapsed * self.cruising_speed, 2)

        return {
            "altitude_m": current_altitude,
            "ground_speed_ms": self.cruising_speed,
            "flight_mode": self.flight_mode,
            "is_armed": self.is_armed,
            "position": [self.x_pos, self.y_pos, current_altitude],
            "valid_flight_state": (1.8 <= current_altitude <= 2.5 and self.is_armed)
        }

if __name__ == "__main__":
    drone = SimulatedDrone()
    print(f"[SimulatedDrone Test] Telemetry: {drone.get_telemetry()}")
