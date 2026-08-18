"""
Simulated Spray Pump Actuator Module
Simulates physical solenoid nozzle actuators without hardware GPIO pins.
Tracks virtual spray pulses, pulse duration, simulated herbicide volume (mL), and logs spray events.
"""

import os
import time
import json
import logging

os.makedirs("dataset/qa", exist_ok=True)
logging.basicConfig(
    filename="dataset/qa/simulation_spray_events.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class SimulatedSprayPump:
    """Virtual spray pump actuator for hardware-in-the-loop simulation."""

    def __init__(self, flow_rate_ml_per_pulse=2.5, pulse_duration_ms=150.0):
        self.flow_rate_ml_per_pulse = flow_rate_ml_per_pulse
        self.pulse_duration_ms = pulse_duration_ms

        self.is_armed = True
        self.total_pulses = 0
        self.total_volume_ml = 0.0
        self.last_pulse_timestamp = 0.0

    def trigger_spray_pulse(self, target_info, frame_id=0):
        """Simulates a targeted spray pulse for an approved weed target."""
        if not self.is_armed:
            return False, "Pump disarmed"

        t_now = time.time()
        self.total_pulses += 1
        self.total_volume_ml += self.flow_rate_ml_per_pulse
        self.last_pulse_timestamp = t_now

        event_log = {
            "event": "SIMULATED_SPRAY_PULSE",
            "frame_id": frame_id,
            "timestamp": round(t_now, 3),
            "pulse_id": self.total_pulses,
            "pulse_duration_ms": self.pulse_duration_ms,
            "volume_dispensed_ml": self.flow_rate_ml_per_pulse,
            "cumulative_volume_ml": round(self.total_volume_ml, 2),
            "nozzle_offset": target_info.get("nozzle_offset"),
            "weed_confidence": target_info.get("weed_confidence")
        }
        logging.info(json.dumps(event_log))
        print(f"  [SimulatedPump] PULSE TRIGGERED #{self.total_pulses} (Vol: {self.flow_rate_ml_per_pulse} mL, Target Conf: {target_info.get('weed_confidence')})")

        return True, "PULSE_SUCCESS"

    def get_telemetry(self):
        """Returns virtual pump operational telemetry."""
        return {
            "is_armed": self.is_armed,
            "total_pulses": self.total_pulses,
            "total_volume_ml": round(self.total_volume_ml, 2),
            "last_pulse_timestamp": self.last_pulse_timestamp
        }

if __name__ == "__main__":
    pump = SimulatedSprayPump()
    pump.trigger_spray_pulse({"weed_confidence": 0.91, "nozzle_offset": [-0.05, 0.0]}, frame_id=1)
    print(f"[SimulatedPump Test] Telemetry: {pump.get_telemetry()}")
