"""
MAVLink Flight Controller Telemetry Bridge Module
Translates validated weed target decisions into ArduPilot MAVLink MAV_CMD_DO_SET_SERVO / MAV_CMD_DO_REPEAT_SERVO messages
for Pixhawk companion computer communication (TELEM2 UART stream).
Maintains physical hardware arming safety guards (hardware_trigger_enabled = False by default).
"""

import os
import sys
import time
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from communication.protocol import MultiConditionSafetyEvaluator

class MAVLinkBridge:
    """MAVLink telemetry bridge interfacing companion computer to Pixhawk flight controller."""

    def __init__(self, aux_servo_channel=9, pwm_trigger_val=1900, pwm_idle_val=1100, hardware_trigger_enabled=False):
        self.aux_channel = aux_servo_channel
        self.pwm_trigger_val = pwm_trigger_val
        self.pwm_idle_val = pwm_idle_val
        self.hardware_trigger_enabled = hardware_trigger_enabled
        self.safety_evaluator = MultiConditionSafetyEvaluator(conf_threshold=0.70, hardware_trigger_enabled=hardware_trigger_enabled)

    def generate_mavlink_command(self, msg_payload, drone_state_valid=True, spray_armed=True):
        """Generates structured MAVLink MAV_CMD_DO_SET_SERVO command payload."""
        approved, status, action, conds = self.safety_evaluator.evaluate_spray_request(
            msg_payload, drone_state_valid=drone_state_valid, spray_armed=spray_armed, comm_healthy=True
        )

        mav_cmd = {
            "command_id": "MAV_CMD_DO_SET_SERVO",
            "param1_channel": self.aux_channel,
            "param2_pwm": self.pwm_trigger_val if approved and self.hardware_trigger_enabled else self.pwm_idle_val,
            "status": status,
            "action": action,
            "timestamp": time.time()
        }

        return approved, mav_cmd

if __name__ == "__main__":
    bridge = MAVLinkBridge(hardware_trigger_enabled=False)
    payload = {"sequence_id": 1001, "target_detected": True, "class": "weed", "confidence": 0.91, "spray_eligible": True}
    app, cmd = bridge.generate_mavlink_command(payload)
    print(f"[MAVLinkBridge Test] Approved: {app} | MAVLink Payload:\n{json.dumps(cmd, indent=2)}")
