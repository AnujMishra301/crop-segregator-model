"""
Socket Protocol & Multi-Condition Safety Evaluator Module
Handles socket framing, duplicate sequence rejection, communication health monitoring,
and enforces the MANDATORY MULTI-CONDITION SAFETY RULE before issuing spray requests.
"""

import os
import sys
import time
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from communication.messages import validate_message

os.makedirs("dataset/qa", exist_ok=True)
logging.basicConfig(
    filename="dataset/qa/communication.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class CommunicationProtocol:
    """Socket protocol encoder/decoder with sequence deduplication and health tracking."""

    def __init__(self, timeout_sec=1.5):
        self.timeout_sec = timeout_sec
        self.last_sequence_id = -1
        self.seen_sequence_ids = set()
        self.last_msg_timestamp = 0.0

    def encode_message(self, message_dict):
        """Encodes JSON payload dict into line-delimited UTF-8 byte stream."""
        json_str = json.dumps(message_dict) + "\n"
        return json_str.encode("utf-8")

    def decode_message(self, raw_bytes):
        """Decodes raw socket byte stream into payload dict with validation and deduplication."""
        try:
            line = raw_bytes.decode("utf-8").strip()
            if not line:
                return None, "Empty payload"
            
            data = json.loads(line)
            valid, reason = validate_message(data)
            if not valid:
                logging.warning(f"[Protocol] Validation rejected: {reason}")
                return None, f"Validation failure: {reason}"

            seq_id = data["sequence_id"]

            # Duplicate Sequence Rejection
            if seq_id in self.seen_sequence_ids or seq_id <= self.last_sequence_id:
                logging.warning(f"[Protocol] Duplicate/out-of-order sequence ID {seq_id} rejected.")
                return None, f"Duplicate or out-of-order sequence ID {seq_id}"

            self.seen_sequence_ids.add(seq_id)
            self.last_sequence_id = seq_id
            self.last_msg_timestamp = time.time()

            return data, "SUCCESS"

        except Exception as e:
            logging.error(f"[Protocol] Decode error: {e}")
            return None, f"Decode error: {e}"

    def is_communication_healthy(self):
        """Checks if communication link is active and within timeout threshold."""
        if self.last_msg_timestamp == 0.0:
            return False
        return (time.time() - self.last_msg_timestamp) <= self.timeout_sec


class MultiConditionSafetyEvaluator:
    """Mandatory Multi-Condition Safety Gate.
    AI detection alone NEVER triggers the pump! All boolean conditions MUST be True.
    """

    def __init__(self, conf_threshold=0.70, hardware_trigger_enabled=False):
        self.conf_threshold = conf_threshold
        self.hardware_trigger_enabled = hardware_trigger_enabled  # SIMULATION MODE DEFAULT: FALSE

    def evaluate_spray_request(self, msg_data, drone_state_valid=True, spray_armed=True, comm_healthy=True):
        """
        Evaluates boolean truth table:
            SPRAY_REQUEST = (
                weed_detected == True
                AND confidence >= threshold (0.70)
                AND target_valid == True (spray_eligible == True)
                AND crop_conflict == False
                AND drone_in_valid_state == True
                AND communication_valid == True
                AND spray_system_armed == True
            )
        """
        weed_detected = bool(msg_data.get("target_detected", False) and msg_data.get("class") == "weed")
        confidence_ok = float(msg_data.get("confidence", 0.0)) >= self.conf_threshold
        target_valid = bool(msg_data.get("spray_eligible", False))
        crop_conflict = bool(msg_data.get("crop_conflict", False))

        conditions = {
            "weed_detected": weed_detected,
            "confidence_ge_thresh": confidence_ok,
            "target_valid": target_valid,
            "crop_conflict_false": not crop_conflict,
            "drone_in_valid_state": drone_state_valid,
            "communication_valid": comm_healthy,
            "spray_system_armed": spray_armed
        }

        all_conditions_met = all(conditions.values())

        if all_conditions_met:
            status = "SPRAY_REQUEST_APPROVED"
            if not self.hardware_trigger_enabled:
                action = "SIMULATED_SPRAY_LOGGED (Hardware Pump Disarmed)"
            else:
                action = "PHYSICAL_PUMP_TRIGGERED"
        else:
            status = "SPRAY_REQUEST_DENIED"
            failed_conditions = [k for k, v in conditions.items() if not v]
            action = f"DO_NOT_SPRAY (Failed safety conditions: {failed_conditions})"

        log_payload = {
            "sequence_id": msg_data.get("sequence_id"),
            "status": status,
            "action": action,
            "conditions": conditions,
            "hardware_trigger_enabled": self.hardware_trigger_enabled
        }
        logging.info(f"[SafetyEvaluator] {json.dumps(log_payload)}")

        return all_conditions_met, status, action, conditions

if __name__ == "__main__":
    evaluator = MultiConditionSafetyEvaluator()
    dummy_msg = {
        "sequence_id": 1001, "timestamp": time.time(), "target_detected": True, "class": "weed",
        "confidence": 0.91, "bbox": [100, 100, 200, 200], "pixel_center": [150, 150],
        "ground_offset": [0.1, 0.2], "nozzle_offset": [-0.05, 0.0], "spray_eligible": True,
        "crop_conflict": False, "checksum": "dummy"
    }
    approved, status, action, conds = evaluator.evaluate_spray_request(dummy_msg)
    print(f"[SafetyEvaluator Test] Approved: {approved} | Status: {status} | Action: {action}")
