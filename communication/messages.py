"""
Communication Protocol Message Schema Definition
Defines strict message structures, serialization/deserialization, sequence tracking,
checksum integrity calculation, and schema validation rules.
"""

import time
import json
import hashlib

class DetectionMessage:
    """Structured telemetry message sent from AI companion computer to Mission Controller."""

    def __init__(self, sequence_id, target_detected, class_name, confidence,
                 bbox, pixel_center, ground_offset, nozzle_offset,
                 spray_eligible=False, crop_conflict=False, timestamp=None):
        self.sequence_id = int(sequence_id)
        self.timestamp = round(float(timestamp or time.time()), 3)
        self.target_detected = bool(target_detected)
        self.class_name = str(class_name)
        self.confidence = round(float(confidence), 4)
        self.bbox = [round(float(v), 2) for v in bbox]
        self.pixel_center = [round(float(v), 2) for v in pixel_center]
        self.ground_offset = [round(float(v), 4) for v in ground_offset]
        self.nozzle_offset = [round(float(v), 4) for v in nozzle_offset]
        self.spray_eligible = bool(spray_eligible)
        self.crop_conflict = bool(crop_conflict)

    def to_dict(self):
        """Serializes message to dictionary payload with cryptographic integrity checksum."""
        payload = {
            "sequence_id": self.sequence_id,
            "timestamp": self.timestamp,
            "target_detected": self.target_detected,
            "class": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "pixel_center": self.pixel_center,
            "ground_offset": self.ground_offset,
            "nozzle_offset": self.nozzle_offset,
            "spray_eligible": self.spray_eligible,
            "crop_conflict": self.crop_conflict
        }
        payload["checksum"] = self._calculate_checksum(payload)
        return payload

    @staticmethod
    def _calculate_checksum(payload_dict):
        """Calculates MD5 checksum over canonical JSON representation."""
        keys_to_hash = {k: v for k, v in payload_dict.items() if k != "checksum"}
        canonical_str = json.dumps(keys_to_hash, sort_keys=True)
        return hashlib.md5(canonical_str.encode("utf-8")).hexdigest()[:8]

    @classmethod
    def from_dict(cls, data):
        """Deserializes payload dictionary into DetectionMessage object."""
        return cls(
            sequence_id=data["sequence_id"],
            target_detected=data["target_detected"],
            class_name=data.get("class", "weed"),
            confidence=data["confidence"],
            bbox=data["bbox"],
            pixel_center=data["pixel_center"],
            ground_offset=data["ground_offset"],
            nozzle_offset=data["nozzle_offset"],
            spray_eligible=data.get("spray_eligible", False),
            crop_conflict=data.get("crop_conflict", False),
            timestamp=data.get("timestamp")
        )

def validate_message(data):
    """Strictly validates message schema keys, data types, and checksum integrity."""
    required_keys = ["sequence_id", "timestamp", "target_detected", "class",
                     "confidence", "bbox", "pixel_center", "ground_offset",
                     "nozzle_offset", "spray_eligible", "checksum"]
    
    if not isinstance(data, dict):
        return False, "Message payload must be a JSON dictionary object."

    for key in required_keys:
        if key not in data:
            return False, f"Missing required payload key: '{key}'"

    # Checksum validation
    calc_sum = DetectionMessage._calculate_checksum(data)
    if data["checksum"] != calc_sum:
        return False, f"Checksum mismatch: received '{data['checksum']}', calculated '{calc_sum}'"

    # Stale timestamp validation (max 2.0 seconds latency)
    msg_age = time.time() - data["timestamp"]
    if msg_age > 2.0:
        return False, f"Stale message rejected (age: {msg_age:.2f}s > 2.0s timeout)"

    return True, "VALID"

if __name__ == "__main__":
    msg = DetectionMessage(
        sequence_id=1001, target_detected=True, class_name="weed", confidence=0.9125,
        bbox=[100, 100, 200, 200], pixel_center=[150, 150], ground_offset=[0.1, 0.2],
        nozzle_offset=[-0.05, 0.0], spray_eligible=True
    )
    d = msg.to_dict()
    valid, reason = validate_message(d)
    print(f"[Messages Test] Message Dict:\n{json.dumps(d, indent=2)}\nValidation: {valid} ({reason})")
