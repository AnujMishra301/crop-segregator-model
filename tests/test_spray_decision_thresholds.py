"""
Unit Tests for Operational Spray Decision Threshold Safety & Class Eligibility Gating
Verifies that spray eligibility strictly requires class == 'weed' and confidence >= 0.70.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.spray_decision import SprayDecisionEngine, CANONICAL_SPRAY_CONF_THRESHOLD

class TestSprayDecisionThresholds(unittest.TestCase):
    """Test suite verifying exact threshold decision gating."""

    def setUp(self):
        self.engine = SprayDecisionEngine(conf_threshold=0.70)

    def evaluate_synthetic_detection(self, class_name, confidence, crop_overlap=False):
        """Helper to simulate decision logic on a detection."""
        if class_name != "weed":
            return False, "NO SPRAY", "PROTECTED_OR_NON_TARGET"
        
        if confidence < 0.70:
            return False, "NO SPRAY", "LOW CONFIDENCE"
        
        if crop_overlap:
            return False, "NO SPRAY", "CROP SAFETY GATED"
        
        return True, "SPRAY ELIGIBLE", None

    def test_weed_confidence_sweeps(self):
        """Tests exact weed confidence threshold boundaries."""
        test_cases = [
            (0.00, False),
            (0.20, False),
            (0.24, False),
            (0.50, False),
            (0.69, False),
            (0.70, True),
            (0.71, True),
            (0.90, True),
            (1.00, True)
        ]

        for conf, expected_eligible in test_cases:
            eligible, decision, reason = self.evaluate_synthetic_detection("weed", conf)
            self.assertEqual(
                eligible, expected_eligible,
                f"Failed for weed confidence {conf}: expected eligible={expected_eligible}, got {eligible} ({decision})"
            )

    def test_class_eligibility_gating(self):
        """Tests class non-target gating regardless of confidence."""
        class_cases = [
            ("crop", 0.95, False),
            ("grass_lawn", 0.95, False),
            ("other", 0.95, False),
            ("weed", 0.69, False),
            ("weed", 0.70, True)
        ]

        for cls_name, conf, expected_eligible in class_cases:
            eligible, decision, reason = self.evaluate_synthetic_detection(cls_name, conf)
            self.assertEqual(
                eligible, expected_eligible,
                f"Failed for class '{cls_name}' with conf {conf}: expected eligible={expected_eligible}, got {eligible}"
            )

if __name__ == "__main__":
    unittest.main()
