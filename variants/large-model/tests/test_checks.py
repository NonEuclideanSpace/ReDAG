import unittest
from types import SimpleNamespace

import numpy as np

from agentic_image.checks.evidence import StructuredChecks
from agentic_image.checks.judge import FrozenJudge
from agentic_image.checks.reward import preservation_score, terminal_reward


class StructuredTools:
    def __init__(self):
        self.count = 2

    def call(self, operation, **arguments):
        if operation == "detect":
            return {
                "known": self.count > 0,
                "detections": [
                    {"box": [0.1 + index * 0.2, 0.1, 0.2 + index * 0.2, 0.4]}
                    for index in range(self.count)
                ],
            }
        if operation == "ocr":
            return {"known": False, "text": "HELLO", "artifact_ids": []}
        raise ValueError(operation)


class EvidenceIntegrationTests(unittest.TestCase):
    def test_structured_conflicts_and_missing_evidence_stay_unknown(self):
        tools = StructuredTools()
        checks = StructuredChecks(tools)
        judge = FrozenJudge.__new__(FrozenJudge)
        judge.config = SimpleNamespace(threshold=0.8)
        judge.thresholds = {
            "count": {"threshold": 0.8, "confirmed": True},
            "text": {"threshold": 0.8, "confirmed": True},
        }
        judge.require_calibration = True
        judge.answer_probability = lambda text, images: 0.95
        arguments = {
            "atom_id": "private:count",
            "request": "Two cups",
            "property_text": "There are two cups",
            "images": [],
            "roles": "candidate",
            "kind": "count",
        }
        expected = {"object": "cup", "count": 2}
        supported = checks.evaluate("private:count", "image:1", "count", expected)
        self.assertTrue(judge.check(**arguments, structured=supported).passed)
        tools.count = 3
        conflict = checks.evaluate("private:count", "image:1", "count", expected)
        score = judge.check(**arguments, structured=conflict)
        self.assertFalse(score.known)
        reward = terminal_reward([score], renders=1, inspects=0, actions=3, invalids=0)
        self.assertFalse(reward.success)
        self.assertEqual(reward.dense, 0)
        tools.count = 0
        missing = checks.evaluate("private:count", "image:1", "count", expected)
        self.assertFalse(missing.known)
        text = checks.evaluate("private:text", "image:1", "text", {"text": "HELLO"})
        self.assertFalse(text.known)

    def test_protection_respects_uint8_scale_for_nearly_black_images(self):
        source = np.zeros((4, 4, 3), dtype=np.uint8)
        output = np.ones_like(source)
        score = preservation_score(source, output, np.ones((4, 4), dtype=np.float32))
        self.assertAlmostEqual(score, float(np.exp(-(1 / 255) / 0.05)), places=6)


if __name__ == "__main__":
    unittest.main()
