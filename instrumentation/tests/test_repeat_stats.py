"""Tests for repeat_stats.py (noise statistics, pass@k, break-even)."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import repeat_stats as rs  # noqa: E402


class TestRepeatStats(unittest.TestCase):
    PASSES = [{"a": 1, "b": 0, "c": 1, "d": 0},
              {"a": 1, "b": 1, "c": 0, "d": 0},
              {"a": 1, "b": 0, "c": 0, "d": 0}]

    def test_noise(self):
        st = rs.noise_stats(self.PASSES)
        self.assertEqual((st["tasks"], st["always_solved"], st["never_solved"], st["flaky"]), (4, 1, 1, 2))
        self.assertAlmostEqual(st["mean_accuracy"], 5 / 12)
        # b and c each solved 1 of 3: p(1-p) = 2/9, averaged over 4 tasks, times 3/2 for R = 3
        self.assertAlmostEqual(st["within_task_var"], (2 * 2 / 9) / 4 * 1.5)
        self.assertGreater(st["tasks_needed"][0.03], st["tasks_needed"][0.10])

    def test_pass_at_k(self):
        self.assertAlmostEqual(rs.pass_at_k(self.PASSES, 1), 5 / 12)
        # pass@3 with 3 passes: any success counts -> a, b, c
        self.assertAlmostEqual(rs.pass_at_k(self.PASSES, 3), 3 / 4)
        # b: c=1 of n=3 -> pass@2 = 1 - C(2,2)/C(3,2) = 2/3
        self.assertAlmostEqual(rs.pass_at_k(self.PASSES, 2), (1 + 2 / 3 + 2 / 3 + 0) / 4)

    def test_break_even(self):
        curve = [(1, 0.40), (2, 0.50), (3, 0.55)]
        # searched agent at 50%: k=2 reaches it; 2 passes cost 2 * 2 h per 100 tasks vs 2 h for the agent
        k, pk, n = rs.break_even(13.0, 0.50, 2.0, curve, 2.0, 100)
        self.assertEqual((k, pk), (2, 0.50))
        self.assertAlmostEqual(n, 13.0 / (4.0 / 100 - 2.0 / 100))
        self.assertEqual(rs.break_even(13.0, 0.60, 2.0, curve, 2.0, 100), (None, None, None))
        self.assertEqual(rs.break_even(13.0, 0.40, 2.0, curve, 2.0, 100)[2], float("inf"))

    def test_load_passes(self):
        with tempfile.TemporaryDirectory() as d:
            for i in (1, 0):
                os.makedirs(os.path.join(d, f"rep{i}"))
                with open(os.path.join(d, f"rep{i}", "result.json"), "w") as f:
                    json.dump({"rsi_run": f"x-rep{i}", "result": {"t": i}}, f)
            self.assertEqual(rs.load_passes(d), [("x-rep0", {"t": 0}), ("x-rep1", {"t": 1})])


if __name__ == "__main__":
    unittest.main()
