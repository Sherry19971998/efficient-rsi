"""Tests for the per-task early-stopping replay.

Run from this directory:  python -m unittest test_micro_stop -v
"""

import unittest

import micro_stop as ms
import simulate_ledger as sl


def traj(solved, walls=(10, 20, 30, 40), agent_wall=1000.0, test_wall=5.0):
    calls = [{"ts": 100.0 + w, "latency": 1.0, "inp": 10, "cached": 0, "out": 5, "ok": True} for w in walls]
    return sl.LedgerTrajectory("n", "python__x", "resolved" if solved else "unresolved", calls,
                               agent_end=100.0 + agent_wall, agent_wall=agent_wall, test_wall=test_wall)


class TestStops(unittest.TestCase):
    def test_cap_cuts_only_long_runs(self):
        self.assertEqual(ms.stop_cap(traj(0), 600), 1000 - 600 + 5)
        self.assertIsNone(ms.stop_cap(traj(0), 1200))

    def test_classifier_waits_for_after(self):
        t = traj(0)
        probs = [0.01, 0.01, 0.01]           # P(solved) after calls 1, 2, 3
        # start is 109; call 1 ends at 110 -> stop at once, saving the rest of the run and the test
        self.assertEqual(ms.stop_classifier(t, probs, 0.95, 0), 1100 - 110 + 5)
        # only after 20 s: call 3 ends at 130 (21 s in)
        self.assertEqual(ms.stop_classifier(t, probs, 0.95, 20), 1100 - 130 + 5)
        self.assertIsNone(ms.stop_classifier(t, [0.5, 0.5, 0.5], 0.95, 0))
        self.assertIsNone(ms.stop_classifier(t, None, 0.95, 0))

    def test_conditional_cap_asks_the_classifier_at_the_limit(self):
        t = traj(0)
        # start 109; call 2 ends at 120 (11 s in) is the first past a 10 s limit
        self.assertEqual(ms.stop_conditional_cap(t, [0.9, 0.1, 0.9], 10, 0.8), 1100 - 120 + 5)
        self.assertIsNone(ms.stop_conditional_cap(t, [0.1, 0.5, 0.1], 10, 0.8))
        self.assertIsNone(ms.stop_conditional_cap(t, [0.1, 0.1, 0.1], 100, 0.8))  # never reaches the limit

    def test_edited_zeroes_and_discounts(self):
        stream = [("a", [1], 100.0, "t1"), ("a", [0], 50.0, "t2")]
        self.assertEqual(ms.edited(stream, {0: 30.0}), [("a", [0], 70.0, "t1"), ("a", [0], 50.0, "t2")])


if __name__ == "__main__":
    unittest.main()
