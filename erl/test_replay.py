"""Tests for the stopping policies and the offline replay.

Run from this directory:  python -m unittest test_replay -v
"""

import json
import os
import tempfile
import unittest

import replay as rp
from policy import FixedCap, Policy, UcbVsIncumbent


def ev(ts, type_, **kw):
    return {"ts": ts, "type": type_, "run": "r", **kw}


EVENTS = [
    ev(0, "eval_result", node="initial", score=[1, 1, 0, 0], source="initial_cache"),
    ev(1, "node_created", node="a", parent="initial"),
    # HGM style: one task per result, the evaluation named "<node>_<i>"
    ev(2, "eval", node="a_0", task="t1", phase="setup", wall_s=10.0),
    ev(3, "eval", node="a_0", task="t1", phase="agent_run", wall_s=100.0),
    ev(4, "eval", node="a_1", task="t2", phase="agent_run", wall_s=50.0),
    ev(5, "eval_result", node="a", task="t2", score=[0]),
    ev(6, "eval_result", node="a", task="t1", score=[0]),
    ev(7, "node_created", node="b", parent="initial"),
    # DGM style: one batch result for several tasks
    ev(8, "eval", node="b", task="t1", phase="agent_run", wall_s=30.0),
    ev(9, "eval", node="b", task="t2", phase="agent_run", wall_s=40.0),
    ev(10, "eval_result", node="b", score=[1, 1]),
]


class TestStream(unittest.TestCase):
    def test_costs_match_tasks_and_batches(self):
        s = rp.result_stream(EVENTS)
        self.assertEqual([x[0] for x in s], ["initial", "a", "a", "b"])
        self.assertEqual([x[2] for x in s], [0.0, 50.0, 110.0, 70.0])
        self.assertEqual([x[3] for x in s], [None, "t2", "t1", None])


class TestPolicies(unittest.TestCase):
    def test_ucb_retires_only_far_below_incumbent(self):
        p = UcbVsIncumbent(0.05, min_evals=3)
        self.assertTrue(p.evaluate("x", [0, 0], 0.5))           # too few results
        self.assertFalse(p.evaluate("x", [0] * 10, 0.3))        # upper bound ~0.26 < 0.3
        self.assertTrue(p.evaluate("x", [1, 0, 1, 0, 1], 0.3))
        self.assertTrue(p.evaluate("initial", [0] * 10, 0.9))   # the root is never retired

    def test_cap(self):
        self.assertFalse(FixedCap(2).evaluate("x", [0, 1], 0.0))


class TestReplay(unittest.TestCase):
    def test_no_stopping_keeps_everything(self):
        s, kept = rp.replay(rp.result_stream(EVENTS), Policy())
        self.assertEqual((s["results"], s["skipped"], s["saved_s"]), (8, 0, 0.0))
        self.assertEqual(kept["a"], [0, 0])

    def test_retired_node_saves_its_later_time(self):
        s, kept = rp.replay(rp.result_stream(EVENTS), FixedCap(1))
        self.assertEqual(kept["a"], [0])
        self.assertEqual(s["saved_s"], 110.0)   # a's second result; b's first batch still runs
        self.assertEqual(s["retired"], 1)

    def test_shuffle_keeps_slots_and_records(self):
        import numpy as np
        s = rp.result_stream(EVENTS)
        t = rp.shuffled(s, np.random.default_rng(1))
        self.assertEqual([x[0] for x in t], [x[0] for x in s])
        self.assertEqual(sorted(x[1] for x in t if x[0] == "a"), sorted(x[1] for x in s if x[0] == "a"))

    def test_load_keeps_model_calls_as_ticks(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.jsonl")
            with open(path, "w") as f:
                for e in EVENTS[:3]:
                    f.write(json.dumps(e) + "\n")
                f.write(json.dumps(ev(2.5, "llm_call", node="a_0", tag="eval", prompt="x" * 50)) + "\n")
                f.write(json.dumps({"ts": 2.6, "type": "gpu_sample", "run": ""}) + "\n")
                f.write(json.dumps({**ev(2.7, "eval", node="z"), "run": "other"}) + "\n")
            got = rp.load(path, ["r"])["r"]
            self.assertEqual(len(got), 4)
            self.assertEqual(got[3], {"type": "tick", "ts": 2.5, "node": "a_0"})


if __name__ == "__main__":
    unittest.main()
