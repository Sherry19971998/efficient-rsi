"""Tests for the ledger-based early-stopping replay (simulate_ledger.py).

Run from this directory:  python -m unittest test_simulate_ledger -v
"""

import gzip
import json
import os
import tempfile
import unittest

import simulate_ledger as sl


def _ledger(path, events):
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


class TestLedgerReplay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.ledger = os.path.join(d, "ledger.jsonl")
        node = "20261020_101010_000001"
        ev = []
        # one evaluation: 4 calls ending at t = 10, 20, 30, 40; agent run ends at 50; tests take 5 s
        for i, t in enumerate((10, 20, 30, 40)):
            ev.append({"ts": t, "type": "llm_call", "run": "r", "tag": "eval", "node": node + "_0",
                       "task": "python__a", "latency_s": 2.0, "input_tokens": 1000 * (i + 1),
                       "cached_tokens": 500 * i, "output_tokens": 10 * (i + 1), "status": 200 if i != 2 else 400})
        ev.append({"ts": 50, "type": "eval", "run": "r", "phase": "agent_run", "wall_s": 42.0,
                   "node": node + "_0", "task": "python__a"})
        ev.append({"ts": 55, "type": "eval", "run": "r", "phase": "test", "wall_s": 5.0,
                   "node": node + "_0", "task": "python__a"})
        # noise the loader must ignore: another run, a propose call, the cached initial agent
        ev.append({"ts": 11, "type": "llm_call", "run": "other", "tag": "eval", "node": node + "_0",
                   "task": "python__a", "input_tokens": 9})
        ev.append({"ts": 12, "type": "llm_call", "run": "r", "tag": "propose", "node": node, "task": "-"})
        ev.append({"ts": 13, "type": "llm_call", "run": "r", "tag": "eval", "node": "initial", "task": "python__a"})
        _ledger(self.ledger, ev)
        self.run_dir = os.path.join(d, "run")
        pred = os.path.join(self.run_dir, node, "predictions", node + "_0")
        os.makedirs(pred)
        with gzip.open(os.path.join(pred, "python__a.json.gz"), "wt") as f:
            json.dump({"eval_result": "unresolved"}, f)
        with open(os.path.join(pred, "python__b.json"), "w") as f:  # an outcome with no calls
            json.dump({"eval_result": "resolved"}, f)
        self.node = node

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_and_build(self):
        data = sl.load_ledger(self.ledger, "r")
        self.assertEqual(list(data), [(self.node, "python__a")])
        rec = data[(self.node, "python__a")]
        self.assertEqual(([c["ts"] for c in rec["calls"]], rec["agent_end"], rec["test_wall"]), ([10, 20, 30, 40], 50, 5.0))
        trajs, no_calls = sl.build(data, sl.load_outcomes(self.run_dir))
        self.assertEqual((len(trajs), no_calls), (1, 1))
        self.assertFalse(trajs[0].solved)

    def test_features_use_only_the_prefix(self):
        (t,), _ = sl.build(sl.load_ledger(self.ledger, "r"), sl.load_outcomes(self.run_dir))
        f2 = dict(zip(sl.FEATURE_NAMES, sl.prefix_features(t, 2)))
        self.assertEqual((f2["k"], f2["elapsed_s"], f2["last_prompt"], f2["prompt_growth"]), (2, 12.0, 2000, 1000))
        self.assertEqual((f2["out_so_far"], f2["n_errors"], f2["lang_python"]), (30, 0, 1))
        self.assertEqual(dict(zip(sl.FEATURE_NAMES, sl.prefix_features(t, 3)))["n_errors"], 1)
        self.assertEqual(len(sl.FEATURE_NAMES), len(sl.prefix_features(t, 1)))

    def test_savings_are_measured(self):
        (t,), _ = sl.build(sl.load_ledger(self.ledger, "r"), sl.load_outcomes(self.run_dir))
        # stop after 2 calls (p = 0.01 for "solved"): calls 3 and 4 are skipped, and 50 - 20 s of
        # the agent run plus the 5 s test phase
        k, pred, tok, gpu, sec = sl.simulate_one([0.5, 0.01, 0.5], t, 0.9, 0.95)
        self.assertEqual((k, pred), (2, False))
        self.assertEqual(tok, (3000 + 30) + (4000 + 40))
        self.assertEqual(gpu, (3000 - 1000 + 30) + (4000 - 1500 + 40))
        self.assertEqual(sec, 30 + 5.0)
        self.assertEqual(sl.simulate_one([0.5, 0.5, 0.5], t, 0.9, 0.95), (None, False, 0, 0, 0.0))
        self.assertEqual(sl.eval_seconds(t), 47.0)


if __name__ == "__main__":
    unittest.main()
