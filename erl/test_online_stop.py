"""The online stopper makes the same decisions as the offline replay on the same stream.

Run from this directory:  python -m unittest test_online_stop -v
"""

import json
import os
import tempfile
import unittest

import numpy as np

import online_stop as os_
import predict_stop as ps


def synthetic(seed=0, n_nodes=30, n_results=500):
    rng = np.random.default_rng(seed)
    rates = {"initial": 0.3, **{f"c{i}": float(rng.beta(2, 5)) for i in range(n_nodes)}}
    parent = {f"c{i}": ("initial" if i < 5 else f"c{rng.integers(0, i)}") for i in range(n_nodes)}
    events = [{"ts": 0.0, "type": "eval_result", "node": "initial", "score": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]}]
    stream = [("initial", [1, 0, 0, 1, 0, 0, 0, 1, 0, 0], 0.0, None, 0.0)]
    created, ts = 0, 1.0
    for _ in range(n_results):
        if created < n_nodes and rng.random() < 0.08:
            events.append({"ts": ts, "type": "node_created", "node": f"c{created}", "parent": parent[f"c{created}"]})
            created += 1
            ts += 1
        if created == 0:
            continue
        node = f"c{rng.integers(0, created)}"
        res = [int(rng.random() < rates[node])]
        stream.append((node, res, float(rng.choice([100.0, 600.0, 3600.0])), "t", ts))
        ts += 1
    return events, stream


def online(events, stream, w, history=None, **kw):
    log = []
    st = os_.OnlineStopper(w=w, log=lambda kind, **k: log.append((kind, k)), history_path=history, **kw)
    created = sorted((e["ts"], e["node"], e["parent"]) for e in events if e["type"] == "node_created")
    ci, saved = 0, 0.0
    for node, res, cost, _, ts in stream:
        while ci < len(created) and created[ci][0] <= ts:
            st.node_created(created[ci][1], created[ci][2])
            ci += 1
        if node == "initial" and node not in st.states:
            st.seed("initial", None, res)
            continue
        if st.allow(node):
            st.result(node, res, cost)
        else:
            saved += cost
    return st, saved, log


class TestMatchesReplay(unittest.TestCase):
    def check(self, w, history=None, hist_rows=None):
        for seed in range(3):
            events, stream = synthetic(seed)
            off, _ = ps.replay(events, stream, {}, w, 0.2, 3, 50, 5, hist_rows)
            st, saved, log = online(events, stream, w, history)
            self.assertFalse([x for x in log if x[0] == "erl_error"], log)
            self.assertEqual(len(st.retired), off["retired"])
            self.assertAlmostEqual(saved, off["saved_s"])

    def test_statistics_only(self):
        self.check(0)

    def test_fixed_and_auto_weight(self):
        self.check(5)
        self.check("auto")

    def test_history_rows(self):
        events, stream = synthetic(9)
        _, states = ps.replay(events, stream, {}, 0, 1e-9, 10 ** 9, 10 ** 9, 1)
        rows = ps.training_rows([v for n, v in states.items() if n != "initial"], 1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "h.json")
            with open(path, "w") as f:
                json.dump({"X": rows[0], "y": rows[1], "w": rows[2]}, f)
            self.check("auto", path, rows)

    def test_errors_answer_evaluate(self):
        log = []
        st = os_.OnlineStopper(log=lambda kind, **kw: log.append(kind))
        st.states["x"] = None          # corrupt state
        self.assertTrue(st.allow("x"))
        self.assertEqual(log, ["erl_error"])



class TestAudit(unittest.TestCase):
    def test_every_stop_is_audited_first(self):
        events, stream = synthetic(1)
        st, _, log = online(events, stream, 5, audit_p=1.0, audit_max_false=1.0)
        kinds = [k for k, _ in log]
        self.assertNotIn("erl_error", kinds)
        starts = {kw["node"] for k, kw in log if k == "erl_audit_start"}
        ends = {kw["node"]: kw for k, kw in log if k == "erl_audit_end"}
        self.assertTrue(starts)
        self.assertTrue(st.retired <= starts)                                 # no stop without an audit
        confirmed = {n for n, kw in ends.items() if not kw["false_stop"]}
        self.assertTrue(confirmed <= st.retired)
        # a revived child can be stopped later without a second audit
        self.assertTrue(st.retired - confirmed <= {n for n, kw in ends.items() if kw["false_stop"]})
        for kw in ends.values():
            self.assertEqual(kw["k"] - kw["k_start"], 5)

    def test_audit_share_follows_the_hash(self):
        st = os_.OnlineStopper(audit_p=0.3)
        share = sum(st._audited(f"20261103_{i:06d}") for i in range(2000)) / 2000
        self.assertAlmostEqual(share, 0.3, delta=0.04)

    def test_false_stops_halve_the_weight_cap(self):
        log = []
        st = os_.OnlineStopper(w=10, audit_p=1.0, log=lambda kind, **kw: log.append(kw) if kind == "erl_audit_end" else None)
        st.seed("initial", None, [1, 0, 0, 0])
        for i in range(6):
            n = f"c{i}"
            st.node_created(n, "initial")
            st.result(n, [1, 0, 1, 0, 1, 0, 1, 0], 1.0)    # as good as the best other node: could compete
            st.auditing[n] = 3
            self.assertTrue(st._allow(n))                   # false stop: revived
        self.assertEqual(st.w_cap, 10 / 4)                  # halved at the 5th and the 6th audit
        self.assertLessEqual(st.w_now, st.w_cap)
        self.assertEqual(len(log), 6)
        self.assertTrue(all(kw["false_stop"] for kw in log))


if __name__ == "__main__":
    unittest.main()
