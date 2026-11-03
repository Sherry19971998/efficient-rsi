"""Tests for the pre-evaluation replay (predict_from_diff.py).

Run from this directory:  python -m unittest test_predict_from_diff -v
"""

import json
import os
import tempfile
import unittest

import numpy as np

import predict_from_diff as pf

PATCH = """diff --git a/coding_agent.py b/coding_agent.py
index 1..2 100644
--- a/coding_agent.py
+++ b/coding_agent.py
@@ -1,3 +1,2 @@
-def forward(self):
-    pass
+model = "gpt-5"
diff --git a/tools/new_tool.py b/tools/new_tool.py
new file mode 100644
--- /dev/null
+++ b/tools/new_tool.py
@@ -0,0 +1,2 @@
+def tool():
+    return 1
diff --git a/__pycache__/llm.cpython-310.pyc b/__pycache__/llm.cpython-310.pyc
Binary files differ
diff --git a/self_evo.md b/self_evo.md
+lots of chat that must not count
"""


class TestDiffFeatures(unittest.TestCase):
    def test_counts_source_changes_only(self):
        f = pf.diff_features(PATCH)
        self.assertEqual(f["n_files"], 2)          # pycache and self_evo.md ignored
        self.assertEqual(f["n_py"], 2)
        self.assertEqual(f["added"], 3)
        self.assertEqual(f["removed"], 2)
        self.assertEqual(f["net"], 1)
        self.assertEqual(f["new_files"], 1)
        self.assertEqual(f["touches_main"], 1)
        self.assertEqual(f["touches_tools"], 1)
        self.assertEqual(f["touches_llm"], 0)
        self.assertEqual(f["main_removed"], 2)
        self.assertEqual(f["removed_defs"], 1)
        self.assertEqual(f["added_defs"], 1)
        self.assertEqual(f["provider_strings"], 1)
        self.assertEqual(f["empty"], 0)

    def test_empty_patch(self):
        self.assertEqual(pf.diff_features("")["empty"], 1)


class TestNodeRows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.ledger = os.path.join(d, "ledger.jsonl")
        self.out = os.path.join(d, "output_hgm")
        a, b = "20261020_000000_000001", "20261020_000000_000002"
        ev = [
            {"ts": 1, "type": "eval_result", "run": "hgm-fixed-r0", "node": "initial", "score": [1, 0, 0, 0]},
            {"ts": 2, "type": "node_created", "run": "hgm-fixed-r0", "node": a, "parent": "initial"},
            {"ts": 3, "type": "propose", "run": "hgm-fixed-r0", "phase": "agent_run", "node": a, "wall_s": 30.0},
            {"ts": 4, "type": "eval", "run": "hgm-fixed-r0", "phase": "agent_run", "node": a + "_0",
             "task": "python__x", "wall_s": 100.0},
            {"ts": 5, "type": "eval_result", "run": "hgm-fixed-r0", "node": a, "score": [1, 1]},
            {"ts": 6, "type": "node_created", "run": "hgm-fixed-r0", "node": b, "parent": a},
            {"ts": 7, "type": "eval", "run": "hgm-fixed-r0", "phase": "agent_run", "node": b + "_0",
             "task": "go__y", "wall_s": 50.0},
            {"ts": 8, "type": "eval_result", "run": "hgm-fixed-r0", "node": b, "score": [0, 0]},
            # skipped by the loader: a model call, and another run
            {"ts": 9, "type": "llm_call", "run": "hgm-fixed-r0", "node": b, "tag": "eval"},
            {"ts": 10, "type": "eval_result", "run": "pilot", "node": b, "score": [1]},
        ]
        with open(self.ledger, "w") as f:
            for e in ev:
                f.write(json.dumps(e) + "\n")
        for nid, entry in ((a, "python__x"), (b, "go__y")):
            os.makedirs(os.path.join(self.out, "hgm-fixed-r0", nid))
            with open(os.path.join(self.out, "hgm-fixed-r0", nid, "metadata.json"), "w") as f:
                json.dump({"entry": entry, "problem_statement": "abc"}, f)
        with open(os.path.join(self.out, "hgm-fixed-r0", b, "model_patch.diff"), "w") as f:
            f.write(PATCH)
        self.a, self.b = a, b

    def tearDown(self):
        self.tmp.cleanup()

    def test_rows_and_targets(self):
        events = pf.load_events(self.ledger, ["hgm-fixed-r0"])
        self.assertNotIn("llm_call", {e["type"] for e in events["hgm-fixed-r0"]})
        rows, nodes = pf.node_rows("hgm-fixed-r0", events["hgm-fixed-r0"],
                                   os.path.join(self.out, "hgm-fixed-r0"))
        by = {r["node"]: r for r in rows}
        self.assertEqual(set(by), {self.a, self.b})
        ra, rb = by[self.a], by[self.b]
        self.assertEqual((ra["broken"], ra["improves"], ra["depth"]), (0, 1, 1))
        self.assertAlmostEqual(ra["parent_acc"], 0.25)
        self.assertEqual(ra["propose_wall"], 30.0)
        self.assertEqual(ra["lang_python"], 1)
        self.assertEqual(ra["empty"], 1)                 # no patch file
        self.assertEqual((rb["broken"], rb["improves"], rb["depth"]), (1, 0, 2))
        self.assertEqual(rb["parent_acc"], 1.0)
        self.assertEqual(rb["eval_wall"], 50.0)
        self.assertEqual(rb["provider_strings"], 1)
        self.assertEqual(pf.ancestors(nodes, self.b), {self.a, "initial"})

    def test_skip_table_counts_lineage(self):
        rows = [dict(run="r", node="n1", eval_wall=10.0, broken=1, improves=0),
                dict(run="r", node="n2", eval_wall=30.0, broken=0, improves=1)]
        lines = pf.skip_table(rows, np.array([0.95, 0.1]), {"r": {"n2"}}, "broken")
        row = [l for l in lines if l.startswith("| 0.9 ")][0]
        self.assertIn("| 1 / 2 | 1 / 1 | 0 / 1 | 25.0% | 0 / 1 |", row)


if __name__ == "__main__":
    unittest.main()
