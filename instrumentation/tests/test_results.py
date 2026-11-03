"""Tests for final-agent selection, bootstrap intervals and compare.py (needs numpy and scipy).

Run from the instrumentation directory:
    python -m unittest discover -s tests -v
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import results
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy and scipy are not installed")
class TestResults(unittest.TestCase):
    def test_search_scores_respect_the_cut(self):
        ev = [{"type": "eval_result", "node": "initial", "score": [1, 0], "ts": 0}]
        for i in range(1, 6):
            ev.append({"type": "eval", "phase": "agent_run", "node": "a", "ts": i})
            ev.append({"type": "eval_result", "node": "a", "score": [1], "ts": i})
        scores = results.search_scores(ev, max_evals=3)
        self.assertEqual(scores, {"initial": [1, 0], "a": [1, 1, 1]})
        # search events at ts 1..5 s: 2 s of GPU occupancy ends at ts 3
        self.assertEqual(results.search_scores(ev, max_hours=2 / 3600), {"initial": [1, 0], "a": [1, 1, 1]})

    def test_selection_rule(self):
        scores = {"lucky": [1], "solid": [1] * 8 + [0, 0], "empty": []}
        self.assertEqual(results.select_best(scores, rule="mean"), "lucky")
        self.assertEqual(results.select_best(scores, rule="beta_lcb"), "solid")
        self.assertLess(results.beta_lcb([1]), results.beta_lcb([1] * 8 + [0, 0]))

    def test_bootstrap(self):
        tasks = [f"t{i}" for i in range(40)]
        half = {t: int(i % 2 == 0) for i, t in enumerate(tasks)}
        m = results.bootstrap_mean([half, half, half], tasks, n_boot=2000)
        self.assertAlmostEqual(m["mean"], 0.5)
        self.assertLess(m["ci_low"], 0.5)
        self.assertGreater(m["ci_high"], 0.5)
        ones, zeros = {t: 1 for t in tasks}, {t: 0 for t in tasks}
        d = results.bootstrap_diff([ones, ones], [zeros, zeros], tasks, n_boot=500)
        self.assertEqual((d["diff"], d["ci_low"], d["ci_high"], d["p_a_better"]), (1.0, 1.0, 1.0, 1.0))

    def test_compare_end_to_end(self):
        tasks = [f"t{i}" for i in range(20)]
        with tempfile.TemporaryDirectory() as d:
            args = []
            for label, rate in (("dgm", 0.3), ("hgm", 0.6)):
                for seed in range(3):
                    res = {t: int((i + seed) % 10 < rate * 10) for i, t in enumerate(tasks)}
                    init = {t: int(i % 10 == 0) for i, t in enumerate(tasks)}
                    path = os.path.join(d, f"{label}{seed}.json")
                    with open(path, "w") as f:
                        json.dump({"run": f"{label}-{seed}", "max_evals": 800, "heldout_tasks": tasks,
                                   "best": res, "initial": init}, f)
                    args += ["--result", f"{label}={path}"]
            out = subprocess.run([sys.executable, os.path.join(ROOT, "compare.py"), *args, "--n-boot", "500"],
                                 capture_output=True, text=True, check=True, cwd=ROOT).stdout
        self.assertIn("| hgm | 3 | 60.0%", out)
        self.assertIn("| dgm | 3 | 30.0%", out)
        self.assertIn("Differences against dgm", out)
        self.assertIn("| hgm | 30.0%", out)


if __name__ == "__main__":
    unittest.main()
