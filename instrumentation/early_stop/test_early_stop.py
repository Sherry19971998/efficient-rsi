"""Tests for the trajectory parser and prefix features.

Run from this directory:  python -m unittest test_early_stop -v
"""

import gzip
import json
import os
import tempfile
import unittest

from features import FEATURE_NAMES, prefix_features
from trajectories import load_predictions, load_run, parse_log

LOG = """
========== User Instruction ==========
Solve the task.
========== Agent Response ==========
Response(id='r0', usage=ResponseUsage(input_tokens=100, input_tokens_details=X, output_tokens=10, total_tokens=110))
Tool Used: editor
Tool Input: {'command': 'view', 'path': '/testbed/a.py', 'file_text': None}
Tool Result: line one
line two
Tool Response: Response(id='r1', usage=ResponseUsage(input_tokens=200, input_tokens_details=X, output_tokens=20, total_tokens=220))
Tool Used: editor
Tool Input: {'command': 'edit', 'path': '/testbed/a.py', 'file_text': 'x = 1'}
Tool Result: File at /testbed/a.py has been overwritten with new content.
Tool Response: Response(id='r2', usage=ResponseUsage(input_tokens=300, input_tokens_details=X, output_tokens=30, total_tokens=330))
Tool Used: bash
Tool Input: {'command': 'python -m pytest -q'}
Tool Result: 1 failed, 2 passed
Traceback (most recent call last):
Tool Response: Response(id='r3', usage=ResponseUsage(input_tokens=400, input_tokens_details=X, output_tokens=40, total_tokens=440))
"""

# Chat Completions through vLLM (efficient-RSI runs): the call that chose a step is logged
# right before it, the first one before any tool is used. Trimmed from a real HGM log.
CHAT_LOG = """========== User Instruction ==========
Please analyze the problem description carefully.
Tool Response: ChatCompletion(id='c0', choices=[...], usage=CompletionUsage(completion_tokens=69, prompt_tokens=1188, total_tokens=1257, completion_tokens_details=None, prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cached_tokens=784, image_tokens=None)))
Tool Used: bash
Tool Input: {'command': 'ls -la /testbed/'}
Tool Result: total 4
-rw-r--r-- 1 root root 225 Oct 17 07:03 grade_school.py
Tool Response: ChatCompletion(id='c1', choices=[...], usage=CompletionUsage(completion_tokens=31, prompt_tokens=1406, total_tokens=1437, completion_tokens_details=None, prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cached_tokens=1188, image_tokens=None)))
Tool Used: editor
Tool Input: {'command': 'view', 'path': '/testbed/grade_school.py'}
Tool Result: class School:
"""


class TestEarlyStop(unittest.TestCase):
    def test_parse_chat_completions_log(self):
        steps, first, first_cached = parse_log(CHAT_LOG)
        self.assertEqual((first, first_cached), ((1188, 69), 784))
        self.assertEqual([s.tool for s in steps], ["bash", "editor"])
        # the call after step 0 chose step 1; nothing was logged after the last step
        self.assertEqual([(s.input_tokens, s.output_tokens, s.cached_tokens) for s in steps],
                         [(1406, 31, 1188), (0, 0, 0)])

    def test_load_run_reads_gzipped_files(self):
        with tempfile.TemporaryDirectory() as run:
            pred = os.path.join(run, "20261019_022332_638193", "predictions", "20261019_022332_638193_0")
            os.makedirs(pred)
            with gzip.open(os.path.join(pred, "python__a.md.gz"), "wt") as f:
                f.write(CHAT_LOG)
            with open(os.path.join(pred, "python__a.json"), "w") as f:
                json.dump({"eval_result": "unresolved"}, f)
            with open(os.path.join(pred, "python__b.md"), "w") as f:
                f.write(CHAT_LOG)
            with gzip.open(os.path.join(pred, "python__b.json.gz"), "wt") as f:
                json.dump({"eval_result": "resolved"}, f)
            os.makedirs(os.path.join(run, "not_a_node"))
            trajs = load_run(run)
        self.assertEqual([(t.agent, t.task, t.solved) for t in trajs],
                         [("20261019_022332_638193", "python__a", False), ("20261019_022332_638193", "python__b", True)])
        self.assertEqual(trajs[0].total_tokens, (1188 + 1406, 69 + 31))

    def test_selection_checks(self):
        import simulate
        # "lucky" solved its only 2 tasks; "solid" solved 9 of 12. Early stopping wrongly fails
        # two of solid's solved tasks: the mean ranking keeps solid on top of "weak", but the
        # Beta lower bound now prefers lucky, so the final pick would change.
        per_agent = {
            "lucky": [(True, True)] * 2,
            "solid": [(True, True)] * 7 + [(False, True)] * 2 + [(False, False)] * 3,
            "weak": [(True, True)] * 3 + [(False, False)] * 9,
        }
        rho, same_mean, same_lcb = simulate.rank_checks(per_agent)
        self.assertIsNone(rho)            # only 2 agents have >= 10 tasks
        self.assertIsNone(same_mean)
        no_errors = {a: [(y, y) for _, y in v] for a, v in per_agent.items()}
        self.assertTrue(simulate.rank_checks(no_errors)[2])
        # LCB: lucky 0.368; solid 0.505 with its true results but 0.355 after early stopping
        self.assertFalse(same_lcb)

    def test_parse_log(self):
        steps, first, _ = parse_log(LOG)
        self.assertEqual(first, (100, 10))
        self.assertEqual([s.tool for s in steps], ["editor", "editor", "bash"])
        self.assertEqual(steps[0].result, "line one\nline two")
        self.assertIn("Traceback", steps[2].result)
        self.assertEqual([(s.input_tokens, s.output_tokens) for s in steps], [(200, 20), (300, 30), (400, 40)])

    def test_load_and_features(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "python__demo.md"), "w") as f:
                f.write(LOG)
            with open(os.path.join(d, "python__demo.json"), "w") as f:
                json.dump({"eval_result": "resolved"}, f)
            with open(os.path.join(d, "python__nolog.json"), "w") as f:
                json.dump({"eval_result": "unresolved"}, f)
            (t,) = load_predictions(d, agent="a1")
        self.assertTrue(t.solved)
        self.assertEqual(t.total_tokens, (1000, 100))
        feats = dict(zip(FEATURE_NAMES, prefix_features(t, 3)))
        self.assertEqual(len(FEATURE_NAMES), len(prefix_features(t, 3)))
        self.assertEqual((feats["k"], feats["n_view"], feats["n_edit"], feats["n_bash"]), (3, 1, 1, 1))
        self.assertEqual((feats["n_test_runs"], feats["last_is_test"], feats["edited_after_last_test"]), (1, 1, 0))
        self.assertEqual((feats["last_error"], feats["last_fail"], feats["lang_python"]), (1, 1, 1))
        self.assertEqual(feats["tokens_so_far"], 100 + 200 + 300 + 400)
        # a prefix only sees its own steps
        self.assertEqual(dict(zip(FEATURE_NAMES, prefix_features(t, 1)))["n_bash"], 0)


if __name__ == "__main__":
    unittest.main()
