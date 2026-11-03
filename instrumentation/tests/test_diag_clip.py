"""Tests for the v2 diagnose-prompt clipping in repos/HGM/prompts/self_improvement_prompt.py.

The module imports HGM's docker helpers, so the clipping block is executed on its own here.
Run from the instrumentation directory:  python -m unittest discover -s tests -v
"""

import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "..", "..", "repos", "HGM", "prompts", "self_improvement_prompt.py")
STEP = os.path.join(HERE, "..", "..", "repos", "HGM", "self_improve_step.py")


def load_clip_block(env):
    with open(PROMPTS) as f:
        src = f.read()
    block = src[src.index("DIAG_CLIP = "):src.index("def process_selfimprove_eval_logs")]
    ns = {"os": type("os", (), {"getenv": staticmethod(lambda k, d=None: env.get(k, d))})}
    exec(block, ns)
    return ns, src


@unittest.skipUnless(os.path.exists(PROMPTS), "HGM repo not present")
class TestDiagClip(unittest.TestCase):
    def setUp(self):
        self.ns, self.src = load_clip_block({"HGM_DIAG_CLIP": "1"})
        # the real diagnose template (shared by Polyglot and SWE-bench): it ends with the answer format
        tmpl = re.search(r'^diagnose_prompt = """(.*?)"""', self.src, re.S | re.M)
        self.template = tmpl.group(1) if tmpl else None

    def test_off_by_default(self):
        ns, _ = load_clip_block({})
        self.assertFalse(ns["DIAG_CLIP"])
        self.assertEqual(ns["clip"]("x" * 10, 3, "patch"), "x" * 10)

    def test_fit_prompt_keeps_head_and_answer_format(self):
        fit = self.ns["fit_prompt"]
        system = "s" * 130_000
        answer_format = "Respond precisely in the following format including the JSON start and end markers:\n```json\n<JSON>\n```"
        prompt = "TASK START " + "log line\n" * 60_000 + answer_format
        out = fit(system, prompt, 300_000)
        self.assertLessEqual(len(system) + len(out), 300_000)
        self.assertTrue(out.startswith("TASK START"))
        self.assertTrue(out.endswith(answer_format))
        self.assertIn("characters clipped to fit the model's context", out)
        self.assertEqual(fit(system, "short", 300_000), "short")

    def test_real_template_survives(self):
        if self.template is None:
            self.fail("diagnose template not found")
        tail = self.template[self.template.rindex("Respond precisely"):]
        self.assertLess(len(tail), self.ns["CLIP_KEEP_TAIL"], "answer format longer than the kept tail")
        prompt = "x" * 500_000 + self.template
        out = self.ns["fit_prompt"]("s" * 120_000, prompt, 225_000)
        self.assertTrue(out.endswith(tail))

    def test_retry_limits_shrink_by_a_quarter(self):
        with open(STEP) as f:
            step = f.read()
        self.assertIn("limit = int(CLIP_TOTAL * 0.75 ** max(0, 2 - max_attempts))", step)
        total = self.ns["CLIP_TOTAL"]
        limits = [int(total * 0.75 ** max(0, 2 - a)) for a in (2, 1, 0)]
        self.assertEqual(limits, [300_000, 225_000, 168_750])
        # worst measured density, 2.62 characters per token: the first attempt fits the context
        self.assertLess(limits[0] / 2.62 + 4096, 131_072)


if __name__ == "__main__":
    unittest.main()
