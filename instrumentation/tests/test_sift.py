"""Pre-test input boundaries, conservative rules and A/B probability handling."""
import importlib.util
import ast
import difflib
import http.server
import json
import math
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HGM = Path(__file__).resolve().parents[2] / "repos/HGM"
sys.path.insert(0, str(HGM))
import sift
import task_records


def view(code="print('ready')\n"):
    return {"contract": {"entry_file": "coding_agent.py", "tests_visible": False},
            "parent_files": {"coding_agent.py": "print('before')\n"},
            "child_files": {"coding_agent.py": code}, "parent_fingerprints": {"coding_agent.py": "before"},
            "child_fingerprints": {"coding_agent.py": "after"}, "parent_history": [], "changed_files": ["coding_agent.py"]}


def reply(pa):
    label = "A" if pa >= .5 else "B"
    return {"choices": [{"message": {"content": label + " Example explanation."},
                         "logprobs": {"content": [{"token": label, "logprob": math.log(pa if label == "A" else 1-pa),
                            "top_logprobs": [{"token": "A", "logprob": math.log(pa)}, {"token": "B", "logprob": math.log(1-pa)}]}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 8}}


class SiftCase(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"SIFT_ENFORCE": "0", "SIFT_MODE": "off", "SIFT_ORDER_WEIGHT": "2"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_entry_syntax_error_skips_but_shadow_keeps_evaluation(self):
        report = sift.evaluate(view("def broken(:\n"), "program")
        self.assertEqual(report["decisions"]["program"]["action"], "skip")
        self.assertFalse(report["skip_evaluation"])
        with patch.dict(os.environ, {"SIFT_ENFORCE": "1"}):
            self.assertTrue(sift.evaluate(view("def broken(:\n"), "program")["skip_evaluation"])

    def test_more_loops_and_hidden_tests_are_warnings_not_certain_failure(self):
        v = view("for i in range(3):\n    print('pytest')\n")
        decision = sift.program_judge(v)
        self.assertEqual(decision["action"], "evaluate")
        self.assertEqual({r["rule"] for r in decision["warnings"]}, {"more_loops", "test_dependency"})

    def test_unreachable_file_syntax_error_is_not_definite_startup_failure(self):
        v = view()
        v["child_files"]["unused.py"] = "def bad(:"
        self.assertEqual(sift.program_judge(v)["action"], "evaluate")

    def test_qwen_one_request_gets_probability_explanation_and_usage(self):
        calls = []
        def transport(body):
            calls.append(body)
            return reply(.1)
        decision = sift.qwen_judge(view(), transport=transport)
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(decision["p_evaluate_ab"], .1)
        self.assertEqual(decision["action"], "evaluate")   # a low score orders the child later; it never skips
        self.assertIn("explanation", decision["reason"])
        self.assertEqual(decision["usage"]["prompt_tokens"], 100)

    def test_missing_alternative_is_unknown_not_zero(self):
        r = reply(.9)
        r["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = [{"token": "A", "logprob": math.log(.9)}]
        decision = sift.qwen_judge(view(), transport=lambda _: r)
        self.assertEqual(decision["status"], "unavailable")
        self.assertIsNone(decision["p_evaluate_ab"])
        self.assertEqual(decision["action"], "evaluate")

    def test_non_label_first_token_is_unknown(self):
        r = reply(.1)
        r["choices"][0]["logprobs"]["content"][0]["token"] = "<think>"
        self.assertEqual(sift.qwen_judge(view(), transport=lambda _: r)["status"], "unavailable")

    def test_compare_never_enforces(self):
        with patch.dict(os.environ, {"SIFT_ENFORCE": "1"}):
            r = sift.evaluate(view("def bad(:"), "compare", transport=lambda _: reply(.1))
        self.assertFalse(r["skip_evaluation"])

    def test_actual_skipping_requires_finite_search_time_budget(self):
        with patch.dict(os.environ, {"SIFT_MODE": "qwen", "SIFT_ENFORCE": "1"}):
            for budget in (None, 0, -1, float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    sift.validate_search_budget(budget)
            sift.validate_search_budget(1)
        with patch.dict(os.environ, {"SIFT_MODE": "compare", "SIFT_ENFORCE": "1"}):
            sift.validate_search_budget(None)

    def test_incomplete_critical_context_cannot_reject(self):
        v = view()
        v["critical_context_missing"] = True
        decision = sift.qwen_judge(v, transport=lambda _: reply(.1))
        self.assertEqual(decision["action"], "evaluate")
        self.assertEqual(decision["status"], "incomplete_input")
        with patch.dict(os.environ, {"SIFT_ENFORCE": "1"}):
            self.assertIsNone(sift.evaluate(v, "qwen", transport=lambda _: reply(.1))["order_score"])

    def test_qwen_never_skips_even_when_enforced(self):
        with patch.dict(os.environ, {"SIFT_ENFORCE": "1"}):
            r = sift.evaluate(view(), "qwen", transport=lambda _: reply(.01))
        self.assertFalse(r["skip_evaluation"])
        self.assertAlmostEqual(r["order_score"], .01)

    def test_qwen_mode_skips_definite_startup_failure_without_a_model_request(self):
        calls = []
        def transport(body):
            calls.append(body)
            return reply(.9)
        with patch.dict(os.environ, {"SIFT_ENFORCE": "1"}):
            r = sift.evaluate(view("def broken(:\n"), "qwen", transport=transport)
        self.assertTrue(r["skip_evaluation"])
        self.assertIsNone(r["order_score"])
        self.assertEqual(calls, [])
        # Record-only: the same child still gets its Qwen score, and nothing is skipped.
        r = sift.evaluate(view("def broken(:\n"), "qwen", transport=transport)
        self.assertFalse(r["skip_evaluation"])
        self.assertEqual(len(calls), 1)

    def test_order_score_only_when_enforced_qwen_and_scored(self):
        self.assertIsNone(sift.evaluate(view(), "qwen", transport=lambda _: reply(.7))["order_score"])
        with patch.dict(os.environ, {"SIFT_ENFORCE": "1"}):
            self.assertIsNone(sift.evaluate(view(), "compare", transport=lambda _: reply(.7))["order_score"])
            self.assertIsNone(sift.evaluate(view(), "program")["order_score"])
            broken = reply(.7)
            broken["choices"][0]["logprobs"] = None
            self.assertIsNone(sift.evaluate(view(), "qwen", transport=lambda _: broken)["order_score"])
            self.assertAlmostEqual(sift.evaluate(view(), "qwen", transport=lambda _: reply(.7))["order_score"], .7)

    def test_order_priors_use_rank_not_raw_value(self):
        self.assertEqual(sift.order_priors([.97], weight=2), [(1.0, 1.0)])   # one score alone: no effect
        low, high = sift.order_priors([.90, .99], weight=2)
        self.assertEqual((low, high), ((0.5, 1.5), (1.5, 0.5)))
        # Uniformly high raw values still average q = 0.5, so the group is not favoured.
        priors = sift.order_priors([.95, .96, .97, .98], weight=2)
        self.assertAlmostEqual(sum(a for a, _ in priors) / len(priors), 1.0)
        self.assertEqual(sift.order_priors([.5, .5], weight=2), [(1.0, 1.0), (1.0, 1.0)])   # ties
        self.assertEqual(sift.order_priors([None, .3], pool=[.3, .8], weight=2), [(0.0, 0.0), (0.5, 1.5)])
        self.assertEqual(sift.order_priors([.3], weight=0), [(0.0, 0.0)])
        self.assertEqual(sift.order_priors([None, None], weight=2), [(0.0, 0.0), (0.0, 0.0)])

    def test_order_weight_typo_fails_at_startup(self):
        for bad in ("-1", "nan", "inf", "two"):
            with patch.dict(os.environ, {"SIFT_ORDER_WEIGHT": bad}):
                with self.assertRaises(ValueError):
                    sift.validate_search_budget(1)

    def test_read_order_score_only_in_enforced_qwen_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp)
            (node / "sift").mkdir()
            (node / "sift/decision.json").write_text(json.dumps({"order_score": .4}))
            self.assertIsNone(sift.read_order_score(node))
            with patch.dict(os.environ, {"SIFT_MODE": "qwen", "SIFT_ENFORCE": "1"}):
                self.assertEqual(sift.read_order_score(node), .4)
                self.assertIsNone(sift.read_order_score(node / "missing"))
                for bad in (True, 1.5, None, "0.4"):
                    (node / "sift/decision.json").write_text(json.dumps({"order_score": bad}))
                    self.assertIsNone(sift.read_order_score(node))
            with patch.dict(os.environ, {"SIFT_MODE": "compare", "SIFT_ENFORCE": "1"}):
                self.assertIsNone(sift.read_order_score(node))

    def test_littlecoder_bundle_sends_changed_child_tools_not_frozen_entry(self):
        old = (HGM / "lc/tools.py").read_text()
        new = old.replace("def _bash(command: str, timeout: int = 30", "def _bash(command: str, timeout: int = 60", 1)
        self.assertNotEqual(old, new)
        diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "a/tools.py", "b/tools.py"))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "child.diff").write_text(diff)
            v = sift.build_input(HGM, [], Path(tmp) / "child.diff", Path(tmp) / "parent",
                                 agent_kind="littlecoder", timeout_s=3600)
        self.assertEqual(v["changed_files"], ["tools.py"])
        self.assertEqual(v["child_files"]["tools.py"], new)
        self.assertFalse(v["critical_context_missing"])
        self.assertNotIn("hgm_agent.py", v["child_files"])          # frozen: identical in every child
        self.assertNotIn("hgm_agent.py", v["parent_files"])
        self.assertIn("hgm_agent.py", v["child_fingerprints"])       # still visible to the program check
        message = sift.judge_message(v)
        self.assertNotIn("child_fingerprints", message)
        self.assertLessEqual(len(json.dumps(message)), 80000)

    def test_local_endpoints_only(self):
        for endpoint in ("https://api.openai.com/v1", "https://8.8.8.8/v1", "http://user:secret@localhost/v1"):
            with self.assertRaises(ValueError):
                sift._local_endpoint(endpoint)
        self.assertEqual(sift._local_endpoint("http://127.0.0.1:8000/v1"), "http://127.0.0.1:8000/v1/chat/completions")

    def test_http_client_one_local_request_without_gpu(self):
        seen = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                payload = json.dumps(reply(.1)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            def log_message(self, *args):
                pass
        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"SIFT_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1"}):
                r = sift.qwen_judge(view())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], "/v1/chat/completions")
        self.assertTrue(seen[0][1]["logprobs"])
        self.assertNotIn("child_fingerprints", seen[0][1]["messages"][1]["content"])
        self.assertEqual(r["action"], "evaluate")
        self.assertEqual(r["status"], "scored")

    def test_dgm_rejection_occurs_before_any_task_evaluation(self):
        calls = []
        def evaluate(*args, **kwargs):
            calls.append("evaluate")
            return [False]
        stub = types.SimpleNamespace(sample_child=lambda *a, **k: "child", sift_child=lambda *a: "failed", eval_agent=evaluate)
        ns = {"hgm_utils": stub, "rsi_ledger": types.SimpleNamespace(event=lambda *a, **k: None), "TEST_MORE_THRESHOLD": .5}
        tree = ast.parse((HGM / "dgm_search.py").read_text())
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_and_evaluate"], type_ignores=[]), "dgm_search.py", "exec"), ns)
        self.assertIsNone(ns["create_and_evaluate"]("parent", "image", "source", ["task"], ["task"], 1))
        self.assertEqual(calls, [])
        stub.sift_child = lambda *a: "child"
        self.assertEqual(ns["create_and_evaluate"]("parent", "image", "source", ["task"], ["task"], 1), "child")
        self.assertEqual(calls, ["evaluate"])

    def test_reconstruction_and_gate_persist_input_without_child_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "initial_source"
            initial.mkdir()
            (initial / "coding_agent_polyglot.py").write_text("print('before')\n")
            parent = root / "parent"
            parent.mkdir()
            child = root / "child"
            child.mkdir()
            (child / "metadata.json").write_text(json.dumps({"parent_commit": "parent"}))
            (child / "secret_child_results.json").write_text('{"future_score": 1.0}')
            (child / "model_patch.diff").write_text("--- a/coding_agent.py\n+++ b/coding_agent.py\n@@ -1 +1 @@\n-print('before')\n+def broken(:\n")
            with patch.dict(os.environ, {"SIFT_MODE": "program", "SIFT_ENFORCE": "1"}):
                skipped = sift.run_gate(child, parent, initial, [], agent_kind="original", timeout_s=3600)
            self.assertTrue(skipped)
            saved = json.loads((child / "sift/input.json").read_text())
            self.assertEqual(saved["child_files"]["coding_agent.py"], "def broken(:\n")
            self.assertNotIn("future_score", json.dumps(saved))
            self.assertTrue(json.loads((child / "metadata.json").read_text())["sift"]["skipped"])

    def test_bad_patch_fails_open_without_touching_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "coding_agent_polyglot.py").write_text("pass\n")
            node = root / "node"
            node.mkdir()
            (node / "model_patch.diff").write_text("--- a/../../outside\n+++ b/../../outside\n@@ -0,0 +1 @@\n+bad\n")
            with patch.dict(os.environ, {"SIFT_MODE": "program", "SIFT_ENFORCE": "1"}):
                self.assertFalse(sift.run_gate(node, root / "parent", root, [], agent_kind="original", timeout_s=3600))
            self.assertEqual((root / "coding_agent_polyglot.py").read_text(), "pass\n")
            self.assertEqual(json.loads((node / "sift/decision.json").read_text())["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
