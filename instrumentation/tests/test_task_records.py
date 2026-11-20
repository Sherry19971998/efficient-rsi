"""Failure-path checks using synthetic containers and real temporary Git repositories."""
import ast
import datetime
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HGM = Path(__file__).resolve().parents[2] / "repos/HGM"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


T = module("task_records", HGM / "task_records.py")
sys.modules.setdefault("task_records", T)
L = module("lc_integration_records_test", HGM / "lc_integration.py")


def result(code=0, text=""):
    return types.SimpleNamespace(exit_code=code, output=text.encode())


class FakeContainer:
    def __init__(self, agent_code=0, grade_code=0, missing_patch=False, grade_text="2 passed in 0.1s"):
        self.agent_code, self.grade_code, self.grade_text = agent_code, grade_code, grade_text
        self.cleaned = False
        self.commands = []
        self.files = {"/hgm/python__x.md": "Tool Used: Bash\nTool Result: worked\n",
                      "/hgm/rsi_attempt.json": json.dumps({"scope": "attempt", "stop_reason": "normal", "turns": 2})}
        if not missing_patch:
            self.files["/hgm/model_patch.diff"] = "submitted answer\n"

    def start(self):
        pass

    def exec_run(self, cmd, **kwargs):
        self.commands.append(cmd)
        if isinstance(cmd, list) and cmd[:2] == ["timeout", "30"]:
            return result(0, "recovered answer\n")
        if isinstance(cmd, list) and cmd[:2] == ["timeout", "3600"]:
            return result(self.agent_code, "agent finished")
        if cmd == "timeout 120 ./eval.sh":
            return result(self.grade_code, self.grade_text)
        return result()


def copy_from(container, src, dest):
    if src not in container.files:
        raise FileNotFoundError(src)
    Path(dest).write_text(container.files[src])


class RecordsCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "node/predictions/node_0"
        self.out.mkdir(parents=True)
        self.entry = dict(instance_id="python__x", problem_statement="example", base_commit="HEAD",
                          test_commit="HEAD", language="python", files={"solution": ["solution.py"]})
        self.env = patch.dict(os.environ, {"HGM_RECOVER_ANSWERS": "0", "HGM_RECORD_FEEDBACK": "0",
                                          "HGM_AGENT": "", "HGM_TEST_PROTOCOL": "hidden", "RSI_LEDGER": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def card(self, kind="original"):
        return T.TaskRecord(self.out, self.entry, "node_0", "local-qwen", kind, 3600, [], HGM)

    def test_timeout_survives_collection_and_grading(self):
        c = self.card()
        c.phase("solve")
        c.agent_exit(result(124))
        c.collect(FakeContainer(missing_patch=True), copy_from)
        with self.assertRaises(RuntimeError):
            c.patch_for_grading()
        with patch.dict(os.environ, {"HGM_RECOVER_ANSWERS": "1"}):
            self.assertEqual(c.patch_for_grading(), "recovered answer\n")
        c.phase("grade")
        c.grade(result(0, "2 passed in 0.1s"))
        c.finish({"success": True, "eval_result": "resolved"})
        saved = T.read_json(c.path)
        self.assertEqual(saved["execution"]["stop_reason"], "hard_timeout")
        self.assertEqual(saved["grading"]["status"], "passed")
        self.assertEqual(saved["answer"]["origin"], "recovered_after_stop")
        self.assertIsNone(saved["cost"]["request_elapsed_s"])

    def test_retries_preserve_both_cards_and_answers(self):
        a, b = self.card(), self.card()
        for c, code in ((a, 124), (b, 0)):
            c.agent_exit(result(code))
            c.collect(FakeContainer(), copy_from)
            c.finish({"success": False, "eval_result": "incomplete"})
        self.assertNotEqual(a.id, b.id)
        self.assertTrue((a.dir / "submitted.diff").exists())
        self.assertTrue((b.dir / "submitted.diff").exists())
        self.assertEqual(len(T.latest_cards(self.root / "node")), 1)

    def test_grading_timeout_uses_exit_code_with_empty_output(self):
        c = self.card()
        c.grade(result(124))
        self.assertEqual(c.data["grading"]["status"], "timeout")
        self.assertEqual(T.failure_class(c.data), "test_timeout")

    def test_zero_tests_is_not_validated_success(self):
        c = self.card()
        c.grade(result(0, "no tests ran in 0.01s"))
        self.assertEqual(c.data["grading"]["status"], "invalid_no_tests")

    def test_unrecognized_summary_is_unknown(self):
        self.assertIsNone(T.test_counts("BUILD SUCCESSFUL")["tests_run"])
        self.assertEqual(T.test_counts("test result: ok. 3 passed; 0 failed; 2 ignored;"),
                         dict(tests_run=3, passed=3, failed=0, skipped=2))

    def test_collection_failure_preserves_initial_reason(self):
        c = self.card()
        c.agent_exit(result(124))
        bad = FakeContainer()
        bad.exec_run = lambda *a, **k: result(1)
        c.collect(bad, lambda *a: (_ for _ in ()).throw(OSError("unavailable")))
        c.error(RuntimeError("could not collect"))
        c.finish({"eval_result": "incomplete", "success": False})
        self.assertEqual(c.data["execution"]["stop_reason"], "hard_timeout")
        self.assertEqual(c.data["answer"]["collection_status"], "failed")

    def test_recovery_only_contains_allowed_paths_including_new_files(self):
        repo = self.root / "git"
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.DEVNULL)
        git("init", "-q")
        (repo / "solution.py").write_text("old\n")
        git("add", ".")
        git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
        (repo / "solution.py").write_text("new\n")
        (repo / "allowed.py").write_text("new allowed file\n")
        (repo / "secret_test.py").write_text("do not collect\n")
        p = subprocess.run([sys.executable, "-c", T.RECOVER_SCRIPT, str(repo), "HEAD",
                            json.dumps(["solution.py", "allowed.py"])], cwd=repo, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("allowed.py", p.stdout)
        self.assertIn("+new", p.stdout)
        self.assertNotIn("secret_test", p.stdout)

    def test_feedback_selects_failure_classes(self):
        cards = []
        for i, reason in enumerate(["normal", "hard_timeout", "agent_exception"]):
            c = self.card()
            c.data["task_id"] = str(i)
            c.data["execution"]["stop_reason"] = reason
            c.data["grading"]["status"] = "failed"
            cards.append(c.data)
        selected = T.representatives(cards, limit=3)
        self.assertEqual({T.failure_class(c) for c in selected}, {"test_failed", "hard_timeout", "agent_exception"})

    def test_local_cost_distinguishes_live_calls_from_replay(self):
        c = self.card("littlecoder")
        live = {"resp": {"in": 100, "out": 10, "cached": 20}, "t0": 1, "t1": 3}
        served = {**live, "served": True, "t0": 0, "t1": 0}
        (c.dir / "record.jsonl").write_text("\n".join(map(json.dumps, [live, served])) + "\n")
        (c.dir / "record.replay").write_text(json.dumps({"event": "live", "in": 100, "out": 10, "t0": 1, "t1": 3}) + "\n")
        c.finish({"eval_result": "resolved"})
        cost = c.data["cost"]
        self.assertEqual(cost["replayed_calls"], 1)
        self.assertEqual(cost["requests_completed"], 1)
        self.assertEqual(cost["prompt_tokens"], 100)
        self.assertEqual(cost["request_elapsed_s"], 2)
        self.assertIsNone(cost["requests_unfinished"])

    def test_proxy_cost_excludes_previous_attempt_late_response(self):
        ledger = self.root / "ledger.jsonl"
        ledger.write_text("")
        with patch.dict(os.environ, {"RSI_LEDGER": str(ledger), "RSI_RUN": "test-run"}):
            c = self.card()
            base = {"type": "llm_call", "tag": "eval", "run": "test-run", "node": "node_0",
                    "task": "python__x", "status": 200, "input_tokens": 100, "output_tokens": 10,
                    "cached_tokens": 0, "ts": c.start_epoch + 5}
            ledger.write_text("\n".join(map(json.dumps, [{**base, "latency_s": 10}, {**base, "latency_s": 2}])) + "\n")
            c.finish({"eval_result": "resolved"})
        self.assertEqual(c.data["cost"]["requests_completed"], 1)
        self.assertEqual(c.data["cost"]["prompt_tokens"], 100)

    def test_soft_stop_and_setup_error_remain_separate_from_wrong_answer(self):
        c = self.card("littlecoder")
        container = FakeContainer()
        container.files["/hgm/rsi_attempt.json"] = json.dumps({"scope": "attempt", "stop_reason": "soft_deadline", "max_turns": 100})
        c.agent_exit(result(0))
        c.collect(container, copy_from)
        self.assertEqual(T.failure_class(c.data), "soft_deadline")
        self.assertEqual(c.data["conditions"]["budget"]["max_turns"], 100)
        setup = self.card()
        setup.error(RuntimeError("container unavailable"))
        setup.finish({"eval_result": "incomplete"})
        self.assertEqual(T.failure_class(setup.data), "setup_error")
        self.assertEqual(setup.data["grading"]["status"], "not_run")

    def harness(self, container):
        class Phases:
            def __init__(self, *a, **k): pass
            def phase(self, *a, **k): pass
            def attach(self, *a, **k): pass
            def done(self, *a, **k): pass
        ns = dict(os=os, json=json, datetime=datetime, Path=Path, tempfile=tempfile, shutil=shutil,
                  task_records=T, lc_integration=L, llm="local-qwen", timeout=3600,
                  docker=types.SimpleNamespace(from_env=lambda: None),
                  rsi_ledger=types.SimpleNamespace(Phases=Phases, proxy_env=lambda *a: {}, event=lambda *a, **k: None),
                  setup_logger=lambda *a: types.SimpleNamespace(info=lambda *a: None, error=lambda *a: None),
                  make_test_spec=lambda e: types.SimpleNamespace(get_instance_container_name=lambda r: "test"),
                  remove_existing_container=lambda *a: None, build_container=lambda *a, **k: container,
                  cleanup_container=lambda *a: setattr(container, "cleaned", True), copy_from_container=copy_from,
                  copy_to_container=lambda *a: None, hide_answers=lambda: False,
                  get_test_description=lambda *a, **k: "", MAP_REPO_VERSION_TO_SPECS={"python": {"test_cmd": "pytest"}},
                  TEST_COMMANDS={"python": ["pytest"]}, safe_log=lambda *a: None)
        def log(r, raise_error=True):
            if raise_error and r.exit_code: raise RuntimeError(f"exit {r.exit_code}")
        ns["log_container_output"] = log
        tree = ast.parse((HGM / "polyglot/harness.py").read_text())
        wanted = {"process_entry", "process_entry_lc", "_score_in_container", "get_eval_script", "_publish_task_artifacts", "_finish_task_record"}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[]), "harness.py", "exec"), ns)
        return ns

    def test_both_harness_paths_normal_timeout_and_recovery(self):
        for kind in ("original", "littlecoder"):
            for code, recover in ((0, "0"), (124, "0"), (124, "1")):
                with self.subTest(kind=kind, code=code, recover=recover), patch.dict(os.environ, {"HGM_AGENT": kind, "HGM_RECOVER_ANSWERS": recover}):
                    container = FakeContainer(agent_code=code, missing_patch=bool(code))
                    ns = self.harness(container)
                    response = ns["process_entry"](self.entry, self.out, "node_0", [], skip_existing=False, init_agent_path=str(HGM))
                    saved = json.loads((self.out / "python__x.json").read_text())
                    card = T.load_card(saved, self.out)
                    self.assertIsNotNone(card)
                    self.assertTrue(container.cleaned)
                    self.assertEqual(saved["eval_result"], "resolved" if code == 0 or recover == "1" else "incomplete")
                    if code == 124:
                        self.assertEqual(card["execution"]["stop_reason"], "hard_timeout")
                    self.assertEqual(card["grading"]["status"], "passed" if code == 0 or recover == "1" else "not_run")


if __name__ == "__main__":
    unittest.main()
