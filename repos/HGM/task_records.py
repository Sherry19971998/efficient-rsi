"""Host-owned, per-attempt evidence for Polyglot evaluations (no model calls).

Cards are additive. HGM_RECOVER_ANSWERS=1 opts into grading recovered work;
HGM_RECORD_FEEDBACK=1 opts into using the cards for proposal selection/diagnosis.
Unknown measurements stay null. The legacy score is retained independently.
"""
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def feedback_enabled():
    return os.getenv("HGM_RECORD_FEEDBACK", "0") == "1"


def recovery_enabled():
    return os.getenv("HGM_RECOVER_ANSWERS", "0") == "1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def source_fingerprint(root, kind):
    """Hash the supplied initial agent sources, including uncommitted local changes."""
    root = Path(root)
    dirs = ("lc",) if kind == "littlecoder" else ("tools", "utils", "prompts")
    files = [p for p in root.glob("*") if p.is_file() and p.suffix in (".py", ".json", ".txt", ".md")]
    for name in dirs:
        files.extend(p for p in (root / name).rglob("*") if p.is_file()
                     and not p.is_symlink() and "__pycache__" not in p.parts and ".git" not in p.parts)
    h = hashlib.sha256()
    try:
        for p in sorted(set(files)):
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
    except OSError:
        return None
    return h.hexdigest() if files else None


def test_counts(text):
    """Parse explicit summaries only; unknown is not zero. Never count assertion lines."""
    counts = {k: None for k in ("tests_run", "passed", "failed", "skipped")}
    # pytest's final summary, including the explicit no-tests case.
    lines = [s for s in text.splitlines() if re.search(r"\bin \d+(?:\.\d+)?s\b", s)]
    if lines and re.search(r"\b(?:passed|failed|skipped|no tests ran)\b", lines[-1]):
        line = lines[-1]
        if "no tests ran" in line:
            return dict(tests_run=0, passed=0, failed=0, skipped=0)
        if re.search(r"\b\d+ errors?\b", line):
            return counts  # collection/setup errors are not executed tests
        found = {name: int(n) for n, name in re.findall(r"(\d+) (passed|failed|skipped|xfailed|xpassed)", line)}
        return dict(tests_run=sum(found.get(k, 0) for k in ("passed", "failed", "xpassed")),
                    passed=found.get("passed", 0), failed=found.get("failed", 0),
                    skipped=found.get("skipped", 0) + found.get("xfailed", 0))
    # Rust prints one summary per test binary, including a separate doctest summary.
    rust = re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored;", text)
    if rust:
        p, f, s = (sum(int(row[i]) for row in rust) for i in range(3))
        return dict(tests_run=p + f, passed=p, failed=f, skipped=s)
    # Gradle's failure summary; do not infer counts from BUILD SUCCESSFUL alone.
    java = re.findall(r"(?m)^\s*(\d+) tests completed(?:, (\d+) failed)?(?:, (\d+) skipped)?\s*$", text)
    if java:
        n, f, s = (int(v or 0) for v in java[-1])
        return dict(tests_run=n - s, passed=n - f - s, failed=f, skipped=s)
    # Jest explicitly identifies the Tests line (not Test Suites).
    jest = re.search(r"(?m)^Tests:\s+(.+)$", text)
    if jest:
        found = {name: int(n) for n, name in re.findall(r"(\d+) (passed|failed|skipped|todo|total)", jest[1])}
        if "total" in found:
            s = found.get("skipped", 0) + found.get("todo", 0)
            return dict(tests_run=found["total"] - s, passed=found.get("passed", 0),
                        failed=found.get("failed", 0), skipped=s)
    return counts


# Read only the benchmark-declared solution paths, including new allowed files.
# Passing paths as argv (and using literal pathspecs) avoids shell interpretation.
RECOVER_SCRIPT = r'''
import json, subprocess, sys
repo, base, paths = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
def git(*args):
    return subprocess.run(['git', '--literal-pathspecs', '-C', repo, *args],
                          capture_output=True, timeout=20)
r = git('diff', '--binary', '--no-ext-diff', '--no-textconv', base, '--', *paths)
if r.returncode:
    sys.stderr.buffer.write(r.stderr); sys.exit(r.returncode)
out = r.stdout
r = git('ls-files', '--others', '--exclude-standard', '-z', '--', *paths)
if r.returncode:
    sys.stderr.buffer.write(r.stderr); sys.exit(r.returncode)
for name in r.stdout.decode().split('\0'):
    if name:
        r = git('diff', '--no-index', '--binary', '--no-ext-diff', '--no-textconv', '--', '/dev/null', name)
        if r.returncode not in (0, 1):
            sys.stderr.buffer.write(r.stderr); sys.exit(r.returncode)
        out += r.stdout
sys.stdout.buffer.write(out)
'''


class TaskRecord:
    def __init__(self, out_dir, entry, node, model, agent_kind, timeout_s, patches, init_path):
        self.out_dir = Path(out_dir)
        self.entry = entry
        self.task = entry["instance_id"]
        self.id = uuid.uuid4().hex
        safe_task = re.sub(r"[^A-Za-z0-9_.-]", "_", self.task)
        self.dir = self.out_dir / "task_records" / safe_task / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.path = self.dir / "card.json"
        self.phase_start = time.monotonic()
        self.current_phase = "setup"
        self.start_epoch = time.time()
        self.collected = False
        self.submitted_patch = None
        self.recovered_patch = None
        self.ledger = os.getenv("RSI_LEDGER")
        try:
            self.ledger_offset = Path(self.ledger).stat().st_size if self.ledger else 0
        except OSError:
            self.ledger_offset = 0
        parent = read_json(self.out_dir.parent.parent / "metadata.json") or {}
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                               stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            revision = None
        patch_hashes = []
        for p in patches or []:
            try:
                patch_hashes.append(hashlib.sha256(Path(p).read_bytes()).hexdigest())
            except OSError:
                patch_hashes.append(None)
        self.data = {
            "schema_version": 1, "attempt_id": self.id, "run_id": os.getenv("RSI_RUN"),
            "node_id": node, "parent_id": parent.get("parent_commit"), "task_id": self.task,
            "started_at": utc_now(), "finished_at": None,
            "conditions": {"model": model, "agent_kind": agent_kind, "code_revision": revision,
                           "initial_source_sha256": source_fingerprint(init_path, agent_kind),
                           "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "language": entry.get("language"), "benchmark_base_commit": entry.get("base_commit"),
                           "benchmark_test_commit": entry.get("test_commit"),
                           "initial_agent_path": str(init_path), "lineage_patch_sha256": patch_hashes,
                           "test_protocol": (os.getenv("HGM_TEST_PROTOCOL", "visible_ro") if agent_kind == "littlecoder"
                                             else "hidden"),
                           "budget": {"agent_wall_s": timeout_s, "grading_wall_s": 120},
                           "recover_answers": recovery_enabled(), "record_feedback": feedback_enabled()},
            "execution": {"status": "pending", "stop_reason": "unknown", "exit_code": None,
                          "stage": "setup", "error_type": None, "error_excerpt": None},
            "answer": {"collection_status": "not_attempted", "origin": None, "patch_path": None,
                       "patch_sha256": None},
            "grading": {"status": "not_run", "exit_code": None, **test_counts("")},
            "cost": {"wall_s_by_stage": {}, "requests_started": None, "requests_completed": None,
                     "requests_failed": None, "requests_unfinished": None, "prompt_tokens": None,
                     "output_tokens": None, "cached_tokens": None, "request_elapsed_s": None,
                     "replayed_calls": None, "local_record_live_calls": None,
                     "coverage": "unknown", "ledger_path": self.ledger},
            "evidence": {}, "data_quality": {"collection_errors": [], "missing_fields": []},
            "legacy_result": None,
        }
        self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def phase(self, name):
        now = time.monotonic()
        times = self.data["cost"]["wall_s_by_stage"]
        times[self.current_phase] = round(times.get(self.current_phase, 0) + now - self.phase_start, 3)
        self.current_phase, self.phase_start = name, now
        self.data["current_stage"] = name
        if self.data["execution"]["status"] == "pending":
            self.data["execution"]["stage"] = name
        self.save()

    def agent_exit(self, result):
        code = result.exit_code
        e = self.data["execution"]
        e.update(status="completed" if code == 0 else "interrupted", exit_code=code,
                 stop_reason="hard_timeout" if code == 124 else ("unknown" if code == 0 else "agent_exception"),
                 stage="solve")
        output = result.output.decode(errors="replace") if isinstance(result.output, bytes) else str(result.output)
        (self.dir / "agent_stdout.log").write_text(output)
        self.data["evidence"]["agent_stdout_path"] = str(self.dir / "agent_stdout.log")
        if code != 0:
            e["error_excerpt"] = output[-2000:]
        self.save()

    def error(self, exc):
        e = self.data["execution"]
        # Preserve the primary stop reason when artifact collection or grading fails later.
        if self.current_phase == "grade":
            self.data["grading"].update(status="error", error_type=type(exc).__name__, error_excerpt=str(exc)[:2000])
        elif e["stop_reason"] == "unknown":
            e.update(status="error", error_type=type(exc).__name__, error_excerpt=str(exc)[:2000],
                     stage=self.current_phase)
        self.save()

    def _copy(self, container, src, name, copier, required=False):
        dest = self.dir / name
        try:
            copier(container, src, dest)
            return dest
        except Exception as exc:
            if required:
                self.data["data_quality"]["collection_errors"].append({"file": src, "error": str(exc)[:500]})
            return None

    def collect(self, container, copier):
        """Called before grading changes the worktree and before container cleanup."""
        if self.collected or container is None:
            return
        self.collected = True
        self.phase("collect")
        # A dedicated disposable task container: stop remaining writers before taking evidence.
        try:
            stopped = container.exec_run("sh -c 'kill -9 -1 || true'", workdir="/")
            quiesced = stopped.exit_code == 0
        except Exception as exc:
            quiesced = False
            self.data["data_quality"]["collection_errors"].append({"operation": "stop_writers", "error": str(exc)[:500]})
        self.data["answer"]["writers_stopped"] = quiesced
        trace = self._copy(container, f"/hgm/{self.task}.md", "trajectory.md", copier, required=True)
        if trace:
            self.data["evidence"]["trace_path"] = str(trace)
            lines = trace.read_text(errors="replace").splitlines()
            self.data["evidence"]["last_events"] = [line[:500] for line in lines[-12:]]
        # Evolved original agents may produce several trajectory files in one attempt.
        try:
            found = container.exec_run(["find", "/hgm", "-maxdepth", "1", "-name", f"{self.task}_*.md"])
            extras = []
            if found.exit_code == 0:
                for src in found.output.decode(errors="replace").splitlines():
                    if PurePosixPath(src).parent == PurePosixPath("/hgm"):
                        path = self._copy(container, src, Path(src).name, copier, required=True)
                        if path:
                            extras.append(str(path))
            self.data["evidence"]["additional_trace_paths"] = extras
        except Exception as exc:
            self.data["data_quality"]["collection_errors"].append({"operation": "additional_traces", "error": str(exc)[:500]})
        submitted = self._copy(container, "/hgm/model_patch.diff", "submitted.diff", copier)
        if submitted and quiesced:
            self.submitted_patch = submitted.read_text(errors="replace")
        attempt = self._copy(container, "/hgm/rsi_attempt.json", "agent_attempt.json", copier)
        if attempt:
            a = read_json(attempt) or {}
            self.data["evidence"]["agent_report"] = a
            self.data["conditions"]["budget"]["max_turns"] = a.get("max_turns")
            e = self.data["execution"]
            if (e["exit_code"] == 0 and a.get("scope", "attempt") == "attempt" and
                a.get("stop_reason") in {"normal", "soft_deadline", "turn_limit", "agent_exception", "request_error", "replay_stop"}):
                e["stop_reason"] = a["stop_reason"]
                e["reason_source"] = "agent_report"
                if a.get("error"):
                    e.update(status="error", error_excerpt=str(a["error"])[:2000])
        if self.data["conditions"]["agent_kind"] == "littlecoder":
            for suffix in (".jsonl", ".req.jsonl.gz", ".replay", ".replay.req.jsonl.gz", ".side.jsonl"):
                self._copy(container, "/hgm/record/rec" + suffix, "record" + suffix, copier)
        paths = self.entry.get("files", {}).get("solution", [])
        safe_paths = [p for p in paths if p and not PurePosixPath(p).is_absolute() and ".." not in PurePosixPath(p).parts]
        if quiesced and safe_paths and len(safe_paths) == len(paths):
            try:
                r = container.exec_run(["timeout", "30", "python", "-c", RECOVER_SCRIPT, "/testbed",
                                        self.entry["base_commit"], json.dumps(safe_paths)], workdir="/testbed")
                if r.exit_code != 0:
                    raise RuntimeError(f"solution snapshot exited {r.exit_code}")
                self.recovered_patch = r.output.decode(errors="replace")
                (self.dir / "workspace.diff").write_text(self.recovered_patch)
            except Exception as exc:
                self.data["data_quality"]["collection_errors"].append({"operation": "snapshot_answer", "error": str(exc)[:500]})
        patch = self.submitted_patch if self.submitted_patch is not None else self.recovered_patch
        origin = "submitted" if self.submitted_patch is not None else "recovered_after_stop"
        if patch is not None:
            path = self.dir / ("submitted.diff" if origin == "submitted" else "workspace.diff")
            self.data["answer"].update(collection_status="saved" if patch else "empty", origin=origin,
                                       patch_path=str(path), patch_sha256=hashlib.sha256(patch.encode()).hexdigest())
        else:
            self.data["answer"]["collection_status"] = "failed"
        self.save()

    def patch_for_grading(self):
        bad_exit = self.data["execution"]["exit_code"] != 0
        if bad_exit and not recovery_enabled():
            raise RuntimeError(f"Agent exited {self.data['execution']['exit_code']}; recovered work retained, grading disabled")
        if not bad_exit and self.submitted_patch is not None:
            return self.submitted_patch
        if recovery_enabled() and self.recovered_patch is not None:
            self.data["answer"].update(origin="recovered_after_stop", patch_path=str(self.dir / "workspace.diff"),
                                       patch_sha256=hashlib.sha256(self.recovered_patch.encode()).hexdigest(),
                                       collection_status="saved" if self.recovered_patch else "empty")
            return self.recovered_patch
        raise RuntimeError("No safely collected answer available for grading")

    def grade(self, result, tests_modified=False):
        text = result.output.decode(errors="replace")
        path = self.dir / "grading.log"
        path.write_text(text)
        code = result.exit_code
        g = self.data["grading"]
        g.update(status="timeout" if code == 124 else ("passed" if code == 0 else "failed"),
                 exit_code=code, log_path=str(path), test_integrity_ok=not tests_modified, **test_counts(text))
        if tests_modified:
            g["status"] = "invalid_tests_modified"
        elif code == 0 and g["tests_run"] == 0:
            g["status"] = "invalid_no_tests"
        self.save()

    def _local_cost(self):
        """Per-attempt little-coder records; served parent responses consume no new model call.

        Only completed recorded calls are observable here. A killed request is still unknown.
        The replay log duplicates main live records, so use it only for side calls/replay counts.
        """
        c, live, replayed = self.data["cost"], [], None
        for name in ("record.jsonl", "record.side.jsonl", "record.replay"):
            path = self.dir / name
            if not path.exists():
                continue
            if name == "record.jsonl":
                replayed = 0
            for line in path.read_text(errors="replace").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    self.data["data_quality"]["collection_errors"].append({"file": name, "error": "partial or invalid record line"})
                    continue
                if name == "record.jsonl":
                    if e.get("served"):
                        replayed += 1
                    elif "resp" in e:
                        live.append({**e["resp"], "t0": e.get("t0"), "t1": e.get("t1")})
                elif e.get("event") == "live" and (name.endswith(".side.jsonl") or e.get("purpose")):
                    live.append(e)
                elif name == "record.replay" and e.get("event") == "side_served":
                    replayed = (replayed or 0) + 1
        c["replayed_calls"] = replayed
        if not live:
            return
        c.update(local_record_live_calls=len(live), requests_completed=len(live),
                 coverage="completed_local_records_only; failed, in-flight and unrecorded requests unknown")
        if all(isinstance(e.get(k), (int, float)) for e in live for k in ("t0", "t1")):
            c["request_elapsed_s"] = round(sum(e["t1"] - e["t0"] for e in live), 3)
        for dest, key in (("prompt_tokens", "in"), ("output_tokens", "out"), ("cached_tokens", "cached")):
            if all(e.get(key) is not None for e in live):
                c[dest] = sum(e[key] for e in live)

    def _cost(self):
        self._local_cost()
        # Read only the appended tail, not the full multi-GB shared ledger for every task.
        if not self.ledger:
            return
        events = []
        try:
            with open(self.ledger, "rb") as f:
                f.seek(self.ledger_offset)
                for line in f:
                    try:
                        e = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if (e.get("type") == "llm_call" and e.get("tag") == "eval" and
                        e.get("node") == self.data["node_id"] and e.get("task") == self.task and
                        e.get("run", "") == (self.data["run_id"] or "") and
                        e.get("ts", 0) - (e.get("latency_s") or 0) >= self.start_epoch):
                        events.append(e)
        except OSError:
            return
        if not events:
            return  # missing/unfinished calls are not a measured zero
        c = self.data["cost"]
        c.update(requests_completed=sum(e.get("status") == 200 for e in events),
                 requests_failed=sum(e.get("status") != 200 for e in events),
                 request_elapsed_s=round(sum(e.get("latency_s", 0) for e in events), 3),
                 coverage="completed_proxy_records_in_run_node_task_time_window; in-flight and unproxied requests unknown")
        for dest, key in (("prompt_tokens", "input_tokens"), ("output_tokens", "output_tokens"), ("cached_tokens", "cached_tokens")):
            values = [e.get(key) for e in events if e.get("status") == 200]
            c[dest] = sum(values) if values and all(v is not None for v in values) else None

    def finish(self, result):
        self.phase("finished")
        self.data["finished_at"] = utc_now()
        self.data["legacy_result"] = {k: result.get(k) for k in ("eval_result", "success")}
        self._cost()
        q = self.data["data_quality"]
        q["missing_fields"] = ["cost." + k for k, v in self.data["cost"].items() if v is None]
        if self.data["execution"]["stop_reason"] == "unknown":
            q["missing_fields"].append("execution.stop_reason")
        if self.data["grading"]["status"] != "not_run" and self.data["grading"]["tests_run"] is None:
            q["missing_fields"].append("grading.tests_run")
        q["record_complete"] = not q["missing_fields"] and not q["collection_errors"]
        self.save()
        result["task_record"] = str(self.path.relative_to(self.out_dir))
        result["attempt_id"] = self.id
        atomic_json(self.dir / "legacy_result.json", result)
        # Existing consumers keep their per-task result; each attempt's evidence is immutable.
        atomic_json(self.out_dir / f"{self.task}.json", result)


def load_card(result, prediction_dir):
    ref = result.get("task_record")
    if not ref:
        return None
    path = Path(ref)
    return read_json(path if path.is_absolute() else Path(prediction_dir) / path)


def failure_class(card):
    e, g = card.get("execution", {}), card.get("grading", {})
    reason = e.get("stop_reason")
    if reason in {"hard_timeout", "soft_deadline", "turn_limit", "agent_exception", "request_error"}:
        return reason
    if g.get("status") == "passed":
        return "resolved"
    if g.get("status") == "timeout":
        return "test_timeout"
    if g.get("status", "").startswith("invalid"):
        return g["status"]
    if g.get("status") == "failed":
        return "test_failed"
    if e.get("stage") == "setup" and e.get("status") == "error":
        return "setup_error"
    if card.get("answer", {}).get("collection_status") == "empty":
        return "empty_patch"
    return "unknown_failure"


def latest_cards(node_dir):
    by_task = {}
    for path in sorted(Path(node_dir).glob("predictions/*/task_records/*/*/card.json")):
        card = read_json(path)
        if not card or not card.get("finished_at"):
            continue
        task = card["task_id"]
        if task not in by_task or card["started_at"] > by_task[task]["started_at"]:
            card["card_path"] = str(path)
            by_task[task] = card
    return list(by_task.values())


def representatives(cards, focus=None, limit=4):
    groups = defaultdict(list)
    for card in cards:
        group = failure_class(card)
        if group != "resolved":
            groups[group].append(card)
    ordered = []
    for key in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        ordered.extend(sorted(groups[key], key=lambda c: c["task_id"])[:1])
    if focus:
        first = next((c for c in cards if c["task_id"] == focus), None)
        if first:
            ordered = [first] + [c for c in ordered if c["task_id"] != focus]
    return ordered[:limit]


def feedback_summary(node_dir, focus=None, limit=4):
    cards = latest_cards(node_dir)
    if not cards:
        return "", []
    selected = representatives(cards, focus, limit)
    counts = Counter(failure_class(c) for c in cards)
    lines = ["Observed task records (latest attempt per task; ungraded is not a wrong answer):",
             json.dumps(dict(counts), sort_keys=True),
             "These are observations, not proven root causes. Do not assume missing tests failed."]
    for c in selected:
        compact = {"task": c["task_id"], "attempt_id": c["attempt_id"],
                   "execution": c["execution"], "grading": c["grading"], "answer": c["answer"],
                   "last_events": c.get("evidence", {}).get("last_events", [])[-4:],
                   "missing_fields": c.get("data_quality", {}).get("missing_fields", [])}
        lines.append(json.dumps(compact, ensure_ascii=False)[:3500])
    return "\n".join(lines), selected
