"""harness-rsi integration (2026-11-16): little-coder as the self-improving agent of HGM, with
coupled evaluation, a file-based proposer workspace and the visible read-only test protocol.

Host-side code (listed in .dockerignore, so no agent can read or patch it). Everything is off
unless HGM_AGENT=littlecoder; the original arms of run_all.sh are unchanged.

Switches (environment):
  HGM_AGENT=littlecoder   evaluate and improve repos/HGM/lc/ (little-coder) instead of coding_agent.py
  HGM_COUPLED=1           coupled evaluation: a child evaluated on a task its parent has a record of
                          replays the parent's trajectory and calls the model only from the first
                          request that differs (the divergence point). Off: every evaluation runs
                          from scratch (record only), which is the comparison arm.
  HGM_COUPLED_TASKS=1     HGM's evaluation step prefers tasks the parent has a record of (hgm.py)
  HGM_TEST_PROTOCOL=visible_ro   the task's test files are in /testbed, read-only, hashed before and
                          after the agent; a changed hash scores the attempt unresolved (default)
                          | hidden  efficient-rsi's HGM_HIDE_ANSWERS behaviour only
  HGM_LC_DIAGNOSE=0       proposer instruction from the template below (default). 1: also run
                          efficient-rsi's LLM diagnosis and append its problem statement.
  LC_FIXED_DATE           passed to the agent; must be the same for a parent and its children

Coupled evaluation, per (node, task):
  parent has a record of the task  -> replay it, diverge to the live model (LC_ON_DIVERGE=live)
       no divergence      route "d0"    the child's harness sends exactly the parent's requests;
                                         no model call, the outcome is re-scored for real
       divergence at k>0  route "fork"  only the calls from k on use the GPU
       divergence at k=0  route "fork0" same cost as a fresh run
  no parent record                 -> route "fresh", record only
Every evaluation writes the node's own complete record, so coupling chains down the tree.

Lineage notes (notes.md), per node:
  - a proposer reads its parent's notes in /hgm/workspace/notes.md and may rewrite them; the result
    becomes the child's notes (<node>/notes.md). Notes are hypotheses written by models.
  - lineage_evidence.md, next to them in the workspace, is written by code and read-only to the
    proposer: for each change along the lineage, the child's results against its parent's on the
    tasks both did (paired), with a standard error and a verdict (better / worse / unclear).
  - correction: HGM's selection rarely expands a weak node, so its notes rarely spread; the
    evidence shows each hypothesis next to its measured outcome; above NOTES_CAP characters the
    old log is cut and the next proposer is asked to rewrite the summary.
"""
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import task_records

LC_DIR = "lc"                         # inside repos/HGM and inside every container (/hgm/lc)
# Characters of notes.md a child inherits (HGM_LC_NOTES_CAP). 24000 characters is about 6K tokens:
# ~10% of the proposer's 65,536-token context and under little-coder's 32,000-character tool output,
# room for a 4,000-character Summary plus ~40 Log entries of ~500 characters (a lineage of 30+ changes).
NOTES_CAP = int(os.getenv("HGM_LC_NOTES_CAP", "24000"))
SUMMARY_CAP = 4000
NOTES_TEMPLATE = """# Lineage notes

Written by the proposers of this lineage. These are hypotheses, not facts: check them against
lineage_evidence.md, which code writes from measured results.

## Summary
(Keep this section under 4000 characters. Rewrite it when the evidence changes what you believe.)

## Log
(One entry per change: what you changed, why, and which tasks or failure classes you expect to change.)
"""
CONTAINER_LC = "/hgm/lc"

# Files a self-improvement patch may change (relative to lc/). The proposer is told to prefer the
# knobs file. Everything else, in particular the recorder, the provider/metering layer, the entry
# point and the tool registry, is frozen: hunks touching it are dropped.
WHITELIST = [
    r"^harness_knobs\.json$",
    r"^ctx_policy\.py$",
    r"^context\.py$",
    r"^compaction\.py$",
    r"^agent\.py$",
    r"^tools\.py$",
    r"^local/[^/]+\.py$",
    r"^skill/.+\.(md|py)$",
]
FROZEN = ["hgm_agent.py", "rsi_record.py", "providers.py", "tool_registry.py", "config.py"]
# A patch may not remove these hooks (it would stop recording, metering or the context policy).
CONTRACT = ["rsi_record.wrapped_stream", "ctx_policy.apply(", "rsi_record.side_stream"]


def enabled():
    return os.getenv("HGM_AGENT", "") == "littlecoder"


def coupled():
    return os.getenv("HGM_COUPLED", "0") == "1"


def protocol():
    return os.getenv("HGM_TEST_PROTOCOL", "visible_ro")


def _safe(task):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", task)


# ---------------------------------------------------------------- records and coupling ----------

def node_dir_from_outdir(out_dname):
    """polyglot harness writes to <root>/<commit>/predictions/<commit>_0; the node dir is <root>/<commit>."""
    return Path(out_dname).parent.parent


def record_prefix(node_dir, task):
    return Path(node_dir) / "records" / _safe(task)


def has_record(node_dir, task):
    for result_path in (Path(node_dir) / "predictions").glob(f"*/{task}.json"):
        result = task_records.read_json(result_path) or {}
        card = task_records.load_card(result, result_path.parent)
        if card and card.get("execution", {}).get("exit_code") != 0:
            return False
    p = record_prefix(node_dir, task)
    return Path(str(p) + ".jsonl").exists() and Path(str(p) + ".req.jsonl.gz").exists()


def parent_dir(node_dir):
    """The parent's node dir (sibling of node_dir), or None for the root."""
    try:
        meta = json.loads((Path(node_dir) / "metadata.json").read_text())
    except Exception:
        return None
    parent = meta.get("parent_commit")
    if not parent or parent == "failed":
        return None
    d = Path(node_dir).parent / parent
    return d if d.exists() else None


def recorded_tasks(node_dir):
    d = Path(node_dir) / "records"
    if not d.exists():
        return set()
    return {f.name[:-len(".req.jsonl.gz")] for f in d.glob("*.req.jsonl.gz")}


def attempt_route(attempt, replayed):
    """Classify one evaluation from hgm_agent.py's rsi_attempt.json."""
    rsi = (attempt or {}).get("rsi") or {}
    if not replayed:
        return "fresh"
    k = rsi.get("diverged_at")
    if k is None:
        return "d0"
    return "fork0" if k == 0 else "fork"


def record_cost(prefix):
    """Model seconds and tokens of the live calls in a record (served calls have t0 = t1 = 0)."""
    s = tin = tout = cached = calls = served = 0
    p = Path(str(prefix) + ".jsonl")
    if p.exists():
        for line in p.read_text().splitlines():
            e = json.loads(line)
            if e.get("served"):
                served += 1
                continue
            r = e.get("resp", {})
            calls += 1
            s += e.get("t1", 0) - e.get("t0", 0)
            tin += r.get("in") or 0
            tout += r.get("out") or 0
            cached += r.get("cached") or 0
    return {"live_calls": calls, "served_calls": served, "model_s": round(s, 2), "in": tin, "out": tout,
            "cached": cached, "uncached_in_plus_out": tin - cached + tout}


# ---------------------------------------------------------------- test protocol -----------------

def test_files(entry):
    return list((entry.get("files") or {}).get("test") or [])


def test_file_contents(entry):
    """{relative path: bytes} of the task's tests at test_commit, read from the task repo on the host."""
    out = {}
    for rel in test_files(entry):
        r = subprocess.run(["git", "-C", str(entry["repo"]), "show", f"{entry['test_commit']}:{rel}"],
                           capture_output=True)
        if r.returncode == 0:
            out[rel] = r.stdout
    return out


def sha256_map(container, paths, workdir="/testbed"):
    """{path: sha256} computed inside the container (missing file -> None)."""
    out = {}
    for rel in paths:
        r = container.exec_run(["sh", "-c", f"sha256sum '{rel}' 2>/dev/null | cut -d' ' -f1"], workdir=workdir)
        h = r.output.decode().strip()
        out[rel] = h or None
    return out


# ---------------------------------------------------------------- lineage notes and evidence ----

def node_results(node_dir):
    """{task: 1 or 0} (resolved = 1, as HGM counts) and {task: lc_route} from a node's predictions."""
    res, routes = {}, {}
    for f in Path(node_dir).glob("predictions/*_0/*.json"):
        if f.name.endswith("_rsi.json"):
            continue
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        t = r.get("instance_id") or f.stem
        res[t] = int(r.get("eval_result") == "resolved")
        routes[t] = r.get("lc_route")
    return res, routes


def paired(child, parent):
    """Child minus parent on the tasks both did: (n, mean, standard error)."""
    shared = sorted(set(child) & set(parent))
    d = [child[t] - parent[t] for t in shared]
    n = len(d)
    if n == 0:
        return 0, 0.0, None
    mean = sum(d) / n
    if n < 2:
        return n, mean, None
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    return n, mean, (var / n) ** 0.5


def verdict(n, mean, se, min_n=5):
    if n < min_n:
        return f"too few shared tasks ({n})"
    if not se:
        return "better" if mean > 0 else ("worse" if mean < 0 else "no difference")
    if mean >= 2 * se:
        return "better"
    if mean <= -2 * se:
        return "worse"
    return "unclear (|difference| < 2 standard errors)"


def lineage_evidence(output_root, node):
    """Markdown: for each change from the root down to `node`, measured against its parent."""
    root = Path(output_root)
    chain, d = [], root / node
    while d is not None and d.exists():
        chain.append(d)
        d = parent_dir(d)
    chain.reverse()
    lines = ["# Lineage evidence", "", "Written by code from measured results; proposers cannot edit it.",
             "Each row compares a version with its parent on the tasks both were evaluated on (paired).",
             "difference = child solve rate minus parent solve rate on those tasks.", "",
             "| version | parent | files changed | shared tasks | difference | std. error | verdict | routes |",
             "|---|---|---|---|---|---|---|---|"]
    for i in range(1, len(chain)):
        c, p = chain[i], chain[i - 1]
        cr, croutes = node_results(c)
        pr, _ = node_results(p)
        n, mean, se = paired(cr, pr)
        patch = c / "model_patch.diff"
        files = ", ".join(_changed_files(patch.read_text() if patch.exists() else "")) or "-"
        rc = {}
        for r in croutes.values():
            rc[r] = rc.get(r, 0) + 1
        lines.append(f"| {c.name} | {p.name} | {files} | {n} | {mean:+.2f} | "
                     f"{'-' if se is None else f'{se:.2f}'} | {verdict(n, mean, se)} | "
                     + ", ".join(f"{k} {v}" for k, v in sorted(rc.items(), key=lambda x: str(x[0]))) + " |")
    if len(chain) <= 1:
        lines.append("| (root) | - | - | - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def read_notes(node_dir):
    p = Path(node_dir) / "notes.md"
    return p.read_text() if p.exists() else NOTES_TEMPLATE


def cap_notes(text, cap=NOTES_CAP):
    """Keep the summary section and the newest part of the log within `cap` characters."""
    if len(text) <= cap:
        return text, False
    marker = "\n[older log entries were removed by the harness: rewrite the Summary to keep what matters]\n"
    head, sep, log = text.partition("## Log")
    head = head[: min(SUMMARY_CAP + 600, cap // 3)]    # title + Summary (with some slack)
    room = max(cap - len(head) - len(sep) - len(marker), 0)
    return head + sep + marker + log[-room:], True


# ---------------------------------------------------------------- proposer workspace ------------

def _failure_category(eval_md):
    """A coarse failure class from the test run log (written by the harness, not by a model)."""
    t = eval_md or ""
    if not t.strip():
        return "no_test_output"
    if re.search(r"timed out|Terminated|exit code 124", t, re.I):
        return "test_timeout"
    if re.search(r"SyntaxError|IndentationError", t):
        return "syntax_error"
    if re.search(r"ImportError|ModuleNotFoundError|cannot find symbol|undefined reference|error\[E", t):
        return "build_or_import_error"
    if re.search(r"AssertionError|assert |FAILED|expected .* but|panicked", t):
        return "wrong_answer"
    return "other_failure"


def _changed_files(patch_text):
    return sorted(set(re.findall(r"^diff --git a/(\S+) b/", patch_text or "", re.M)))


def build_workspace(output_root, parent_commit, focus_task, dest, max_trajectories=8):
    """Write the proposer's workspace (CLM: context as files the proposer opens on demand).

    dest/README.md            what is here and what may be changed
    dest/digest.json, .md     the parent's results per task: outcome, failure class, turns, route,
                              model seconds (computed by code from the harness outputs)
    dest/trajectories/*.md    the parent's chat logs of failed tasks (focus task first)
    dest/tests/*.txt          the parent's test output of those tasks
    dest/patch_ledger.jsonl   every node so far: parent, files changed, accuracy, evaluations
    dest/ancestor_patches/    the patches that make up the parent, oldest first
    dest/notes.md             the parent's lineage notes (the proposer may rewrite them)
    dest/lineage_evidence.md  measured effect of each change along the lineage (code-written)
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "trajectories").mkdir(parents=True)
    (dest / "tests").mkdir()
    (dest / "ancestor_patches").mkdir()
    root = Path(output_root)
    pdir = root / parent_commit
    pred = pdir / "predictions" / f"{parent_commit}_0"
    if not pred.exists():   # the root was evaluated under initial_polyglot/<name> and copied to <root>/initial
        cands = sorted((pdir / "predictions").glob("*_0")) if (pdir / "predictions").exists() else []
        pred = cands[0] if cands else pred
    rows = []
    for res in sorted(pred.glob("*.json")) if pred.exists() else []:
        if res.name.endswith("_rsi.json"):
            continue
        try:
            r = json.loads(res.read_text())
        except Exception:
            continue
        tid = r.get("instance_id") or res.stem
        att = {}
        a = pred / f"{tid}_rsi.json"
        if a.exists():
            try:
                att = json.loads(a.read_text())
            except Exception:
                pass
        ev = pred / f"{tid}_eval.md"
        cat = "resolved" if r.get("eval_result") == "resolved" else (
            r.get("eval_result") if r.get("eval_result") in ("empty_patch", "incomplete", "error")
            else _failure_category(ev.read_text(errors="replace") if ev.exists() else ""))
        card = task_records.load_card(r, pred)
        if card and task_records.feedback_enabled():
            cat = task_records.failure_class(card)
        rows.append({"task": tid, "eval_result": r.get("eval_result"), "failure_class": cat,
                     "tests_modified": r.get("tests_modified", False), "turns": att.get("turns"),
                     "route": r.get("lc_route"), "agent_error": att.get("error"),
                     "task_record": r.get("task_record"),
                     **{k: v for k, v in record_cost(record_prefix(pdir, tid)).items()
                        if k in ("live_calls", "model_s", "uncached_in_plus_out")}})
    (dest / "digest.json").write_text(json.dumps(rows, indent=1))
    counts = {}
    for x in rows:
        counts[x["failure_class"]] = counts.get(x["failure_class"], 0) + 1
    md = [f"# Parent {parent_commit}: {sum(x['eval_result'] == 'resolved' for x in rows)}/{len(rows)} tasks resolved",
          "", "failure classes: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
          "| task | result | class | turns | model s |", "|---|---|---|---|---|"]
    md += [f"| {x['task']} | {x['eval_result']} | {x['failure_class']} | {x['turns']} | {x['model_s']} |" for x in rows]
    (dest / "digest.md").write_text("\n".join(md) + "\n")

    failed = [x["task"] for x in rows if x["eval_result"] != "resolved"]
    if task_records.feedback_enabled():
        summary, selected = task_records.feedback_summary(pdir, focus_task, max_trajectories)
        if summary:
            (dest / "task_records.md").write_text(summary)
            (dest / "selection.json").write_text(json.dumps({
                "policy": "failure_classes_v1", "provided_attempts": [c["attempt_id"] for c in selected]}, indent=2))
            priority = [c["task_id"] for c in selected]
            failed = priority + [tid for tid in failed if tid not in priority]
    if focus_task in failed:
        failed.remove(focus_task)
        failed.insert(0, focus_task)
    for tid in failed[:max_trajectories]:
        for src, sub, suffix in ((pred / f"{tid}.md", "trajectories", ".md"), (pred / f"{tid}_eval.md", "tests", ".txt")):
            if src.exists():
                shutil.copy2(src, dest / sub / (_safe(tid) + suffix))

    ledger = []
    for meta in sorted(root.glob("*/metadata.json")):
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        perf = m.get("overall_performance") or {}
        patch = meta.parent / "model_patch.diff"
        ledger.append({"node": meta.parent.name, "parent": m.get("parent_commit"),
                       "files_changed": _changed_files(patch.read_text() if patch.exists() else ""),
                       "resolved": perf.get("total_resolved_instances"),
                       "evaluated": perf.get("total_submitted_instances"),
                       "accuracy": perf.get("accuracy_score"),
                       "failed_proposal": m.get("error"),          # rejected or empty patches
                       "focus_task": m.get("entry")})
    (dest / "patch_ledger.jsonl").write_text("\n".join(json.dumps(x) for x in ledger) + "\n")

    chain, node = [], pdir
    while node is not None and (node / "model_patch.diff").exists():
        chain.append(node / "model_patch.diff")
        node = parent_dir(node)
    for i, p in enumerate(reversed(chain)):
        shutil.copy2(p, dest / "ancestor_patches" / f"{i:03d}_{p.parent.name}.diff")

    (dest / "notes.md").write_text(read_notes(pdir))
    (dest / "lineage_evidence.md").write_text(lineage_evidence(root, parent_commit))
    (dest / "README.md").write_text(WORKSPACE_README.format(parent=parent_commit, focus=focus_task, cap=NOTES_CAP))
    return rows


WORKSPACE_README = """# Proposer workspace

You are improving the coding-agent harness in /hgm/lc (little-coder). The harness is the code around
the model: what goes into the model's context, how tool results are shown, when the agent stops. The
model is fixed. Your change is kept only if the changed harness solves more tasks.

Files here (open what you need; do not try to read everything):
- digest.md / digest.json   results of the current version ({parent}) per task, with a failure class
                            computed from the test logs, turns used and model seconds
- task_records.md           when present, observed stop reasons, grading and missing evidence from
                            several representative tasks; ungraded is not the same as wrong
- trajectories/<task>.md    full agent logs of failed tasks; start with {focus}
- tests/<task>.txt          the test output of those tasks
- patch_ledger.jsonl        every earlier version: which files it changed and how it scored
- ancestor_patches/         the changes that already make up the current version, oldest first
- lineage_evidence.md       measured effect of every change in this lineage (written by code)
- notes.md                  notes the earlier proposers of this lineage kept. They are hypotheses:
                            compare them with lineage_evidence.md. You may rewrite this file; it is
                            handed to the next version. Keep it under {cap} characters; the
                            harness cuts the oldest Log entries beyond that.

What you may change, in this order of preference:
1. /hgm/lc/harness_knobs.json  (values only). Knobs: tool_result_max_chars, keep_tool_rounds,
   compress_test_output, max_turns.
2. Code in /hgm/lc: ctx_policy.py, context.py, compaction.py, agent.py, tools.py, local/*.py,
   skill/*. Changes to any other file are discarded. Do not remove the calls to rsi_record or
   ctx_policy.apply; a change that does is discarded.

Make one focused change that addresses a failure pattern you can see in more than one task. Before
finishing, check that the changed Python files still import (python -c "import agent").

Then update /hgm/workspace/notes.md:
- in Log, add one entry: what you changed, why, and a prediction that can be checked (which tasks or
  failure classes should change);
- in Summary, correct anything the evidence contradicts (a change marked "worse" in
  lineage_evidence.md means the idea behind it did not work here).
"""


def proposer_instruction(focus_task, diagnosis=None):
    text = ("Improve the coding-agent harness in /hgm/lc. Read /hgm/workspace/README.md first; it "
            "explains the files in /hgm/workspace and which files you may change. Look at the failed "
            f"task {focus_task} and at the failure classes in /hgm/workspace/digest.md, then make one "
            "focused change to /hgm/lc. Finally record the change and your prediction in "
            "/hgm/workspace/notes.md, after checking /hgm/workspace/lineage_evidence.md.")
    if diagnosis:
        text += "\n\nA diagnosis of the focus task, for reference:\n" + diagnosis
    return text


# ---------------------------------------------------------------- patch filter ------------------

def filter_patch(patch_text):
    """Keep only whitelisted files; reject a patch that removes a contract hook.

    Returns (kept_patch, dropped_files, violation or None)."""
    blocks = re.split(r"(?=^diff --git )", patch_text or "", flags=re.M)
    kept, dropped = [], []
    for b in blocks:
        m = re.match(r"diff --git a/(\S+) b/(\S+)", b)
        if not m:
            continue
        path = m.group(2)
        if path in FROZEN or not any(re.match(w, path) for w in WHITELIST):
            dropped.append(path)
            continue
        kept.append(b)
    out = "".join(kept)
    for tok in CONTRACT:
        removed = sum(l.count(tok) for l in out.splitlines() if l.startswith("-") and not l.startswith("---"))
        added = sum(l.count(tok) for l in out.splitlines() if l.startswith("+") and not l.startswith("+++"))
        if removed > added:
            return out, dropped, f"patch removes {tok}"
    return out, dropped, None


# ---------------------------------------------------------------- ledger -----------------------

def ledger_event(rsi_ledger, node, task, route, attempt, cost, tests_modified, eval_result):
    rsi = (attempt or {}).get("rsi") or {}
    rsi_ledger.event("lc_eval", node=node, task=task, route=route, eval_result=eval_result,
                     diverged_at=rsi.get("diverged_at"), recorded_calls=rsi.get("recorded_calls"),
                     turns=(attempt or {}).get("turns"), agent_error=(attempt or {}).get("error"),
                     guard=(attempt or {}).get("guard"), tests_modified=tests_modified, **cost)
