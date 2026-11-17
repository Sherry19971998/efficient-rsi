"""harness-rsi smoke test. Run on the GPU host from repos/HGM, after `./run_all.sh vllm` and `proxy`.

    python scripts/lc_smoke.py checks            # A: integration functions on synthetic data (no GPU, no Docker)
    python scripts/lc_smoke.py coupled [N]       # B: root + two hand-made children on N search tasks

B writes output_hgm/lc-smoke/{initial,child_noop,child_knob}/ and prints, per evaluation, the
route (fresh / d0 / fork / fork0), the divergence call and the live model seconds.
Binary criteria (decide before running; record the outcome in HARNESS_RSI.md):
  - root: every evaluation route "fresh", a record written for every task
  - child_noop (patch changes only the _doc string of harness_knobs.json): route "d0" on >= 90% of
    tasks, 0 live model seconds on those, same eval_result as the root on all of them
  - child_knob (keep_tool_rounds: 3): route "fork" or "fork0" on the tasks with > 3 tool rounds,
    live model seconds below the root's on the same tasks
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HGM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HGM))
import lc_integration as L  # noqa: E402


def checks():
    ok = True

    def expect(name, cond):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + name)
        ok &= bool(cond)

    good = ("diff --git a/harness_knobs.json b/harness_knobs.json\n--- a/harness_knobs.json\n+++ b/harness_knobs.json\n"
            "@@ -1 +1 @@\n-  \"keep_tool_rounds\": null,\n+  \"keep_tool_rounds\": 3,\n")
    frozen = ("diff --git a/rsi_record.py b/rsi_record.py\n--- a/rsi_record.py\n+++ b/rsi_record.py\n@@ -1 +1 @@\n-a\n+b\n")
    hook = ("diff --git a/agent.py b/agent.py\n--- a/agent.py\n+++ b/agent.py\n@@ -1 +1 @@\n"
            "-        for event in rsi_record.wrapped_stream(\n+        for event in stream(\n")
    kept, dropped, v = L.filter_patch(good + frozen)
    expect("whitelisted file kept, frozen file dropped", "harness_knobs.json" in kept and dropped == ["rsi_record.py"] and v is None)
    expect("removing a recorder hook is a violation", L.filter_patch(hook)[2] is not None)
    expect("route d0", L.attempt_route({"rsi": {"diverged_at": None}}, True) == "d0")
    expect("route fork", L.attempt_route({"rsi": {"diverged_at": 4}}, True) == "fork")
    expect("route fresh", L.attempt_route({}, False) == "fresh")

    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "out"
        p = root / "initial" / "predictions" / "lc_agent_0"
        p.mkdir(parents=True)
        (root / "initial" / "metadata.json").write_text(json.dumps({"overall_performance": {"accuracy_score": 0.5}}))
        (p / "python__x.json").write_text(json.dumps({"instance_id": "python__x", "eval_result": "unresolved"}))
        (p / "python__x.md").write_text("Tool Used: Bash\n")
        (p / "python__x_eval.md").write_text("E   AssertionError: 1 != 2\nFAILED x_test.py::t\n")
        (p / "python__y.json").write_text(json.dumps({"instance_id": "python__y", "eval_result": "resolved"}))
        c = root / "child"
        c.mkdir()
        (c / "metadata.json").write_text(json.dumps({"parent_commit": "initial"}))
        rows = L.build_workspace(root, "initial", "python__x", Path(t) / "ws")
        ws = Path(t) / "ws"
        expect("workspace digest has 2 tasks", len(rows) == 2)
        expect("failure class wrong_answer", [r for r in rows if r["task"] == "python__x"][0]["failure_class"] == "wrong_answer")
        expect("failed trajectory copied", (ws / "trajectories" / "python__x.md").exists())
        expect("parent_dir of child is initial", L.parent_dir(c) == root / "initial")
        expect("workspace has notes and evidence", (ws / "notes.md").exists() and (ws / "lineage_evidence.md").exists())
        cp = c / "predictions" / "child_0"
        cp.mkdir(parents=True)
        (cp / "python__x.json").write_text(json.dumps({"instance_id": "python__x", "eval_result": "resolved", "lc_route": "fork"}))
        (cp / "python__y.json").write_text(json.dumps({"instance_id": "python__y", "eval_result": "resolved", "lc_route": "d0"}))
        ev = L.lineage_evidence(root, "child")
        expect("evidence row for the child, +0.50 on 2 shared tasks", "| child | initial |" in ev and "+0.50" in ev and "too few" in ev)

    n, m, se = L.paired({f"t{i}": 1 for i in range(10)}, {f"t{i}": int(i < 3) for i in range(10)})
    expect("paired: 10 tasks, +0.70, verdict better", n == 10 and abs(m - 0.7) < 1e-9 and L.verdict(n, m, se) == "better")
    n, m, se = L.paired({f"t{i}": i % 2 for i in range(10)}, {f"t{i}": (i + 1) % 2 for i in range(10)})
    expect("paired: balanced flips are unclear", L.verdict(n, m, se).startswith("unclear"))
    long = L.NOTES_TEMPLATE + "\n".join(f"- entry {i}: " + "x" * 200 for i in range(100))
    capped, cut = L.cap_notes(long)
    expect("notes capped at NOTES_CAP, newest entry kept", cut and len(capped) <= L.NOTES_CAP + 200 and "entry 99" in capped)
    print("checks:", "PASS" if ok else "FAIL")
    return ok


def coupled(n):
    if not (os.getenv("RSI_LEDGER") and (os.getenv("RSI_PROXY_URL") or os.getenv("RSI_PROXY_URL_CONTAINER"))):
        sys.exit("run this through ./run_all.sh smoke-lc (it sets RSI_LEDGER and the proxy address "
                 "the containers use to reach vLLM)")
    from polyglot.harness import harness   # needs Docker and the dataset; run on the GPU host
    os.environ["HGM_AGENT"] = "littlecoder"
    os.environ.setdefault("HGM_HIDE_ANSWERS", "1")
    os.environ.setdefault("LC_FIXED_DATE", "2026-11-16 Monday")
    import polyglot.harness as H
    H.llm = os.getenv("LC_SMOKE_MODEL", "vllm-qwen3c-local")   # run_all.sh passes the served model name
    tasks = json.load(open("polyglot/subsets/small.json"))[:n]
    out = Path("output_hgm/lc-smoke")
    if out.exists():
        shutil.rmtree(out)
    src = Path("initial_polyglot/lc-smoke/src")
    if not src.exists():
        from utils.docker_utils import copy_src_files
        copy_src_files(str(src))
    knobs = json.loads((src / "lc" / "harness_knobs.json").read_text())

    def make_patch(new_knobs):
        a = (src / "lc" / "harness_knobs.json").read_text().splitlines(keepends=True)
        b = (json.dumps(new_knobs, indent=2) + "\n").splitlines(keepends=True)
        import difflib
        body = "".join(difflib.unified_diff(a, b, "a/harness_knobs.json", "b/harness_knobs.json"))
        return "diff --git a/harness_knobs.json b/harness_knobs.json\n" + body

    noop = dict(knobs, _doc=knobs["_doc"] + " (smoke)")
    knob = dict(knobs, keep_tool_rounds=3)
    nodes = [("initial", None, None), ("child_noop", "initial", make_patch(noop)), ("child_knob", "initial", make_patch(knob))]
    for node, parent, patch in nodes:
        d = out / node
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps({"parent_commit": parent} if parent else {}))
        paths = []
        if patch:
            (d / "model_patch.diff").write_text(patch)
            paths = [str((d / "model_patch.diff").resolve())]
        os.environ["HGM_COUPLED"] = "1" if parent else "0"
        harness(test_task_list=tasks, max_workers=min(4, len(tasks)), model_name_or_path=node, model_patch_paths=paths,
                pred_dname=str(d / "predictions"), output_dir=str(d), init_agent_path=str(src))
        for t in tasks:
            r = json.loads((d / "predictions" / f"{node}_0" / f"{t}.json").read_text())
            a = d / "predictions" / f"{node}_0" / f"{t}_rsi.json"
            att = json.loads(a.read_text()) if a.exists() else {}
            print(f"{node:11s} {t:28s} {r['eval_result']:12s} route={r.get('lc_route')} "
                  f"diverged_at={(att.get('rsi') or {}).get('diverged_at')} live_s={(r.get('lc_cost') or {}).get('model_s')}")


if __name__ == "__main__":
    os.chdir(HGM)
    if sys.argv[1:2] == ["checks"]:
        sys.exit(0 if checks() else 1)
    if sys.argv[1:2] == ["coupled"]:
        coupled(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
        sys.exit(0)
    print(__doc__)
