"""Offline replay of stopping policies on finished searches, without any GPU.

The ledger holds every search-task result each node got, in the order it arrived, and the
time each evaluation took. The replay walks those results in order and asks a policy,
before each one, whether the node is still worth evaluating. A retired node's later
results are dropped and their evaluation time is counted as saved. At the end the final
agent is selected from what was kept, with the same rule as final_eval.py (Beta lower
bound, eps 0.05), at the same equal-GPU-hour cut.

What the replay cannot show: the real search would have spent the saved time on other
nodes, and in HGM the dropped results also feed parent selection (descendant evaluations),
so later children could differ. The savings are the room a policy frees; whether the search
uses it well is for an online run.

    python replay.py --ledger ../runs/ledger.jsonl --output-root ../repos/HGM/output_hgm \
        --runs hgm-fixed-r0 dgm-r0 --max-hours 9.646
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "instrumentation"))

import analyze_ledger  # noqa: E402
from results import select_best  # noqa: E402

from policy import FixedCap, Policy, UcbVsIncumbent, beta_quantile  # noqa: E402

KEEP_TYPES = {"eval", "eval_result", "node_created", "budget_stop"}
TS = re.compile(r'"ts": ([0-9.]+)')
RUN = re.compile(r'"run": "([^"]*)"')
NODE = re.compile(r'"node": "([^"]*)"')


def load(path, runs):
    """run -> events in ledger order. Model calls are kept as bare ticks (ts and node only):
    the GPU-hour cut needs their timestamps, the replay needs nothing else from them."""
    wanted = set(runs)
    out = defaultdict(list)
    with open(path) as f:
        for line in f:
            if '"llm_call"' in line:
                m = RUN.search(line)
                if m and m.group(1) in wanted:
                    n = NODE.search(line)
                    out[m.group(1)].append({"type": "tick", "ts": float(TS.search(line).group(1)),
                                            "node": n.group(1) if n else None})
                continue
            if '"gpu_sample"' in line:
                continue
            e = json.loads(line)
            if e.get("run") in wanted:
                out[e["run"]].append(e if e.get("type") in KEEP_TYPES
                                     else {"type": "tick", "ts": e["ts"], "node": e.get("node")})
    return out


def canonical_map(events):
    """HGM names an evaluation "<commit>_<i>"; map it back to the node "<commit>"."""
    known = {e.get("node") for e in events if e["type"] in ("node_created", "eval_result")}

    def canon(i):
        if i and i not in known:
            m = re.fullmatch(r"(.+)_\d+", i)
            if m and m.group(1) in known:
                return m.group(1)
        return i
    return canon


def result_stream(events):
    """[(node, results, eval seconds, task, ts)] in arrival order: one entry per eval_result event
    (task is None for a batch result).

    An eval_result's cost is the wall time of the evaluation phases (setup, agent run, test)
    of that node logged since its previous result; HGM's one-task results are matched on
    the task, DGM's batch results take everything pending for the node.
    """
    canon = canonical_map(events)
    pending = defaultdict(lambda: defaultdict(float))  # node -> task -> seconds
    stream = []
    for e in sorted(events, key=lambda e: e["ts"]):
        if e["type"] == "eval":
            pending[canon(e.get("node"))][e.get("task")] += e.get("wall_s") or 0.0
        elif e["type"] == "eval_result" and e.get("node"):
            node, task = e["node"], e.get("task")
            if task is not None:
                cost = pending[node].pop(task, 0.0)
            else:
                cost = sum(pending[node].values())
                pending[node].clear()
            stream.append((node, [int(x) for x in (e.get("score") or [])], cost, task, e["ts"]))
    return stream


def replay(stream, policy, eps=0.05):
    kept = defaultdict(list)
    retired = set()
    lcb = {}
    stats = {"results": 0, "skipped": 0, "eval_s": 0.0, "saved_s": 0.0}
    for node, res, cost, *_ in stream:
        stats["eval_s"] += cost
        incumbent = max(lcb.values(), default=0.0)
        if node in retired or not policy.evaluate(node, kept[node], incumbent):
            retired.add(node)
            stats["skipped"] += len(res)
            stats["saved_s"] += cost
            continue
        kept[node].extend(res)
        lcb[node] = beta_quantile(eps, sum(kept[node]), len(kept[node]))
        stats["results"] += len(res)
    stats["best"] = select_best(dict(kept), eps=eps)
    stats["retired"] = len(retired)
    return stats, kept


def shuffled(stream, rng):
    """The same stream with each node's results dealt to its own slots in a random order.

    Arrival times (which node is evaluated when) stay as they were; only which result a node
    gets first changes. A node's full record at the cut, and so the selection without
    stopping, is unchanged, so any change in selection comes from the policy alone.
    """
    slots = defaultdict(list)
    for i, (node, *_) in enumerate(stream):
        slots[node].append(i)
    out = list(stream)
    for node, idx in slots.items():
        for i, j in zip(idx, rng.permutation(idx)):
            out[i] = stream[j]
    return out


def robustness(run, stream, policies, n, seed=0):
    """Share of n shuffled orders in which each policy changes the selection, and its mean saving."""
    rng = np.random.default_rng(seed)
    full, scores = replay(stream, Policy())
    mean = {k: np.mean(v) for k, v in scores.items() if v}
    changed = defaultdict(int)
    saved = defaultdict(float)
    loss = defaultdict(float)
    for _ in range(n):
        s_stream = shuffled(stream, rng)
        for p in policies:
            s, _ = replay(s_stream, p)
            changed[p.name] += s["best"] != full["best"]
            saved[p.name] += s["saved_s"] / full["eval_s"] if full["eval_s"] else 0.0
            loss[p.name] += mean[full["best"]] - mean[s["best"]]
    return {p.name: (changed[p.name] / n, saved[p.name] / n, loss[p.name] / n) for p in policies}


def final_best(output_root, run):
    path = os.path.join(output_root, run, "final_eval", "final_result.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["best_node"]


def default_policies():
    return ([Policy(), FixedCap(10), FixedCap(20)]
            + [UcbVsIncumbent(a, m) for a in (0.2, 0.1, 0.05, 0.01) for m in (3, 5)])


def report(run, stream, policies, recorded_best):
    full, scores = replay(stream, Policy())
    lines = [f"## {run}", ""]
    if recorded_best and recorded_best != full["best"]:
        lines.append(f"WARNING: replay selects {full['best']}, final_eval.py selected {recorded_best}")
        lines.append("")
    lines += [f"Nodes with results: {len(scores)}; results {full['results']}; "
              f"evaluation time {full['eval_s'] / 3600:.1f} h (summed over parallel evaluations). "
              f"Selected without stopping: {full['best']} "
              f"({sum(scores[full['best']])}/{len(scores[full['best']])}).", "",
              "| policy | results kept | nodes retired | evaluation time saved | selected | same as without stopping "
              "| selected node's full record |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    rows = []
    for p in policies:
        s, _ = replay(stream, p)
        b = s["best"]
        same = "yes" if b == full["best"] else "no"
        record = f"{sum(scores[b])}/{len(scores[b])}"
        saved = 100 * s["saved_s"] / full["eval_s"] if full["eval_s"] else 0.0
        lines.append(f"| {p.name} | {s['results']} | {s['retired']} | {saved:.1f}% | {b} | {same} | {record} |")
        rows.append({"run": run, "policy": p.name, "kept": s["results"], "skipped": s["skipped"],
                     "retired": s["retired"], "saved_share": saved / 100, "best": b, "same": same == "yes",
                     "best_full": [sum(scores[b]), len(scores[b])]})
    return lines + [""], rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--output-root", required=True, help="directory holding <run>/final_eval/final_result.json")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--max-hours", type=float, default=9.646, help="equal-GPU-hour cut (the final evaluation's)")
    ap.add_argument("--rows-out", help="also write one JSON row per run and policy here")
    ap.add_argument("--permutations", type=int, default=0,
                    help="also replay this many random orders of each node's results (robustness check)")
    args = ap.parse_args()

    by_run = load(args.ledger, args.runs)
    out, rows = [f"Cut at {args.max_hours} GPU hours. Saved time is the share of summed evaluation wall time "
                 f"whose results a policy drops.", ""], []
    robust = {}
    for run in args.runs:
        events, _ = analyze_ledger.truncate(by_run[run], max_hours=args.max_hours)
        stream = result_stream(events)
        lines, r = report(run, stream, default_policies(), final_best(args.output_root, run))
        out += lines
        rows += r
        if args.permutations:
            robust[run] = robustness(run, stream, default_policies()[1:], args.permutations)
    if robust:
        names = list(next(iter(robust.values())))
        out += [f"## Robustness: {args.permutations} random orders of each node's results", "",
                "Each cell: share of orders in which the selection changes / mean evaluation time saved / mean loss "
                "in the selected node's full-record solve rate (percentage points; negative = better).", "",
                "| policy | " + " | ".join(robust) + " |", "|" + " --- |" * (len(robust) + 1)]
        for name in names:
            out.append(f"| {name} | " + " | ".join(f"{100 * robust[r][name][0]:.0f}% / {100 * robust[r][name][1]:.1f}% / {100 * robust[r][name][2]:+.1f}"
                                                for r in robust) + " |")
        out.append("")
    print("\n".join(out))
    if args.rows_out:
        opener = gzip.open if args.rows_out.endswith(".gz") else open
        with opener(args.rows_out, "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
