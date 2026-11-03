"""Per-task early stopping, replayed online and judged by the search's final selection.

EarlyEval stops a single agent run once a classifier is confident it will fail. Inside
self-improvement every child is a scaffold the classifier has not seen, so here the
classifier is trained online: an evaluation that starts at time t is judged only by a
model fitted on evaluations that finished before t, refitted every --refit completions.
The features are the proxy's record of the model calls (early_stop/simulate_ledger.py),
which the agent cannot rewrite.

A stopped evaluation counts as unsolved and its remaining time (rest of the agent run and
the test phase) as saved. The edited result stream then goes through the selection replay
(replay.py), alone and together with the node-level stopping policy, so each setting is
judged by what it does to the final pick, not by its per-task accuracy.

Three families are compared:
  cap<C>         a lower time limit: every run longer than C seconds is cut at C
  cap<C>-if<g>   a conditional limit: at C seconds, stop only if P(unsolved) >= g
  f<f>-after<m>  the classifier: stop at the first call where P(unsolved) >= f, but only
                 once the run has gone m seconds (aims the stops at the long runs)

Only HGM runs: their results arrive one task at a time. DGM reports a batch per node, so a
single task's result cannot be edited in its stream.

    python micro_stop.py --ledger ../runs/ledger.jsonl --runs hgm-fixed-r0 hgm-orig-r0 --max-hours 9.646
"""

import argparse
import os
import sys
from bisect import bisect_left

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "instrumentation"))
sys.path.insert(0, os.path.join(HERE, "..", "instrumentation", "early_stop"))

import analyze_ledger  # noqa: E402
import simulate_ledger as sl  # noqa: E402

import replay as rp  # noqa: E402
from policy import Policy, UcbVsIncumbent  # noqa: E402

CAPS = (1200, 1800, 2400)
F_LEVELS = (0.95, 0.97, 0.99)
COND_G = (0.7, 0.8, 0.9)
AFTER_S = (0, 600, 1200)


def trajectories(stream, calls_by_key):
    """{stream index: LedgerTrajectory} for one-task results of searched nodes with >= 2 calls."""
    out = {}
    for i, (node, res, _, task, *_) in enumerate(stream):
        rec = calls_by_key.get((node, task))
        if task is None or node == "initial" or len(res) != 1 or not rec or len(rec["calls"]) < 2:
            continue
        out[i] = sl.LedgerTrajectory(node, task, "resolved" if res[0] else "unresolved", rec["calls"],
                                     rec["agent_end"], rec["agent_wall"], rec["test_wall"])
    return out


def start_of(t):
    return t.calls[0]["ts"] - t.calls[0]["latency"]


def end_of(t):
    return t.agent_end if t.agent_end is not None else t.calls[-1]["ts"]


def online_probabilities(trajs, refit, min_train):
    """{index: per-prefix P(solved)} using, for each run, the latest model fitted only on runs
    that had finished before it started; runs before the first model get None (never stopped)."""
    by_end = sorted(trajs, key=lambda i: end_of(trajs[i]))
    ends = [end_of(trajs[i]) for i in by_end]
    models = []  # (number of finished runs it was fitted on, model)
    out = {}
    for i in sorted(trajs, key=lambda i: start_of(trajs[i])):
        n_done = bisect_left(ends, start_of(trajs[i]))
        if n_done >= min_train and (not models or n_done - models[-1][0] >= refit):
            m = sl.fit([trajs[j] for j in by_end[:n_done]])
            if m is not None:
                models.append((n_done, m))
        out[i] = sl.probabilities(models[-1][1], trajs[i]) if models else None
    return out, len(models)


def stop_classifier(t, probs, f, after):
    """Seconds saved if the classifier stops t, else None."""
    if probs is None:
        return None
    start = start_of(t)
    for k, p in enumerate(probs, start=1):
        last = t.calls[k - 1]["ts"]
        if 1 - p >= f and last - start >= after:
            return max(0.0, end_of(t) - last) + t.test_wall
    return None


def stop_conditional_cap(t, probs, cap, g):
    """Seconds saved if, at the first call that ends cap seconds into the run, the classifier
    puts P(unsolved) >= g; a run the classifier cannot judge yet is never stopped."""
    if probs is None:
        return None
    start = start_of(t)
    for k, p in enumerate(probs, start=1):
        last = t.calls[k - 1]["ts"]
        if last - start >= cap:
            return max(0.0, end_of(t) - last) + t.test_wall if 1 - p >= g else None
    return None


def stop_cap(t, cap):
    return t.agent_wall - cap + t.test_wall if t.agent_wall > cap else None


def edited(stream, stops):
    """stream with each stopped result set to 0 and its cost reduced by the time saved."""
    out = list(stream)
    for i, saved in stops.items():
        node, _, cost, task, *rest = stream[i]
        out[i] = (node, [0], max(0.0, cost - saved), task, *rest)
    return out


def evaluate(stream, stops, trajs, mean, ref_best, total_s):
    lost = sum(trajs[i].solved for i in stops)
    row = {"stopped": len(stops), "lost_solves": lost, "micro_saved": sum(stops.values()) / total_s}
    s_stream = edited(stream, stops)
    for label, pol in (("alone", Policy()), ("with_ucb", UcbVsIncumbent(0.2, 3))):
        s, _ = rp.replay(s_stream, pol)
        saved = (sum(stops.values()) + s["saved_s"]) / total_s
        row[label] = (saved, s["best"] == ref_best, 100 * (mean[ref_best] - mean[s["best"]]))
    return row


def run_one(run, events, calls_by_key, refit, min_train):
    stream = rp.result_stream(events)
    full, scores = rp.replay(stream, Policy())
    mean = {k: np.mean(v) for k, v in scores.items() if v}
    trajs = trajectories(stream, calls_by_key)
    probs, n_models = online_probabilities(trajs, refit, min_train)
    total = full["eval_s"]
    long_share = sum(t.agent_wall + t.test_wall for t in trajs.values() if t.agent_wall >= analyze_ledger.TIMEOUT_S)
    lines = [f"## {run}", "",
             f"{len(trajs)} one-task evaluations with calls; {sum(t.solved for t in trajs.values())} solved; "
             f"{sum(p is not None for p in probs.values())} judged by an online model ({n_models} fits). "
             f"Runs that hit the time limit hold {100 * long_share / total:.1f}% of evaluation time.", "",
             "| setting | stopped | of them solved (lost) | time saved by stopping | selection same / loss (pp) "
             "| + node policy ucb0.2: total saved | + node policy: same / loss (pp) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    settings = [(f"cap{c}", {i: s for i, t in trajs.items() if (s := stop_cap(t, c)) is not None}) for c in CAPS]
    for c in (1800, 2400):
        for g in COND_G:
            settings.append((f"cap{c}-if{g}", {i: s for i, t in trajs.items()
                                               if (s := stop_conditional_cap(t, probs[i], c, g)) is not None}))
    for f in F_LEVELS:
        for a in AFTER_S:
            settings.append((f"f{f}-after{a}", {i: s for i, t in trajs.items()
                                                if (s := stop_classifier(t, probs[i], f, a)) is not None}))
    rows = []
    base = evaluate(stream, {}, trajs, mean, full["best"], total)
    for name, stops in [("none", {})] + settings:
        r = base if name == "none" else evaluate(stream, stops, trajs, mean, full["best"], total)
        a, w = r["alone"], r["with_ucb"]
        lines.append(f"| {name} | {r['stopped']} | {r['lost_solves']} | {100 * r['micro_saved']:.1f}% | "
                     f"{'yes' if a[1] else 'no'} / {a[2]:+.1f} | {100 * w[0]:.1f}% | {'yes' if w[1] else 'no'} / {w[2]:+.1f} |")
        rows.append({"run": run, "setting": name, **r})
    return lines + [""], rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--max-hours", type=float, default=9.646)
    ap.add_argument("--refit", type=int, default=100, help="refit after this many more finished evaluations")
    ap.add_argument("--min-train", type=int, default=100, help="finished evaluations before the first model")
    args = ap.parse_args()

    by_run = rp.load(args.ledger, args.runs)
    header = [f"Cut at {args.max_hours} GPU hours. Time shares are of the run's summed evaluation wall time. "
           "Loss: the selected node's full-record solve rate below the selection without stopping "
           "(percentage points; negative = better).", ""]
    print("\n".join(header), flush=True)
    for run in args.runs:
        events, cutoff = analyze_ledger.truncate(by_run[run], max_hours=args.max_hours)
        calls = sl.load_ledger(args.ledger, run)
        if cutoff is not None:
            calls = {k: v for k, v in calls.items() if v["agent_end"] is not None and v["agent_end"] <= cutoff}
        lines, _ = run_one(run, events, calls, args.refit, args.min_train)
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
