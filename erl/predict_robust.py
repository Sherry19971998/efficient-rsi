"""Robustness of prediction-driven stopping, alone and stacked on per-task stopping.

Like replay.py --permutations: each node's results are dealt to its own evaluation slots in
a random order, n times, and every setting is replayed on each order. A slot keeps its
time stamp (so node creation and the parent's observed rate stay causal); the result and
its evaluation time move together. The selection without stopping is the same in every
order, so a change in selection comes from the setting alone.

Settings (all 80% upper bound, at least 3 tasks): the plain statistical policy (w = 0),
the prediction prior with w = 5 and with the automatic weight (both trained with history
from runs that finished earlier), and the automatic weight on a stream already edited by
the conditional time limit cap1800-if0.9 of micro_stop.py (judged once, on the recorded
order, since its classifier is trained on the runs' real timeline; the stopped results
then move with their slots like any other result).

    python predict_robust.py --ledger ../runs/ledger.jsonl --prescreen-rows ../runs/prescreen_rows.jsonl \
        --runs hgm-fixed-r0 ... --history-runs dgm-r0 ... --permutations 20
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "instrumentation"))
sys.path.insert(0, os.path.join(HERE, "..", "instrumentation", "early_stop"))

import analyze_ledger  # noqa: E402
import simulate_ledger as sl  # noqa: E402

import micro_stop as ms  # noqa: E402
import predict_stop as ps  # noqa: E402
import replay as rp  # noqa: E402

ALPHA, MIN_EVALS = 0.2, 3


def shuffled_keep_ts(stream, rng):
    t = rp.shuffled(stream, rng)
    return [(*t[i][:4], stream[i][4], *t[i][5:]) for i in range(len(stream))]


def micro_edited(stream, calls, refit=100, min_train=100, cap=1800, g=0.9):
    trajs = ms.trajectories(stream, calls)
    probs, _ = ms.online_probabilities(trajs, refit, min_train)
    stops = {i: s for i, t in trajs.items() if (s := ms.stop_conditional_cap(t, probs[i], cap, g)) is not None}
    out = ms.edited(stream, stops)
    for i in stops:
        out[i] = (*out[i][:5], True)   # marked, so the predictor can leave it out of its labels
    return out, sum(stops.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--prescreen-rows", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--history-runs", nargs="*", default=[])
    ap.add_argument("--max-hours", type=float, default=9.646)
    ap.add_argument("--refit", type=int, default=50)
    ap.add_argument("--min-label", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only the settings whose name contains one of these strings")
    ap.add_argument("--root-metadata",
                    help="the initial agent's metadata.json (per-task results of the root, for the paired settings)")
    args = ap.parse_args()

    diff_rows = {}
    with open(args.prescreen_rows) as f:
        for line in f:
            r = json.loads(line)
            diff_rows[(r["run"], r["node"])] = r
    all_runs = list(dict.fromkeys(args.runs + args.history_runs))
    by_run = rp.load(args.ledger, all_runs)
    cut, cutoffs = {}, {}
    for run in all_runs:
        events, cutoffs[run] = analyze_ledger.truncate(by_run[run], max_hours=args.max_hours)
        cut[run] = (events, rp.result_stream(events), {n: r for (rr, n), r in diff_rows.items() if rr == run})
    starts = {r: (min(e["ts"] for e in ev), max(e["ts"] for e in ev)) for r, (ev, _, _) in cut.items()}
    root_tasks = None
    if args.root_metadata:
        with open(args.root_metadata) as f:
            perf = json.load(f)["overall_performance"]
        solved = set(perf["total_resolved_ids"])
        root_tasks = {t: int(t in solved) for t in perf["total_submitted_ids"]}
    finished = {False: {}, True: {}}   # history states without / with the paired features
    for run, (ev, stream, diffs) in cut.items():
        for paired in (False, True):
            _, states = ps.replay(ev, stream, diffs, 0, 1e-9, 10 ** 9, 10 ** 9, 1, paired=paired,
                                  root_tasks=root_tasks)
            finished[paired][run] = [v for n, v in states.items() if n != "initial"]

    # (name, w, micro stop threshold g or None, what the prediction policy does with stopped results,
    #  paired features with the parent)
    settings = [("w0 (statistics only)", 0, None, "count", False), ("w5 + history", 5, None, "count", False),
                ("auto + history", "auto", None, "count", False),
                ("auto + cap1800-if0.9", "auto", 0.9, "count", False),
                ("auto + cap1800-if0.95", "auto", 0.95, "count", False),
                ("auto + cap1800-if0.9, stopped left out of labels", "auto", 0.9, "labels", False),
                ("auto + cap1800-if0.9, stopped left out of labels and policy", "auto", 0.9, "policy", False),
                ("w5 + history + paired", 5, None, "count", True),
                ("auto + history + paired", "auto", None, "count", True)]
    if args.only:
        settings = [x for x in settings if any(o == x[0] for o in args.only)]
    if any(x[4] for x in settings) and root_tasks is None:
        ap.error("paired settings need --root-metadata")
    names = [x[0] for x in settings]
    print(f"Cut at {args.max_hours} GPU hours; alpha {ALPHA}, at least {MIN_EVALS} tasks; "
          f"{args.permutations} random orders per run plus the recorded one. Loss: the selected node's "
          "full-record solve rate below the selection without stopping (pp). Saved: share of the run's "
          "evaluation time, per-task stopping included.\n")
    summary = {}
    for run in args.runs:
        events, stream, diffs = cut[run]
        hist = {p: ps.history_for(run, starts, finished[p]) for p in (False, True)}
        full, states = ps.replay(events, stream, diffs, 0, 1e-9, 10 ** 9, 10 ** 9, 1)
        mean = {n: np.mean(v.results) for n, v in states.items() if v.results}
        total = full["eval_s"]
        calls = sl.load_ledger(args.ledger, run)
        if cutoffs[run] is not None:
            calls = {k: v for k, v in calls.items() if v["agent_end"] is not None and v["agent_end"] <= cutoffs[run]}
        micro = {None: (stream, 0.0)}
        for g in {x[2] for x in settings if x[2] is not None}:
            micro[g] = micro_edited(stream, calls, g=g)

        def run_all(streams):
            out = []
            for _, w, g, mode, paired in settings:
                s, _ = ps.replay(events, streams[g], diffs, w, ALPHA, MIN_EVALS, args.refit, args.min_label,
                                 hist[paired], censored=mode, paired=paired, root_tasks=root_tasks)
                errs = [abs(p - mean[n]) for k, p, n in s["preds"] if k <= 3 and n in mean]
                out.append(((s["saved_s"] + micro[g][1]) / total, s["best"] != full["best"],
                            100 * (mean[full["best"]] - mean[s["best"]]), float(np.mean(errs)) if errs else np.nan))
            return out

        recorded = run_all({g: v[0] for g, v in micro.items()})
        perms = []
        for i in range(args.permutations):
            # the same order for every setting: one seed per permutation
            perms.append(run_all({g: shuffled_keep_ts(v[0], np.random.default_rng((args.seed, i)))
                                  for g, v in micro.items()}))
        summary[run] = (recorded, perms)
        print(f"## {run}\n")
        print("| setting | recorded order: saved / same / loss | random orders: mean saved | selection changed "
              "| mean loss | worst loss | predictor MAE at k<=3 |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        for j, name in enumerate(names):
            r = recorded[j]
            col = [p[j] for p in perms]
            print(f"| {name} | {100 * r[0]:.1f}% / {'no' if r[1] else 'yes'} / {r[2]:+.1f} | "
                  f"{100 * np.mean([c[0] for c in col]):.1f}% | {sum(c[1] for c in col)}/{len(col)} | "
                  f"{np.mean([c[2] for c in col]):+.2f} | {max(c[2] for c in col):+.1f} | "
                  f"{np.nanmean([c[3] for c in col]):.3f} |", flush=True)
        print()

    print("## All runs\n")
    print("| setting | mean saved (range over runs) | orders with a changed selection | mean loss | worst loss "
          "| predictor MAE at k<=3 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for j, name in enumerate(names):
        per_run = [np.mean([p[j][0] for p in perms]) for _, perms in summary.values()]
        cols = [p[j] for _, perms in summary.values() for p in perms]
        print(f"| {name} | {100 * np.mean(per_run):.1f}% ({100 * min(per_run):.1f}–{100 * max(per_run):.1f}%) | "
              f"{sum(c[1] for c in cols)}/{len(cols)} | {np.mean([c[2] for c in cols]):+.2f} | "
              f"{max(c[2] for c in cols):+.1f} | {np.nanmean([c[3] for c in cols]):.3f} |")


if __name__ == "__main__":
    main()
