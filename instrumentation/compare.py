"""Compare configurations across seeds: held-out accuracy with bootstrap CIs, and cost.

Each --result is LABEL=path/to/final_result.json (written by repos/HGM/final_eval.py);
repeat it once per seed. With --ledger, the cost of each run's search (cut at the same
budget, GPU hours or task evaluations, that the final evaluation used) is added.

Usage:
    python compare.py --result hgm-fixed=a/final_result.json --result hgm-fixed=b/final_result.json \\
                      --result dgm=c/final_result.json ... --reference dgm --ledger runs/ledger.jsonl
"""

import argparse
import json
from collections import OrderedDict

import numpy as np

import analyze_ledger
from results import bootstrap_diff, bootstrap_mean


def pct(x):
    return f"{100 * x:.1f}%"


def run_cost(events):
    _, usd, _, _, wall, cpu, _, _, gpu = analyze_ledger.build(events)
    eval_wall = sum(v for (stage, _), v in wall.items() if stage == "eval")
    return {"gpu_h": analyze_ledger.gpu_hours(events), "usd": sum(usd.values()),
            "eval_wall_h": eval_wall / 3600, "gpu_busy_h": gpu["busy_s"] / 3600}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", action="append", required=True, help="LABEL=final_result.json, once per seed")
    ap.add_argument("--reference", help="label the others are compared against (default: the first)")
    ap.add_argument("--ledger")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--initial", help="final_result.json whose held-out evaluation of the initial agent is used "
                                      "for runs evaluated with final_eval.py --skip-initial")
    args = ap.parse_args()

    shared_initial = None
    if args.initial:
        with open(args.initial) as f:
            shared_initial = json.load(f)["initial"]
    groups = OrderedDict()
    for item in args.result:
        label, _, path = item.partition("=")
        with open(path) as f:
            r = json.load(f)
        if not r.get("initial") and shared_initial:
            r["initial"] = shared_initial
        groups.setdefault(label, []).append(r)
    task_sets = {tuple(r["heldout_tasks"]) for g in groups.values() for r in g}
    if len(task_sets) != 1:
        raise SystemExit("All results must use the same held-out tasks; rerun final_eval.py with the same settings.")
    tasks = list(next(iter(task_sets)))
    ref = args.reference or next(iter(groups))

    print(f"Held-out tasks: {len(tasks)}. Intervals are 95% two-level bootstrap (seeds and tasks).\n")
    print("| configuration | seeds | held-out accuracy [95% CI] | initial agent | gain over initial [95% CI] |")
    print("| --- | --- | --- | --- | --- |")
    for label, runs in groups.items():
        m = bootstrap_mean([r["best"] for r in runs], tasks, n_boot=args.n_boot)
        inits = [r["initial"] for r in runs if r.get("initial")]
        if inits:
            g = bootstrap_diff([r["best"] for r in runs], inits, tasks, n_boot=args.n_boot)
            init_txt, gain_txt = pct(np.mean([np.mean(list(i.values())) for i in inits])), \
                f"{pct(g['diff'])} [{pct(g['ci_low'])}, {pct(g['ci_high'])}]"
        else:
            init_txt = gain_txt = "—"
        print(f"| {label} | {m['seeds']} | {pct(m['mean'])} [{pct(m['ci_low'])}, {pct(m['ci_high'])}] | "
              f"{init_txt} | {gain_txt} |")

    print(f"\nDifferences against {ref} (same held-out tasks, seeds resampled independently):\n")
    print("| configuration | minus reference [95% CI] | P(better than reference) |")
    print("| --- | --- | --- |")
    for label, runs in groups.items():
        if label == ref:
            continue
        d = bootstrap_diff([r["best"] for r in runs], [r["best"] for r in groups[ref]], tasks, n_boot=args.n_boot)
        print(f"| {label} | {pct(d['diff'])} [{pct(d['ci_low'])}, {pct(d['ci_high'])}] | {d['p_a_better']:.2f} |")

    if args.ledger:
        print("\nSearch cost per seed, cut at the budget the final evaluation used (mean, with min to max across seeds):\n")
        print("| configuration | GPU occupancy hours | API USD | evaluation wall hours | GPU busy hours |")
        print("| --- | --- | --- | --- | --- |")
        for label, runs in groups.items():
            costs = []
            for r in runs:
                events, _ = analyze_ledger.truncate(analyze_ledger.load(args.ledger, r["run"]),
                                                    r.get("max_evals"), r.get("max_hours"))
                costs.append(run_cost(events))
            cells = []
            for k in ("gpu_h", "usd", "eval_wall_h", "gpu_busy_h"):
                v = [c[k] for c in costs]
                cells.append(f"{np.mean(v):.2f} ({min(v):.2f} to {max(v):.2f})")
            print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
