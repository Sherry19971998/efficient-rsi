"""Noise and test-time-scaling statistics from repeated evaluations (repos/HGM/repeat_eval.py).

Noise (experiment 1): several passes of one agent over the same tasks.
  - per-pass accuracy and its spread across passes
  - per-task solve rate p_t; flaky tasks (0 < p_t < 1)
  - within-task variance s2 = mean p_t (1 - p_t): the part of a one-pass accuracy that is luck.
    One pass over n tasks has luck SE sqrt(s2 / n); telling two agents apart by Delta with
    80% power at alpha 0.05 (each scored once, same tasks) needs about
    n = 2 s2 (1.96 + 0.84)^2 / Delta^2 tasks from luck alone (real differences between
    tasks add to this, so it is a lower bound).

Test-time scaling (experiment 2): passes of the initial agent over the held-out tasks.
  - pass@k (unbiased estimator) for k up to the number of passes. Polyglot's tests are hidden
    from the agent, so choosing the passing attempt needs an oracle: an upper bound on what
    repeated attempts buy.
  - GPU hours per pass, from the ledger (RSI_RUN=<label>-rep<i>)
  - against searched agents (final_result.json): the smallest k whose pass@k reaches the
    searched agent's accuracy, and how many deployed tasks the search's one-off GPU hours
    take to pay off against attempting each task k times.

Usage:
    python repeat_stats.py --ledger runs/ledger.jsonl --noise ../repos/HGM/output_hgm/noise-search
    python repeat_stats.py --ledger runs/ledger.jsonl --tts ../repos/HGM/output_hgm/tts-heldout \\
        --tts-extra <initial held-out final_result.json> --result hgm-fixed=<run>/final_eval/final_result.json ...
"""

import argparse
import glob
import json
import os
from math import comb

import analyze_ledger

Z = 1.96 + 0.84  # two-sided alpha 0.05, power 0.8


def load_passes(label_dir):
    """[(rsi_run label, {task: 0/1})] for every rep*/result.json under an output_hgm/<label> dir."""
    out = []
    for path in sorted(glob.glob(os.path.join(label_dir, "rep*", "result.json"))):
        with open(path) as f:
            r = json.load(f)
        out.append((r["rsi_run"], {t: int(v) for t, v in r["result"].items()}))
    return out


def task_rates(passes):
    tasks = sorted(set.intersection(*(set(p) for p in passes)))
    return tasks, {t: sum(p[t] for p in passes) / len(passes) for t in tasks}


def noise_stats(passes):
    tasks, rate = task_rates(passes)
    n = len(tasks)
    accs = [sum(p[t] for t in tasks) / n for p in passes]
    mean = sum(accs) / len(accs)
    s2 = sum(r * (1 - r) for r in rate.values()) / n
    # p(1-p) from R passes underestimates the variance by a factor (R-1)/R
    s2 *= len(passes) / (len(passes) - 1) if len(passes) > 1 else 1
    return {
        "passes": len(passes), "tasks": n, "pass_accuracies": accs, "mean_accuracy": mean,
        "accuracy_sd_across_passes": (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5 if len(accs) > 1 else None,
        "always_solved": sum(r == 1 for r in rate.values()), "never_solved": sum(r == 0 for r in rate.values()),
        "flaky": sum(0 < r < 1 for r in rate.values()),
        "within_task_var": s2, "luck_se_one_pass": (s2 / n) ** 0.5,
        "tasks_needed": {d: int(2 * s2 * Z ** 2 / d ** 2) + 1 for d in (0.03, 0.05, 0.10)},
    }


def pass_at_k(passes, k):
    """Unbiased pass@k averaged over tasks (Chen et al. 2021): 1 - C(n-c, k) / C(n, k)."""
    tasks, _ = task_rates(passes)
    n = len(passes)
    vals = []
    for t in tasks:
        c = sum(p[t] for p in passes)
        vals.append(1.0 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0)
    return sum(vals) / len(vals)


def pass_hours(ledger_events, rsi_run):
    return analyze_ledger.gpu_hours([e for e in ledger_events if e.get("run") == rsi_run])


def break_even(search_hours, acc, per_pass_searched, curve, per_pass_initial, n_tasks):
    """(k, pass@k, deployed tasks for the search to pay off) for the cheapest k with pass@k >= acc."""
    for k, pk in curve:
        if pk >= acc:
            c_tts = k * per_pass_initial / n_tasks
            c_search = per_pass_searched / n_tasks
            extra = c_tts - c_search
            return k, pk, (search_hours / extra if extra > 0 else float("inf"))
    return None, None, None


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--noise", help="output_hgm/<label> of the noise passes")
    ap.add_argument("--tts", help="output_hgm/<label> of the held-out passes of the initial agent")
    ap.add_argument("--tts-extra", help="final_result.json whose 'initial' is one more held-out pass")
    ap.add_argument("--result", action="append", default=[], help="LABEL=final_result.json of a searched agent")
    args = ap.parse_args()
    events = analyze_ledger.load(args.ledger)

    if args.noise:
        passes = load_passes(args.noise)
        st = noise_stats([p for _, p in passes])
        hours = [pass_hours(events, r) for r, _ in passes]
        print(f"## Evaluation noise: {st['passes']} passes of the same agent over {st['tasks']} tasks\n")
        print("| pass | accuracy | GPU hours |\n| --- | --- | --- |")
        for (r, _), a, h in zip(passes, st["pass_accuracies"], hours):
            print(f"| {r} | {pct(a)} | {h:.2f} |")
        sd = st["accuracy_sd_across_passes"]
        print(f"\nMean {pct(st['mean_accuracy'])}, SD across passes {pct(sd) if sd is not None else '—'}. "
              f"Tasks always solved {st['always_solved']}, never {st['never_solved']}, flaky {st['flaky']}.")
        print(f"Luck SE of one pass: {pct(st['luck_se_one_pass'])}. Tasks needed to tell two agents apart "
              f"(80% power, luck only): " + ", ".join(f"{pct(d)}: {n}" for d, n in st["tasks_needed"].items()) + "\n")

    if args.tts:
        passes = [p for _, p in load_passes(args.tts)]
        runs = [r for r, _ in load_passes(args.tts)]
        if args.tts_extra:
            with open(args.tts_extra) as f:
                extra = json.load(f).get("initial")
            if extra:
                passes.append({t: int(v) for t, v in extra.items()})
        tasks, _ = task_rates(passes)
        per_pass = sum(pass_hours(events, r) for r in runs) / len(runs) if runs else float("nan")
        curve = [(k, pass_at_k(passes, k)) for k in range(1, len(passes) + 1)]
        print(f"## Test-time scaling of the initial agent: {len(passes)} passes over {len(tasks)} held-out tasks\n")
        print(f"GPU hours per pass: {per_pass:.2f} (mean of {len(runs)} logged passes). "
              "pass@k needs an oracle to pick the passing attempt (tests are hidden), so it is an upper bound.\n")
        print("| k | pass@k (upper bound) | GPU hours for k passes |\n| --- | --- | --- |")
        for k, pk in curve:
            print(f"| {k} | {pct(pk)} | {k * per_pass:.2f} |")
        if args.result:
            print("\n## Search vs repeated attempts\n")
            print("| configuration | held-out accuracy | search GPU hours | GPU hours per held-out pass | "
                  "cheapest k with pass@k >= accuracy | search pays off after (tasks) |")
            print("| --- | --- | --- | --- | --- | --- |")
            for item in args.result:
                label, _, path = item.partition("=")
                with open(path) as f:
                    r = json.load(f)
                h = r.get("cut", {}).get("gpu_hours")
                own = pass_hours(events, f"{r['run']}-final")
                k, pk, n = break_even(h or 0, r["best_accuracy"], own, curve, per_pass, len(tasks))
                verdict = (f"k={k} ({pct(pk)})" if k else f"none up to k={len(passes)}")
                payoff = "—" if k is None else ("never (k attempts cost no more)" if n == float("inf") else f"{n:,.0f}")
                print(f"| {label} ({r['run']}) | {pct(r['best_accuracy'])} | {h if h is not None else '?'} | "
                      f"{own:.2f} | {verdict} | {payoff} |")
            print("\nThe searched agent's held-out pass also includes the initial agent's pass when "
                  "final_eval.py ran without --skip-initial; its per-pass hours are then about double.")


if __name__ == "__main__":
    main()
