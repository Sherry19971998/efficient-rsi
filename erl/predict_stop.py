"""Prediction-driven early stopping, retrained as the search goes (offline replay).

The node policy in policy.py stops a child once its Beta upper bound falls below the
incumbent's lower bound, from the child's own results alone. Here a predictor of the
child's eventual solve rate adds a prior to that posterior:

    Beta(1 + solved + w * p_hat, 1 + failed + w * (1 - p_hat))

p_hat comes from what is known before or early in the child's evaluation: its patch (the
diff features of prescreen/predict_from_diff.py), its parent's observed solve rate when it
was created, its depth, and its first k results with their evaluation times. The prior
weight w says how many tasks' worth of evidence the prediction is trusted for; w = 0 is
the plain statistical policy. A confident low prediction lets a weak child be stopped
after fewer tasks. The final selection still uses only real results (the Beta lower bound
on what was kept), so a wrong prediction can cost evaluations or drop a node, but it
never scores one.

w can also be "auto": the prior is then worth as many tasks as the predictor has earned
in this search. Each node's first prediction is kept with the number of results it had
then; once the node has --min-label results *after* that point, the gap between its rate
on those later results and the prediction, minus their binomial noise, estimates how far
true rates scatter around predictions (sigma^2). Only later results count: a node the
predictor stopped has none, so the predictor cannot confirm itself with the very results
that made it stop the node (labelling with all results did exactly that, and the weight
ran up to the cap).
A Beta prior with mean p and that variance has weight p(1 - p) / sigma^2 - 1, recomputed
at every refit and capped at W_MAX; with fewer than MIN_CALIB such nodes it is 0. A
predictor that only knows the average earns a weight near 0, so the policy falls back to
the plain statistical one instead of trusting it blindly.

The predictor is retrained every --refit results on what the search has observed by then:
for each node with at least --min-label results, its prefixes k = 1 .. K are rows and its
current solve rate is the target. A stopped node keeps only the results it got, exactly
as an online search would see it. With --history, nodes from runs that finished before
this one started are added to the training rows (their full records), like EarlyEval's
labelled runs from the same benchmark.

    python predict_stop.py --ledger ../runs/ledger.jsonl --prescreen-rows ../runs/prescreen_rows.jsonl \
        --runs hgm-fixed-r0 ... [--history]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "instrumentation"))

import analyze_ledger  # noqa: E402
from results import select_best  # noqa: E402

import replay as rp  # noqa: E402
from policy import beta_quantile  # noqa: E402

DIFF_KEYS = ("n_files", "n_py", "added", "removed", "net", "new_files", "touches_main", "touches_llm",
             "touches_tools", "main_removed", "removed_defs", "added_defs", "provider_strings", "empty")
FEATURES = ("k", "solved", "rate", "parent_rate", "parent_n", "depth", "mean_eval_s", "long_share") + DIFF_KEYS
MAX_PREFIX = 10      # prefixes k = 0 .. MAX_PREFIX become training rows
W_MAX = 10           # cap on the automatic prior weight; w = 10 already changed selections offline
MIN_CALIB = 10       # nodes with a prediction and a label before the weight is trusted
LONG_S = analyze_ledger.TIMEOUT_S
# Paired with the parent, over the child's first k tasks: tasks the parent solved (and how many of
# them the child lost), tasks the parent failed (and how many the child won). Used with paired=True.
PAIRED_KEYS = ("reg_n", "reg_lost", "reg_lost_rate", "gain_n", "gain_won")


class NodeState:
    """What the search knows about one node: its static features and its kept results."""

    def __init__(self, static, parent=None, paired=False):
        self.static = static
        self.parent = parent      # the parent's NodeState, for the paired features
        self.paired = paired
        self.task_map = None      # fixed task -> result (the root's cached evaluation), else from tasks
        self.tasks = []           # task of each result (None for a batch result)
        self.results = []
        self.costs = []   # seconds per result (a batch's time is split evenly over its results)
        self.censored = []   # True for a result set to unsolved by per-task stopping

    def uncensored(self, start=0):
        cens = self.censored + [False] * (len(self.results) - len(self.censored))
        return [r for r, c in zip(self.results[start:], cens[start:]) if not c]

    def features(self, k=None):
        k = len(self.results) if k is None else k
        r, c = self.results[:k], self.costs[:k]
        s = sum(r)
        return [k, s, s / k if k else 0.0, self.static["parent_rate"], self.static["parent_n"], self.static["depth"],
                float(np.mean(c)) if c else 0.0, float(np.mean([x >= LONG_S for x in c])) if c else 0.0] + \
               [self.static.get(key, 0) for key in DIFF_KEYS] + (self.paired_features(k) if self.paired else [])

    def solved_tasks(self):
        """task -> 1/0 as known now (the latest result per task)."""
        if self.task_map is not None:
            return self.task_map
        return {t: r for t, r in zip(self.tasks, self.results) if t is not None}

    def paired_features(self, k):
        pmap = self.parent.solved_tasks() if self.parent is not None else {}
        rn = rl = gn = gw = 0
        for t, r in zip(self.tasks[:k], self.results[:k]):
            if t is None or t not in pmap:
                continue
            if pmap[t]:
                rn += 1; rl += r == 0
            else:
                gn += 1; gw += r == 1
        return [rn, rl, rl / rn if rn else 0.0, gn, gw]


def training_rows(states, min_label):
    X, y, w = [], [], []
    for st in states:
        r = st.uncensored()
        if len(r) < min_label:
            continue
        target = sum(r) / len(r)
        n = len(st.results)
        ks = range(0, min(n, MAX_PREFIX) + 1)
        for k in ks:
            X.append(st.features(k)); y.append(target); w.append(1.0 / len(ks))
    return X, y, w


def fit(X, y, w):
    if len(set(y)) < 2 or len(X) < 30:
        return None
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=10,
                                      random_state=0)
    m.fit(np.array(X, float), np.array(y, float), sample_weight=np.array(w, float))
    return m


def auto_weight(first_pred, states, min_label):
    """Prior weight the predictor has earned: p(1 - p) / sigma^2 - 1 from its first predictions,
    scored on each node's results after the prediction."""
    gaps, noise, ps = [], [], []
    for node, (p_hat, k) in first_pred.items():
        r = states[node].uncensored(k)
        if len(r) >= min_label:
            obs = sum(r) / len(r)
            gaps.append((obs - p_hat) ** 2)
            noise.append(obs * (1 - obs) / len(r))
            ps.append(p_hat)
    if len(gaps) < MIN_CALIB:
        return 0.0
    var = np.mean(gaps) - np.mean(noise)
    p = np.mean(ps)
    if var <= 0:
        return float(W_MAX)
    return float(np.clip(p * (1 - p) / var - 1, 0, W_MAX))


def static_features(events, diff_rows):
    """node -> diff features, depth, and the parent's observed rate when the node was created."""
    parent = {}
    for e in events:
        if e["type"] == "node_created" and e.get("node"):
            parent[e["node"]] = e.get("parent")

    def depth(n):
        d = 0
        while n in parent and d < 1000:
            n, d = parent[n], d + 1
        return d
    return {n: {**{k: diff_rows.get(n, {}).get(k, 0) for k in DIFF_KEYS}, "depth": depth(n), "parent": p}
            for n, p in parent.items()}


def replay(events, stream, diff_rows, w, alpha, min_evals, refit, min_label, history_rows=None, eps=0.05,
           censored="count", paired=False, root_tasks=None):
    """censored: what to do with results that per-task stopping set to unsolved (marked by a
    sixth stream field). "count" treats them as real failures everywhere; "labels" leaves them
    out of the predictor's targets and calibration; "policy" also leaves them out of the
    stopping posterior. The final selection always counts them as unsolved."""
    static = static_features(events, diff_rows)
    created = sorted((e["ts"], e["node"]) for e in events if e["type"] == "node_created" and e.get("node"))
    states = {}
    lcb = {}
    retired = set()
    stats = {"results": 0, "saved_s": 0.0, "eval_s": 0.0, "fits": 0}
    model, since_fit = None, 0
    hX, hy, hw = history_rows or ([], [], [])
    preds = []   # (k, p_hat, node) for the predictor's error report
    first_pred = {}
    auto = w == "auto"
    w_now = 0.0 if auto else w
    stats["w_trace"] = []
    ci = 0

    def state(n):
        if n not in states:
            st = static.get(n, {k: 0 for k in DIFF_KEYS} | {"depth": 0, "parent": None})
            p = states.get(st.get("parent"))
            st = dict(st, parent_rate=(sum(p.results) / len(p.results)) if p and p.results else 0.0,
                      parent_n=len(p.results) if p else 0)
            states[n] = NodeState(st, parent=p, paired=paired)
            if n == "initial" and root_tasks is not None:
                states[n].task_map = root_tasks
        return states[n]

    for node, res, cost, task, ts, *flag in stream:
        cens = bool(flag and flag[0])
        # nodes created before this result fix their parent's observed rate now
        while ci < len(created) and created[ci][0] <= ts:
            state(created[ci][1])
            ci += 1
        st = state(node)
        stats["eval_s"] += cost
        incumbent = max(lcb.values(), default=0.0)
        if censored == "policy":
            u = st.uncensored()
            k, s = len(u), sum(u)
        else:
            k, s = len(st.results), sum(st.results)
        go = True
        if node != "initial" and (node in retired or k >= min_evals):
            if node in retired:
                go = False
            else:
                a, b = 1 + s, 1 + k - s
                if model is not None:
                    p_hat = float(np.clip(model.predict(np.array([st.features()], float))[0], 0.01, 0.99))
                    preds.append((k, p_hat, node))
                    first_pred.setdefault(node, (p_hat, k))
                    a, b = a + w_now * p_hat, b + w_now * (1 - p_hat)
                go = float(beta.ppf(1 - alpha, a, b)) >= incumbent
        if not go:
            retired.add(node)
            stats["saved_s"] += cost
            continue
        per = cost / len(res) if res else 0.0
        st.results.extend(res)
        st.tasks.extend([task if len(res) == 1 else None] * len(res))
        st.costs.extend([per] * len(res))
        st.censored.extend([cens and censored != "count"] * len(res))
        lcb[node] = beta_quantile(eps, sum(st.results), len(st.results))
        stats["results"] += len(res)
        since_fit += len(res)
        if (auto or w > 0) and since_fit >= refit:
            X, y, sw = training_rows([v for n, v in states.items() if n != "initial"], min_label)
            model = fit(hX + X, hy + y, hw + sw) or model
            stats["fits"] += model is not None
            if auto:
                w_now = auto_weight(first_pred, states, min_label)
                stats["w_trace"].append(round(w_now, 1))
            since_fit = 0
    kept = {n: v.results for n, v in states.items() if v.results}
    stats["best"] = select_best(kept, eps=eps)
    stats["retired"] = len(retired)
    stats["preds"] = preds
    return stats, states


def history_for(run, starts, finished_states):
    """Training rows from the runs that ended before `run` started."""
    X, y, w = [], [], []
    for other, (st, en) in starts.items():
        if other != run and en < starts[run][0] and other in finished_states:
            a, b, c = training_rows(finished_states[other], 1)
            X += a; y += b; w += c
    return X, y, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--prescreen-rows", required=True, help="prescreen_rows.jsonl (diff features per node)")
    ap.add_argument("--runs", nargs="+", required=True, help="runs to replay")
    ap.add_argument("--history-runs", nargs="*", default=[],
                    help="also load these runs, as history for later runs (with --history)")
    ap.add_argument("--history", action="store_true", help="train on runs that finished earlier, too")
    ap.add_argument("--max-hours", type=float, default=9.646)
    ap.add_argument("--refit", type=int, default=50)
    ap.add_argument("--min-label", type=int, default=5)
    args = ap.parse_args()

    diff_rows = {}
    with open(args.prescreen_rows) as f:
        for line in f:
            r = json.loads(line)
            diff_rows[(r["run"], r["node"])] = r
    all_runs = list(dict.fromkeys(args.runs + args.history_runs))
    by_run = rp.load(args.ledger, all_runs)
    cut = {}
    for run in all_runs:
        events, _ = analyze_ledger.truncate(by_run[run], max_hours=args.max_hours)
        cut[run] = (events, rp.result_stream(events), {n: r for (rr, n), r in diff_rows.items() if rr == run})
    starts = {r: (min(e["ts"] for e in ev), max(e["ts"] for e in ev)) for r, (ev, _, _) in cut.items()}
    finished = {}
    for run, (ev, stream, diffs) in cut.items():
        _, states = replay(ev, stream, diffs, 0, 1e-9, 10 ** 9, 10 ** 9, 1)
        finished[run] = [v for n, v in states.items() if n != "initial"]

    print(f"Cut at {args.max_hours} GPU hours; refit every {args.refit} results; labels need {args.min_label} "
          f"results; history: {'runs that finished earlier' if args.history else 'none'}. "
          "Loss: selected node's full-record solve rate below the selection without stopping (pp).\n")
    for run in args.runs:
        events, stream, diffs = cut[run]
        hist = history_for(run, starts, finished) if args.history else None
        full, states = replay(events, stream, diffs, 0, 1e-9, 10 ** 9, 10 ** 9, 1)
        mean = {n: np.mean(v.results) for n, v in states.items() if v.results}
        print(f"## {run}\n")
        if hist:
            print(f"History rows: {len(hist[0])}\n")
        print("| w | alpha | min evals | results kept | nodes retired | time saved | same / loss (pp) | fits "
              "| predictor MAE at k<=3 (n) | auto w: last (max) |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for alpha in (0.2, 0.1):
            for min_evals in (1, 3):
                for w in (0, 5, 10, 20, "auto"):
                    s, _ = replay(events, stream, diffs, w, alpha, min_evals, args.refit, args.min_label, hist)
                    errs = [abs(p - mean[n]) for k, p, n in s["preds"] if k <= 3 and n in mean]
                    mae = f"{np.mean(errs):.3f} ({len(errs)})" if errs else "—"
                    same = "yes" if s["best"] == full["best"] else "no"
                    print(f"| {w} | {alpha} | {min_evals} | {s['results']} | {s['retired']} | "
                          f"{100 * s['saved_s'] / full['eval_s']:.1f}% | {same} / "
                          f"{100 * (mean[full['best']] - mean[s['best']]):+.1f} | {s['fits']} | {mae} | "
                          f"{(str(s['w_trace'][-1]) + ' (' + str(max(s['w_trace'])) + ')') if s['w_trace'] else '—'} |",
                          flush=True)
        print()


if __name__ == "__main__":
    main()
