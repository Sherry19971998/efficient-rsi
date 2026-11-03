"""Offline simulation of EarlyEval-style early stopping on recorded agent trajectories.

For each held-out trajectory we walk its prefixes k = 1 .. T-1 and stop at the first
prefix where the calibrated success probability p satisfies p >= s (predict solved)
or 1 - p >= f (predict unsolved). Savings count the LLM tokens of the calls that
would not have been made, both all tokens and the GPU-relevant part (uncached input +
output; prefix-cache hits cost the GPU almost nothing); test and container time saved
are not modelled.

Splits:
  one agent           grouped by task (--folds)
  several, --split time    (default) agents in creation order (node ids start with a timestamp),
                      cut into --folds blocks; block b is predicted by a model fit on blocks
                      before it. This is what an online early stopper can actually know.
  several, --split agent   grouped by agent, every other agent may be used (optimistic: it sees
                      later agents too)

Usage:
    python simulate.py PRED_DIR[=AGENT] ... [--folds 5]
    python simulate.py --run-dir output_hgm/hgm-fixed-r0 [--run-dir ...] [--split time]
"""

import argparse
import itertools

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import beta, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from features import FEATURE_NAMES, prefix_features
from trajectories import load_predictions, load_run

# Success predictions have never reached 0.8 on Polyglot, so s matters little; the failure
# threshold f sets the trade-off and is tried finely where selection starts to change.
S_THRESHOLDS = (0.90, 0.95)
F_THRESHOLDS = (0.80, 0.85, 0.90, 0.92, 0.93, 0.94, 0.95, 0.97)
MIN_TRAJ_FOR_RANK = 10  # agents with fewer evaluated tasks are left out of the Spearman / mean checks
LCB_EPS = 0.05          # final-agent selection rule of the experiments (instrumentation/results.py)


def call_tokens(traj, j):
    s = traj.steps[j]
    return s.input_tokens + s.output_tokens


def call_gpu_tokens(traj, j):
    s = traj.steps[j]
    return s.input_tokens - s.cached_tokens + s.output_tokens


def gpu_tokens(traj):
    first = traj.first_call_tokens[0] - traj.first_call_cached + traj.first_call_tokens[1]
    return first + sum(call_gpu_tokens(traj, j) for j in range(len(traj.steps)))


def rows(trajs):
    X, y, w, g = [], [], [], []
    for t_i, t in enumerate(trajs):
        T = len(t.steps)
        for k in range(1, T + 1):
            X.append(prefix_features(t, k)); y.append(int(t.solved)); w.append(1.0 / T); g.append(t_i)
    return np.array(X, float), np.array(y), np.array(w), np.array(g)


def fit(trajs):
    X, y, w, g = rows(trajs)
    if len(set(y)) < 2:
        return None
    base = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=10, random_state=0)
    # calibration folds are grouped by trajectory, so prefixes of one run never sit on both sides
    folds = list(GroupKFold(n_splits=3).split(X, y, groups=g))
    model = CalibratedClassifierCV(base, method="sigmoid", cv=folds)
    model.fit(X, y, sample_weight=w)
    return model


def probabilities(model, t):
    """Calibrated P(solved) after each prefix k = 1 .. T-1 (one batched call per trajectory)."""
    T = len(t.steps)
    if T < 2:
        return []
    X = np.array([prefix_features(t, k) for k in range(1, T)], float)
    return list(model.predict_proba(X)[:, 1])


def simulate_one(probs, t, s, f):
    """(stop_k or None, predicted_solved, tokens_saved, gpu_tokens_saved) given per-prefix probabilities."""
    T = len(t.steps)
    for k, p in enumerate(probs, start=1):
        if p >= s or 1 - p >= f:
            saved = sum(call_tokens(t, j) for j in range(k - 1, T))
            gpu_saved = sum(call_gpu_tokens(t, j) for j in range(k - 1, T))
            return k, p >= s, saved, gpu_saved
    return None, t.solved, 0, 0


def _lcb(results):
    s = sum(results)
    return beta.ppf(LCB_EPS, 1 + s, 1 + len(results) - s)


def rank_checks(per_agent):
    """Agents scored with vs without early stopping: (Spearman of solve rates, same best by mean,
    same best by the Beta lower bound that picks the final agent). per_agent: {agent: [(pred, true)]}."""
    agents = [a for a, v in per_agent.items() if len(v) >= MIN_TRAJ_FOR_RANK]
    rho = same_mean = None
    if len(agents) >= 3:
        est = [np.mean([p for p, _ in per_agent[a]]) for a in agents]
        true = [np.mean([y for _, y in per_agent[a]]) for a in agents]
        rho = spearmanr(est, true).correlation
        same_mean = agents[int(np.argmax(est))] == agents[int(np.argmax(true))]
    # the lower bound already discounts agents with few tasks, so every agent takes part
    every = list(per_agent)
    est = [_lcb([p for p, _ in per_agent[a]]) for a in every]
    true = [_lcb([y for _, y in per_agent[a]]) for a in every]
    same_lcb = every[int(np.argmax(est))] == every[int(np.argmax(true))] if every else None
    return rho, same_mean, same_lcb


def make_splits(trajs, agents, split, folds):
    """[(train indices, test indices)], scheme description."""
    if len(agents) == 1:
        tasks = np.array([t.task for t in trajs])
        return list(GroupKFold(n_splits=folds).split(np.zeros(len(trajs)), groups=tasks)), \
            f"{folds}-fold, grouped by task"
    if split == "agent":
        groups = np.array([t.agent for t in trajs])
        k = min(folds, len(agents))
        return list(GroupKFold(n_splits=k).split(np.zeros(len(trajs)), groups=groups)), \
            f"{k}-fold, grouped by agent (may train on later agents)"
    blocks = np.array_split(np.array(sorted(agents)), folds)  # node ids sort by creation time
    block_of = {a: b for b, block in enumerate(blocks) for a in block}
    splits = []
    for b in range(1, len(blocks)):
        tr = [i for i, t in enumerate(trajs) if block_of[t.agent] < b]
        te = [i for i, t in enumerate(trajs) if block_of[t.agent] == b]
        if tr and te:
            splits.append((tr, te))
    return splits, f"chronological, {folds} blocks of agents; block 1 (the earliest) is training only"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="predictions directory, optionally DIR=AGENT_NAME")
    ap.add_argument("--run-dir", action="append", default=[],
                    help="a search run's output directory; every node's predictions are loaded, agent = node id")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split", choices=["time", "agent"], default="time")
    args = ap.parse_args()

    trajs = []
    for d in args.dirs:
        path, _, name = d.partition("=")
        trajs += load_predictions(path, agent=name or None)
    for r in args.run_dir:
        trajs += load_run(r)
    if not trajs:
        ap.error("no trajectories: pass predictions directories or --run-dir")
    empty = [t for t in trajs if not t.steps]
    trajs = [t for t in trajs if t.steps]
    agents = sorted({t.agent for t in trajs})
    base_rate = np.mean([t.solved for t in trajs])
    print(f"{len(trajs)} trajectories from {len(agents)} agent(s); {len(empty)} without steps dropped; "
          f"solved {base_rate:.1%} (always predicting 'unsolved' is {1 - base_rate:.1%} accurate)")
    missing_usage = sum(1 for t in trajs if sum(t.total_tokens) == 0)
    print(f"{missing_usage} trajectories have no token usage in their log (their savings count as 0)\n")

    splits, scheme = make_splits(trajs, agents, args.split, args.folds)
    folds = []
    for tr, te in splits:
        model = fit([trajs[i] for i in tr])
        # per-prefix probabilities are computed once and reused for every threshold pair
        folds.append([(i, probabilities(model, trajs[i]) if model else None) for i in te])
    tested = [i for fold in folds for i, _ in fold]

    # discrimination of the prefix classifier, per prefix length
    print(f"Prefix classifier ({scheme}): ROC AUC on held-out prefixes; {len(tested)} trajectories tested")
    by_k = {}
    for fold in folds:
        for i, probs in fold:
            for k, p in enumerate(probs or [], start=1):
                by_k.setdefault(min(k, 6), []).append((p, int(trajs[i].solved)))
    for k, pairs in sorted(by_k.items()):
        ps, ys = zip(*pairs)
        auc = roc_auc_score(ys, ps) if len(set(ys)) == 2 else float("nan")
        print(f"  k={'6+' if k == 6 else k}: AUC {auc:.3f} on {len(pairs)} prefixes")

    test_trajs = [trajs[i] for i in tested]
    test_rate = np.mean([t.solved for t in test_trajs])
    total_tokens = sum(sum(t.total_tokens) for t in test_trajs)  # includes the first call, which is never saved
    total_gpu = sum(gpu_tokens(t) for t in test_trajs)
    total_steps = sum(len(t.steps) for t in test_trajs)
    multi = len(agents) > 1
    print("\n| s | f | stopped early | stop accuracy | steps saved | tokens saved | GPU tokens saved | "
          "est. solve rate (true) |"
          + (" agent rank Spearman | same best (mean) | same best (Beta LCB) |" if multi else ""))
    print("| --- | --- | --- | --- | --- | --- | --- | --- |" + (" --- | --- | --- |" if multi else ""))
    yes_no = {True: "yes", False: "no", None: "—"}
    for s, f in itertools.product(S_THRESHOLDS, F_THRESHOLDS):
        stopped = correct = steps_saved = tokens_saved = gpu_saved = 0
        predicted, per_agent = [], {}
        for fold in folds:
            for i, probs in fold:
                t = trajs[i]
                k, pred, saved, g = simulate_one(probs, t, s, f) if probs is not None else (None, t.solved, 0, 0)
                predicted.append(pred)
                per_agent.setdefault(t.agent, []).append((pred, t.solved))
                if k is not None:
                    stopped += 1; correct += pred == t.solved
                    steps_saved += len(t.steps) - k; tokens_saved += saved; gpu_saved += g
        acc = f"{correct / stopped:.1%}" if stopped else "—"
        row = (f"| {s} | {f} | {stopped / len(test_trajs):.1%} | {acc} | {steps_saved / total_steps:.1%} | "
               f"{tokens_saved / max(total_tokens, 1):.1%} | {gpu_saved / max(total_gpu, 1):.1%} | "
               f"{np.mean(predicted):.1%} ({test_rate:.1%}) |")
        if multi:
            rho, same_mean, same_lcb = rank_checks(per_agent)
            row += f" {'—' if rho is None else f'{rho:.3f}'} | {yes_no[same_mean]} | {yes_no[same_lcb]} |"
        print(row)


if __name__ == "__main__":
    main()
