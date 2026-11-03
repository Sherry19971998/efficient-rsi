"""Final-agent selection and bootstrap confidence intervals.

Selection rule (same for every method): after the search, at the equal-budget cut-off,
pick the node whose own search-task results have the highest lower eps-quantile of a
Beta(1 + solved, 1 + unsolved) posterior (eps = 0.05 by default). Using a lower bound
rather than the mean keeps a node that was lucky on a handful of tasks from beating a
node with a solid record. The rule is ours and is disclosed in the report; `--rule mean`
gives the plain mean for a sensitivity check.

Bootstrap: seeds and held-out tasks are both resampled (a two-level bootstrap). For a
difference between two configurations, the task resample is shared, because both are
scored on the same held-out tasks, while seeds are resampled independently.
"""

import numpy as np
from scipy.stats import beta

from analyze_ledger import truncate


def search_scores(events, max_evals=None, max_hours=None):
    """node -> list of 0/1 results on search tasks, up to the equal-budget cut-off."""
    events, _ = truncate(events, max_evals, max_hours)
    scores = {}
    for e in events:
        if e["type"] == "eval_result" and e.get("node"):
            scores.setdefault(e["node"], []).extend(int(x) for x in (e.get("score") or []))
    return scores


def beta_lcb(results, eps=0.05):
    s = sum(results)
    return float(beta.ppf(eps, 1 + s, 1 + len(results) - s))


def select_best(scores, rule="beta_lcb", eps=0.05):
    """Node with the best score under the rule; ties go to the node with more evaluations."""
    def key(item):
        node, res = item
        value = beta_lcb(res, eps) if rule == "beta_lcb" else (np.mean(res) if res else 0.0)
        return (value, len(res))
    candidates = [(n, r) for n, r in scores.items() if r]
    if not candidates:
        raise ValueError("no node has search results")
    return max(candidates, key=key)[0]


def _matrix(runs, tasks):
    """runs: list of {task: 0/1}; returns seeds x tasks array (tasks in the given order)."""
    return np.array([[r[t] for t in tasks] for r in runs], float)


def bootstrap_mean(runs, tasks, n_boot=10000, seed=0, level=0.95):
    """Mean held-out accuracy of one configuration over seeds, with a two-level bootstrap CI."""
    m = _matrix(runs, tasks)
    rng = np.random.default_rng(seed)
    S, T = m.shape
    stats = np.empty(n_boot)
    for b in range(n_boot):
        rows = rng.integers(0, S, S)
        cols = rng.integers(0, T, T)
        stats[b] = m[np.ix_(rows, cols)].mean()
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"mean": float(m.mean()), "ci_low": float(lo), "ci_high": float(hi), "seeds": S, "tasks": T}


def bootstrap_diff(runs_a, runs_b, tasks, n_boot=10000, seed=0, level=0.95):
    """Accuracy of A minus B on the same held-out tasks, with a CI and P(A > B)."""
    a, b = _matrix(runs_a, tasks), _matrix(runs_b, tasks)
    rng = np.random.default_rng(seed)
    T = len(tasks)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        cols = rng.integers(0, T, T)  # shared: both configurations face the same tasks
        ra = rng.integers(0, a.shape[0], a.shape[0])
        rb = rng.integers(0, b.shape[0], b.shape[0])
        stats[i] = a[np.ix_(ra, cols)].mean() - b[np.ix_(rb, cols)].mean()
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"diff": float(a.mean() - b.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_a_better": float((stats > 0).mean())}
