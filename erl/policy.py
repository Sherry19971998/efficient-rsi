"""Stopping policies: before each evaluation of a node, decide whether it is still worth running.

A policy sees only what the search has observed so far: each node's results on search
tasks, in the order they arrived. It answers one question per pending evaluation:
evaluate this node now, or retire it (skip this and every later evaluation of it).

The selection rule the experiments use is the Beta lower bound (results.beta_lcb, eps 0.05):
the final agent is the node with the highest lower bound. A node can only be selected if
its lower bound ends above the incumbent's, and its lower bound stays below its true solve
rate with high probability. So once a node's upper bound falls below the incumbent's lower
bound, further evaluations of it are unlikely to change the selection. That is the
decision-aware threshold: how much evidence a node needs depends on how far it is from the
current best, not on a fixed number of tasks.
"""

from functools import lru_cache

from scipy.stats import beta


@lru_cache(maxsize=None)
def beta_quantile(q, solved, n):
    """q-quantile of the Beta(1 + solved, 1 + n - solved) posterior; cached, since replays ask
    for the same few hundred (solved, n) pairs millions of times."""
    return float(beta.ppf(q, 1 + solved, 1 + n - solved))


def beta_ucb(results, alpha):
    return beta_quantile(1 - alpha, sum(results), len(results))


class Policy:
    name = "none"

    def evaluate(self, node, results, incumbent_lcb):
        """True to run the next evaluation of node; False retires it.

        results: node's observed 0/1 results so far; incumbent_lcb: the best Beta lower
        bound over all nodes right now (the score the final selection maximises).
        """
        return True


class FixedCap(Policy):
    """Evaluate each node on at most `cap` tasks: the static baseline."""

    def __init__(self, cap):
        self.cap = cap
        self.name = f"cap{cap}"

    def evaluate(self, node, results, incumbent_lcb):
        return len(results) < self.cap


class UcbVsIncumbent(Policy):
    """Retire a node once its (1 - alpha) upper bound is below the incumbent's lower bound.

    min_evals keeps a node from being retired on its first one or two results, when a
    single failure says little; the root ("initial") is never retired because every
    method's incumbent starts there.
    """

    def __init__(self, alpha, min_evals=3):
        self.alpha = alpha
        self.min_evals = min_evals
        self.name = f"ucb{alpha:g}-min{min_evals}"

    def evaluate(self, node, results, incumbent_lcb):
        if node == "initial" or len(results) < self.min_evals:
            return True
        return beta_ucb(results, self.alpha) >= incumbent_lcb
