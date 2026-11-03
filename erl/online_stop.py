"""Prediction-driven node stopping inside a live HGM search (the online form of predict_stop.py).

hgm.py calls, under its own lock:
    node_created(commit, parent_commit, diff_path)   after a child is proposed
    allow(commit)                                    before choosing a node to evaluate
    result(commit, scores, wall_s)                   after each evaluation returns

allow() is False for a child whose Beta upper bound (1 - alpha), with the predictor's
prior of weight w added, is below the incumbent's lower bound (eps); such a child is
retired for good and gets no more evaluations. It stays in the tree, so HGM can still pick
it as a parent. The root ("initial") is never retired. The predictor and the automatic
weight are those of predict_stop.py, refitted every `refit` results on what this search has
observed, plus the history rows written by `python online_stop.py dump-history`.

Audits (audit_p > 0, off by default): a stop decision is, for a fixed share of children
(chosen by a hash of the id, so a rerun audits the same ones), turned into audit_n more
evaluations instead. The audit then asks the data alone (no prior) whether the child could
still compete: with audit_rule "mean" (default) the stop was false if the child's posterior
mean (1 + solved) / (2 + n) is at least the incumbent's solve rate; with "ucb" if its plain
Beta upper bound is back above the incumbent's lower bound (this fires on noise: a stop
happens right at that bound, so a few more tasks often push it back over, and offline it
called 95% of the plain statistical stops false). A confirmed stop retires the child; a
false one lets it go on as usual. The share of false stops is the online
check on the predictor that the offline replay cannot give. Once audit_min audits are in
and more than audit_max_false of them were false, the cap on w is halved (and again at each
later audit while the rate stays above it): the search rolls back towards the plain
statistical policy when the predictor is stopping children it should not.

Every decision is logged to the ledger (erl_retire, erl_fit, erl_audit_start, erl_audit_end). Any error inside the stopper
is logged and answered with "evaluate", so a bug here cannot stop the search.

Enable in hgm.py with HGM_ERL_STOP=1; HGM_ERL_HISTORY points to the history rows; HGM_ERL_AUDIT
sets audit_p (default 0) and HGM_ERL_AUDIT_N audit_n (default 5).

    python online_stop.py dump-history --ledger ../runs/ledger.jsonl \
        --prescreen-rows ../runs/prescreen_rows.jsonl --runs hgm-fixed-r0 ... --out ../runs/erl_history.json
"""

import argparse
import hashlib
import json
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "instrumentation"))
sys.path.insert(0, os.path.join(HERE, "..", "instrumentation", "prescreen"))

import predict_stop as ps  # noqa: E402
from policy import beta_quantile  # noqa: E402


class OnlineStopper:
    def __init__(self, alpha=0.2, min_evals=3, w="auto", refit=50, min_label=5, eps=0.05,
                 history_path=None, log=None, audit_p=0.0, audit_n=5, audit_min=5, audit_max_false=0.2,
                 audit_rule="mean"):
        self.alpha, self.min_evals, self.w, self.refit, self.min_label, self.eps = \
            alpha, min_evals, w, refit, min_label, eps
        self.log = log or (lambda *a, **k: None)
        self.states, self.depth, self.lcb = {}, {"initial": 0}, {}
        self.retired, self.first_pred = set(), {}
        self.model, self.since_fit = None, 0
        self.w_now = 0.0 if w == "auto" else float(w)
        self.audit_p, self.audit_n, self.audit_min, self.audit_max_false = audit_p, audit_n, audit_min, audit_max_false
        self.audit_rule = audit_rule
        self.auditing = {}            # commit -> number of results when its audit started
        self.first_audit = set()      # children audited once (a revived child is not audited again)
        self.audits = []              # True for each audit that found a false stop
        self.w_cap = ps.W_MAX if w == "auto" else float(w)
        self.hist = ([], [], [])
        if history_path:
            with open(history_path) as f:
                h = json.load(f)
            self.hist = (h["X"], h["y"], h["w"])

    # --- hooks -------------------------------------------------------------------------
    def seed(self, commit, parent_commit, results):
        """A node that already has results when the search starts (the root, or a continued run)."""
        self._guard(self._seed, commit, parent_commit, results)

    def node_created(self, commit, parent_commit, diff_path=None):
        self._guard(self._created, commit, parent_commit, diff_path)

    def allow(self, commit):
        out = self._guard(self._allow, commit)
        return True if out is None else out

    def result(self, commit, scores, wall_s):
        self._guard(self._result, commit, scores, wall_s)

    # --- internals ---------------------------------------------------------------------
    def _guard(self, fn, *args):
        try:
            return fn(*args)
        except Exception as e:   # never let the stopper break the search
            self.log("erl_error", where=fn.__name__, error=repr(e), trace=traceback.format_exc()[-2000:])
            return None

    def _state(self, commit, parent_commit=None, diff=None):
        if commit not in self.states:
            p = self.states.get(parent_commit)
            d = 0 if parent_commit is None else self.depth.get(parent_commit, 0) + 1
            self.depth[commit] = d
            static = {**{k: (diff or {}).get(k, 0) for k in ps.DIFF_KEYS}, "depth": d, "parent": parent_commit,
                      "parent_rate": (sum(p.results) / len(p.results)) if p and p.results else 0.0,
                      "parent_n": len(p.results) if p else 0}
            self.states[commit] = ps.NodeState(static)
        return self.states[commit]

    def _seed(self, commit, parent_commit, results):
        st = self._state(commit, parent_commit)
        st.results.extend(results)
        st.costs.extend([0.0] * len(results))
        self.since_fit += len(results)   # as the replay counts the root's cached results
        if st.results:
            self.lcb[commit] = beta_quantile(self.eps, sum(st.results), len(st.results))

    def _created(self, commit, parent_commit, diff_path):
        diff = None
        if diff_path and os.path.exists(diff_path):
            from predict_from_diff import diff_features
            with open(diff_path, errors="ignore") as f:
                diff = diff_features(f.read())
        self._state(commit, parent_commit, diff)

    def _audited(self, commit):
        h = int(hashlib.sha1(commit.encode()).hexdigest()[:8], 16) / 16 ** 8
        return h < self.audit_p

    def _end_audit(self, commit, st, incumbent):
        """Called once an audited child has audit_n more results: confirm or revive."""
        start = self.auditing.pop(commit)
        k, s = len(st.results), sum(st.results)
        ucb = beta_quantile(1 - self.alpha, s, k)
        others = {n: v for n, v in self.lcb.items() if n != commit}
        best = max(others, key=others.get)    # the incumbent among the other nodes
        b = self.states[best].results
        best_rate = sum(b) / len(b)
        if self.audit_rule == "ucb":
            false_stop = ucb >= incumbent      # the data alone would not stop it (sensitive to noise near the bound)
        else:
            false_stop = (1 + s) / (2 + k) >= best_rate   # it could compete with the incumbent
        self.audits.append(false_stop)
        rate = sum(self.audits) / len(self.audits)
        if len(self.audits) >= self.audit_min and rate > self.audit_max_false:
            self.w_cap /= 2
            self.w_now = min(self.w_now, self.w_cap)
        self.log("erl_audit_end", node=commit, k_start=start, k=k, solved=s, ucb=ucb, incumbent=incumbent,
                 best_rate=best_rate, rule=self.audit_rule,
                 false_stop=false_stop, n_audits=len(self.audits), false_rate=rate, w_cap=self.w_cap)
        if false_stop:
            return True
        self.retired.add(commit)
        return False

    def _allow(self, commit):
        if commit == "initial":
            return True
        if commit in self.retired:
            return False
        st = self._state(commit)
        k, s = len(st.results), sum(st.results)
        incumbent = max(self.lcb.values(), default=0.0)
        if commit in self.auditing:
            if k - self.auditing[commit] < self.audit_n:
                return True
            return self._end_audit(commit, st, incumbent)
        if k < self.min_evals:
            return True
        a, b = 1 + s, 1 + k - s
        p_hat = None
        if self.model is not None:
            p_hat = float(np.clip(self.model.predict(np.array([st.features()], float))[0], 0.01, 0.99))
            self.first_pred.setdefault(commit, (p_hat, k))
            a, b = a + self.w_now * p_hat, b + self.w_now * (1 - p_hat)
        from scipy.stats import beta
        ucb = float(beta.ppf(1 - self.alpha, a, b))
        if ucb >= incumbent:
            return True
        if self.audit_p > 0 and self._audited(commit) and commit not in self.first_audit:
            self.first_audit.add(commit)
            self.auditing[commit] = k
            self.log("erl_audit_start", node=commit, k=k, solved=s, p_hat=p_hat, w=self.w_now, ucb=ucb,
                     incumbent=incumbent)
            return True
        self.retired.add(commit)
        self.log("erl_retire", node=commit, k=k, solved=s, p_hat=p_hat, w=self.w_now, ucb=ucb,
                 incumbent=incumbent)
        return False

    def _result(self, commit, scores, wall_s):
        st = self._state(commit)
        per = wall_s / len(scores) if scores else 0.0
        st.results.extend(scores)
        st.costs.extend([per] * len(scores))
        self.lcb[commit] = beta_quantile(self.eps, sum(st.results), len(st.results))
        self.since_fit += len(scores)
        if self.since_fit >= self.refit:
            self.since_fit = 0
            X, y, sw = ps.training_rows([v for n, v in self.states.items() if n != "initial"], self.min_label)
            hX, hy, hw = self.hist
            self.model = ps.fit(hX + X, hy + y, hw + sw) or self.model
            if self.w == "auto":
                self.w_now = min(ps.auto_weight(self.first_pred, self.states, self.min_label), self.w_cap)
            self.log("erl_fit", rows=len(X), history_rows=len(hX), w=self.w_now, retired=len(self.retired),
                     fitted=self.model is not None)


def from_env(log=None):
    """The stopper hgm.py uses when HGM_ERL_STOP=1, else None."""
    if os.getenv("HGM_ERL_STOP", "0") != "1":
        return None
    w = os.getenv("HGM_ERL_W", "auto")
    return OnlineStopper(alpha=float(os.getenv("HGM_ERL_ALPHA", "0.2")),
                         min_evals=int(os.getenv("HGM_ERL_MIN_EVALS", "3")),
                         w=w if w == "auto" else float(w),
                         history_path=os.getenv("HGM_ERL_HISTORY") or None, log=log,
                         audit_p=float(os.getenv("HGM_ERL_AUDIT", "0")),
                         audit_n=int(os.getenv("HGM_ERL_AUDIT_N", "5")))


def dump_history(args):
    """Training rows from finished runs' full records (cut at the common GPU hours)."""
    import analyze_ledger
    import replay as rp
    diff_rows = {}
    with open(args.prescreen_rows) as f:
        for line in f:
            r = json.loads(line)
            diff_rows[(r["run"], r["node"])] = r
    by_run = rp.load(args.ledger, args.runs)
    X, y, w = [], [], []
    for run in args.runs:
        events, _ = analyze_ledger.truncate(by_run[run], max_hours=args.max_hours)
        diffs = {n: r for (rr, n), r in diff_rows.items() if rr == run}
        _, states = ps.replay(events, rp.result_stream(events), diffs, 0, 1e-9, 10 ** 9, 10 ** 9, 1)
        a, b, c = ps.training_rows([v for n, v in states.items() if n != "initial"], 1)
        X += a; y += b; w += c
        print(f"{run}: {len(a)} rows")
    with open(args.out, "w") as f:
        json.dump({"X": X, "y": y, "w": w, "features": list(ps.FEATURES), "runs": args.runs}, f)
    print(f"wrote {len(X)} rows to {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump-history")
    d.add_argument("--ledger", required=True)
    d.add_argument("--prescreen-rows", required=True)
    d.add_argument("--runs", nargs="+", required=True)
    d.add_argument("--max-hours", type=float, default=9.646)
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    dump_history(args)


if __name__ == "__main__":
    main()
