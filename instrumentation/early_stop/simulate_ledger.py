"""Early-stopping replay on the proxy's record of each evaluation, not the agent's own log.

Self-modifying agents rewrite how they log: some drop the "Tool Used" markers, some write
only the context left after their own pruning (a task with 82 calls in the ledger kept 13
steps in its log). Prefix features read from those logs are wrong exactly for the agents
that changed the most. The metering proxy records every model call of an evaluation
(node, task, time, latency, prompt / cached / output tokens, status) whatever the agent
does, so this replay builds trajectories and features from the ledger alone. Only the
outcome (solved or not) comes from the evaluation's <task>.json.

A trajectory is the sequence of eval calls c_0 .. c_{T-1} of one (node, task). Prefix k
(k = 1 .. T-1) means calls c_0 .. c_{k-1} are done; stopping there skips c_k .. c_{T-1}
and the test phase. Savings are measured, not estimated: their tokens (all, and the GPU
part: uncached input + output) and the evaluation time from the end of c_{k-1} to the end
of the agent run, plus the test phase.

Splits and the selection checks (Spearman, same best agent by mean and by the Beta lower
bound the experiments use) are those of simulate.py.

    python simulate_ledger.py --ledger runs/ledger.jsonl --run hgm-fixed-r0 --run-dir output_hgm/hgm-fixed-r0
"""

import argparse
import gzip
import itertools
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from simulate import F_THRESHOLDS, S_THRESHOLDS, make_splits, rank_checks

LANGUAGES = ("cpp", "go", "java", "javascript", "python", "rust")
SHORT_OUTPUT = 64  # tokens; a call this short is usually a bare tool call
FEATURE_NAMES = (
    ["k", "elapsed_s", "last_prompt", "prompt_growth", "last_prompt_delta", "out_so_far", "mean_out",
     "last_out", "short_out_share", "last_cached_share", "n_errors", "mean_latency", "last_latency"]
    + [f"lang_{l}" for l in LANGUAGES]
)


@dataclass
class LedgerTrajectory:
    agent: str              # node id (without the _<i> suffix the evaluation adds)
    task: str
    outcome: str
    calls: list = field(default_factory=list)   # dicts: ts, latency, inp, cached, out, ok
    agent_end: float = None  # end of the agent_run phase
    agent_wall: float = 0.0
    test_wall: float = 0.0

    @property
    def solved(self):
        return self.outcome == "resolved"

    def tokens(self, calls):
        return sum(c["inp"] + c["out"] for c in calls)

    def gpu_tokens(self, calls):
        return sum(c["inp"] - c["cached"] + c["out"] for c in calls)


def canonical(node):
    return re.sub(r"_\d+$", "", node or "")


def load_ledger(path, run):
    """{(node, task): {"calls": [...], "agent_end", "agent_wall", "test_wall"}} for one run's evaluations."""
    out = defaultdict(lambda: {"calls": [], "agent_end": None, "agent_wall": 0.0, "test_wall": 0.0})
    tag = f'"run": "{run}"'
    with open(path) as f:
        for line in f:
            if tag not in line:
                continue
            e = json.loads(line)
            if e.get("run") != run or not e.get("task") or e.get("node") in (None, "initial"):
                continue
            key = (canonical(e["node"]), e["task"])
            if e["type"] == "llm_call" and e.get("tag") == "eval":
                out[key]["calls"].append({
                    "ts": e["ts"], "latency": e.get("latency_s") or 0.0, "inp": e.get("input_tokens") or 0,
                    "cached": e.get("cached_tokens") or 0, "out": e.get("output_tokens") or 0,
                    "ok": e.get("status", 200) == 200})
            elif e["type"] == "eval" and e.get("phase") == "agent_run":
                out[key]["agent_end"], out[key]["agent_wall"] = e["ts"], e.get("wall_s") or 0.0
            elif e["type"] == "eval" and e.get("phase") == "test":
                out[key]["test_wall"] = e.get("wall_s") or 0.0
    for v in out.values():
        v["calls"].sort(key=lambda c: c["ts"])
    return out


def _read_json(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def load_outcomes(run_dir):
    """{(node, task): eval_result} from <run_dir>/<node>/predictions/<node>_<i>/<task>.json[.gz]."""
    out = {}
    for node in sorted(os.listdir(run_dir)):
        pred_root = os.path.join(run_dir, node, "predictions")
        if not os.path.isdir(pred_root):
            continue
        for sub in os.listdir(pred_root):
            d = os.path.join(pred_root, sub)
            if not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                if name.endswith((".json", ".json.gz")) and "_eval" not in name:
                    task = name[:-len(".json.gz")] if name.endswith(".gz") else name[:-len(".json")]
                    try:
                        out[(node, task)] = _read_json(os.path.join(d, name)).get("eval_result", "")
                    except (OSError, ValueError):
                        continue
    return out


def build(ledger_data, outcomes):
    trajs, no_calls = [], 0
    for (node, task), outcome in sorted(outcomes.items()):
        rec = ledger_data.get((node, task))
        if not rec or not rec["calls"]:
            no_calls += 1
            continue
        trajs.append(LedgerTrajectory(node, task, outcome, rec["calls"], rec["agent_end"],
                                      rec["agent_wall"], rec["test_wall"]))
    return trajs, no_calls


def prefix_features(t, k):
    """Features of calls c_0 .. c_{k-1}: only what the proxy has seen by then."""
    cs = t.calls[:k]
    start = cs[0]["ts"] - cs[0]["latency"]
    outs = [c["out"] for c in cs]
    last = cs[-1]
    lang = t.task.split("__")[0]
    return [
        k, last["ts"] - start, last["inp"], last["inp"] - cs[0]["inp"],
        last["inp"] - cs[-2]["inp"] if k > 1 else 0, sum(outs), sum(outs) / k, last["out"],
        sum(o < SHORT_OUTPUT for o in outs) / k, last["cached"] / last["inp"] if last["inp"] else 0.0,
        sum(not c["ok"] for c in cs), sum(c["latency"] for c in cs) / k, last["latency"],
    ] + [int(lang == l) for l in LANGUAGES]


def fit(trajs):
    X, y, w, g = [], [], [], []
    for i, t in enumerate(trajs):
        T = len(t.calls)
        for k in range(1, T):
            X.append(prefix_features(t, k)); y.append(int(t.solved)); w.append(1.0 / (T - 1)); g.append(i)
    if len(set(y)) < 2:
        return None
    X, y, w, g = np.array(X, float), np.array(y), np.array(w), np.array(g)
    base = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=10, random_state=0)
    folds = list(GroupKFold(n_splits=3).split(X, y, groups=g))  # calibration grouped by trajectory
    model = CalibratedClassifierCV(base, method="sigmoid", cv=folds)
    model.fit(X, y, sample_weight=w)
    return model


def probabilities(model, t):
    T = len(t.calls)
    if T < 2:
        return []
    return list(model.predict_proba(np.array([prefix_features(t, k) for k in range(1, T)], float))[:, 1])


def eval_seconds(t):
    return t.agent_wall + t.test_wall


def simulate_one(probs, t, s, f):
    """(stop_k or None, predicted_solved, tokens saved, GPU tokens saved, seconds saved)."""
    for k, p in enumerate(probs, start=1):
        if p >= s or 1 - p >= f:
            rest = t.calls[k:]
            end = t.agent_end if t.agent_end is not None else t.calls[-1]["ts"]
            seconds = max(0.0, end - t.calls[k - 1]["ts"]) + t.test_wall
            return k, p >= s, t.tokens(rest), t.gpu_tokens(rest), seconds
    return None, t.solved, 0, 0, 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--run", required=True, help="RSI_RUN label in the ledger")
    ap.add_argument("--run-dir", required=True, help="the run's output directory (for the outcomes)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split", choices=["time", "agent"], default="time")
    args = ap.parse_args()

    trajs, no_calls = build(load_ledger(args.ledger, args.run), load_outcomes(args.run_dir))
    short = [t for t in trajs if len(t.calls) < 2]
    agents = sorted({t.agent for t in trajs})
    rate = np.mean([t.solved for t in trajs])
    print(f"{len(trajs)} evaluations with ledger calls from {len(agents)} agents ({no_calls} outcomes without calls, "
          f"{len(short)} with one call and nothing to stop); solved {rate:.1%}")
    print(f"calls per evaluation: median {np.median([len(t.calls) for t in trajs]):.0f}, "
          f"max {max(len(t.calls) for t in trajs)}\n")

    splits, scheme = make_splits(trajs, agents, args.split, args.folds)
    folds = []
    for tr, te in splits:
        model = fit([trajs[i] for i in tr])
        folds.append([(i, probabilities(model, trajs[i]) if model else None) for i in te])
    tested = [trajs[i] for fold in folds for i, _ in fold]

    print(f"Prefix classifier ({scheme}): ROC AUC on held-out prefixes; {len(tested)} evaluations tested")
    buckets = [(str(k), k, k) for k in range(1, 6)] + [("6-10", 6, 10), ("11-30", 11, 30), ("31+", 31, 10 ** 9)]
    by_bucket = defaultdict(list)
    for fold in folds:
        for i, probs in fold:
            for k, p in enumerate(probs or [], start=1):
                name = next(b for b, lo, hi in buckets if lo <= k <= hi)
                by_bucket[name].append((p, int(trajs[i].solved)))
    for name, _, _ in buckets:
        if by_bucket[name]:
            ps, ys = zip(*by_bucket[name])
            auc = roc_auc_score(ys, ps) if len(set(ys)) == 2 else float("nan")
            print(f"  calls done = {name}: AUC {auc:.3f} on {len(ps)} prefixes")

    tot_tok = sum(t.tokens(t.calls) for t in tested)
    tot_gpu = sum(t.gpu_tokens(t.calls) for t in tested)
    tot_sec = sum(eval_seconds(t) for t in tested)
    tot_calls = sum(len(t.calls) for t in tested)
    test_rate = np.mean([t.solved for t in tested])
    print("\n| s | f | stopped early | stop accuracy | calls saved | tokens saved | GPU tokens saved | "
          "evaluation time saved | est. solve rate (true) | agent rank Spearman | same best (mean) | same best (Beta LCB) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    yes_no = {True: "yes", False: "no", None: "—"}
    for s, f in itertools.product(S_THRESHOLDS, F_THRESHOLDS):
        stopped = correct = calls_saved = tok = gpu = sec = 0
        predicted, per_agent = [], {}
        for fold in folds:
            for i, probs in fold:
                t = trajs[i]
                k, pred, a, b, c = simulate_one(probs, t, s, f) if probs is not None else (None, t.solved, 0, 0, 0.0)
                predicted.append(pred)
                per_agent.setdefault(t.agent, []).append((pred, t.solved))
                if k is not None:
                    stopped += 1; correct += pred == t.solved
                    calls_saved += len(t.calls) - k; tok += a; gpu += b; sec += c
        rho, same_mean, same_lcb = rank_checks(per_agent)
        acc = f"{correct / stopped:.1%}" if stopped else "—"
        print(f"| {s} | {f} | {stopped / len(tested):.1%} | {acc} | {calls_saved / tot_calls:.1%} | "
              f"{tok / max(tot_tok, 1):.1%} | {gpu / max(tot_gpu, 1):.1%} | {sec / max(tot_sec, 1):.1%} | "
              f"{np.mean(predicted):.1%} ({test_rate:.1%}) | {'—' if rho is None else f'{rho:.3f}'} | "
              f"{yes_no[same_mean]} | {yes_no[same_lcb]} |")


if __name__ == "__main__":
    main()
