"""Predict a child agent's evaluation from what is known before it is evaluated.

Early stopping (early_stop/) saves time inside an evaluation. This replay asks whether some
evaluations need not start at all: when a child is created we already know its parent's
record and the patch that made it. If those predict that the child is broken (scores 0) or
will not beat its parent, the search could skip or shorten its evaluation.

Per child node (every run, initial agent excluded), features use only pre-evaluation data:
  - the patch (model_patch.diff, the child's own change on top of its parent): files and
    lines touched, which parts of the agent (main loop, LLM client, tools, prompts), new
    files, removed functions, strings that point at other model providers;
  - the parent: accuracy and number of evaluations so far, depth in the tree;
  - the proposal: language of the task it learned from, length of the diagnosis, time spent.
Targets come from the ledger: broken (accuracy 0) and improves (accuracy > parent's).

Evaluation is leave-one-run-out: train on 8 runs, predict the 9th, so the predictor never
sees the run it scores (the shift a live predictor would face). Reported: ROC AUC against
simple baselines, and for skip thresholds the evaluation time saved, how many skipped nodes
were really broken, how many improving nodes were lost, and whether any ancestor of the
run's finally selected agent would have been skipped.

    python predict_from_diff.py --ledger ../../runs/ledger.jsonl --output-root ../../repos/HGM/output_hgm
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analyze_ledger import build  # noqa: E402

RUNS = [f"{c}-r{r}" for r in (0, 1, 2) for c in ("hgm-fixed", "hgm-orig", "dgm")]
LANGUAGES = ("cpp", "go", "java", "javascript", "python", "rust")
SKIP_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
# Events the replay needs; model calls are by far the bulk of the ledger and are not read.
KEEP_TYPES = ("node_created", "eval_result", "node_evaluated", "eval", "propose", "select_parent")
JUNK = re.compile(r"(__pycache__/|\.pyc$|^self_evo\.md$|\.log$)")
PROVIDER = re.compile(r"gpt-|openai|openrouter|anthropic|claude|gemini", re.I)
MAIN_FILES = ("coding_agent.py", "coding_agent_polyglot.py")
LLM_FILES = ("llm.py", "llm_withtools.py")
DIFF_FEATURES = ["n_files", "n_py", "n_text", "added", "removed", "net", "new_files", "touches_main",
                 "touches_llm", "touches_tools", "main_removed", "removed_defs", "added_defs",
                 "provider_strings", "empty"]
NODE_FEATURES = ["parent_acc", "parent_evals", "depth", "propose_wall", "diagnosis_len"]
FEATURE_NAMES = DIFF_FEATURES + NODE_FEATURES + [f"lang_{l}" for l in LANGUAGES]


def load_events(path, runs):
    """run -> its events of the types the replay needs, read in one pass."""
    out = defaultdict(list)
    wanted = set(runs)
    with open(path) as f:
        for line in f:
            if '"llm_call"' in line or '"gpu_sample"' in line:
                continue
            e = json.loads(line)
            if e.get("run") in wanted and e.get("type") in KEEP_TYPES:
                out[e["run"]].append(e)
    return out


def diff_features(text):
    """Counts over the child's own patch, ignoring caches, logs and the chat transcript."""
    f = dict.fromkeys(DIFF_FEATURES, 0)
    path = None
    for line in text.splitlines():
        if line.startswith("diff --git"):
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            path = m.group(2) if m else None
            if path and JUNK.search(path):
                path = None
                continue
            if path:
                f["n_files"] += 1
                f["n_py"] += path.endswith(".py")
                f["n_text"] += path.endswith((".md", ".txt", ".json", ".yaml", ".yml"))
                base = os.path.basename(path)
                f["touches_main"] |= base in MAIN_FILES
                f["touches_llm"] |= base in LLM_FILES
                f["touches_tools"] |= path.startswith("tools/")
            continue
        if path is None:
            continue
        if line.startswith("new file mode"):
            f["new_files"] += 1
        elif line.startswith("+") and not line.startswith("+++"):
            f["added"] += 1
            f["added_defs"] += line[1:].lstrip().startswith("def ")
            f["provider_strings"] += bool(PROVIDER.search(line))
        elif line.startswith("-") and not line.startswith("---"):
            f["removed"] += 1
            f["removed_defs"] += line[1:].lstrip().startswith("def ")
            f["main_removed"] += os.path.basename(path) in MAIN_FILES
    f["net"] = f["added"] - f["removed"]
    f["empty"] = int(f["n_files"] == 0)
    return f


def _read(path):
    try:
        with open(path, errors="ignore") as fh:
            return fh.read()
    except OSError:
        return ""


def node_rows(run, events, run_dir):
    """One dict per evaluated child: features, targets and its evaluation wall time."""
    nodes = build(events)[0]
    rows = []
    for nid, nd in nodes.items():
        if nid == "initial" or not nd.parent or nd.acc is None or not nd.n_evals:
            continue
        parent = nodes.get(nd.parent)
        if parent is None or parent.acc is None:
            continue
        depth, p = 1, parent
        while p is not None and p.parent:
            depth, p = depth + 1, nodes.get(p.parent)
        meta = {}
        try:
            meta = json.loads(_read(os.path.join(run_dir, nid, "metadata.json")) or "{}")
        except ValueError:
            pass
        entry = meta.get("entry") or ""
        row = diff_features(_read(os.path.join(run_dir, nid, "model_patch.diff")))
        # The parent's accuracy as known when the child was created: its own record (initial: cached).
        row.update(parent_acc=parent.acc, parent_evals=parent.n_evals, depth=depth,
                   propose_wall=nd.propose_wall, diagnosis_len=len(meta.get("problem_statement") or ""))
        for lang in LANGUAGES:
            row[f"lang_{lang}"] = int(entry.startswith(lang))
        row.update(run=run, node=nid, acc=nd.acc, broken=int(nd.acc == 0),
                   improves=int(nd.acc > parent.acc), eval_wall=nd.eval_wall, parent=nd.parent)
        rows.append(row)
    return rows, nodes


def ancestors(nodes, nid):
    out = set()
    while nid in nodes and nodes[nid].parent:
        nid = nodes[nid].parent
        out.add(nid)
    return out


def leave_one_run_out(rows, target):
    """Out-of-run probability of `target` for every row."""
    X = np.array([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r[target] for r in rows])
    runs = np.array([r["run"] for r in rows])
    p = np.full(len(rows), np.nan)
    for run in sorted(set(runs)):
        train, test = runs != run, runs == run
        if len(set(y[train])) < 2:
            continue
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                               min_samples_leaf=10, random_state=0)
        model.fit(X[train], y[train])
        p[test] = model.predict_proba(X[test])[:, 1]
    return p


def auc(y, score):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    ok = ~np.isnan(score)
    if len(set(y[ok])) < 2:
        return float("nan")
    return roc_auc_score(y[ok], score[ok])


def skip_table(rows, p, best_lineage, label):
    """What skipping every child with predicted P(broken) >= t would have cost and saved."""
    wall = np.array([r["eval_wall"] for r in rows])
    broken = np.array([r["broken"] for r in rows])
    improves = np.array([r["improves"] for r in rows])
    lines = [f"\n### Skip children with P({label}) >= t (leave-one-run-out)\n",
             "| t | children skipped | of them broken | improving children lost | eval time saved "
             "| runs whose selected agent's lineage is hit |",
             "| --- | --- | --- | --- | --- | --- |"]
    for t in SKIP_THRESHOLDS:
        skip = p >= t
        hit = sum(1 for run, lineage in best_lineage.items()
                  if any(s and r["run"] == run and r["node"] in lineage for s, r in zip(skip, rows)))
        n = int(skip.sum())
        lines.append(f"| {t} | {n} / {len(rows)} | {int(broken[skip].sum())} / {n} "
                     f"| {int(improves[skip].sum())} / {int(improves.sum())} "
                     f"| {100 * wall[skip].sum() / max(wall.sum(), 1e-9):.1f}% | {hit} / {len(best_lineage)} |")
    return lines


def report(rows, best_lineage):
    y_b = [r["broken"] for r in rows]
    y_i = [r["improves"] for r in rows]
    p_b = leave_one_run_out(rows, "broken")
    p_i = leave_one_run_out(rows, "improves")
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)

    lines = [f"{len(rows)} evaluated children in {len(by_run)} runs; "
             f"broken {100 * np.mean(y_b):.1f}%, improve on their parent {100 * np.mean(y_i):.1f}%; "
             f"they used {sum(r['eval_wall'] for r in rows) / 3600:.1f} h of evaluation wall time",
             "", "| run | children | broken | improve | AUC broken | AUC improves |", "| --- | --- | --- | --- | --- | --- |"]
    idx = defaultdict(list)
    for i, r in enumerate(rows):
        idx[r["run"]].append(i)
    for run in RUNS:
        if run not in idx:
            continue
        ii = idx[run]
        lines.append(f"| {run} | {len(ii)} | {sum(y_b[i] for i in ii)} | {sum(y_i[i] for i in ii)} "
                     f"| {auc([y_b[i] for i in ii], p_b[ii]):.2f} | {auc([y_i[i] for i in ii], p_i[ii]):.2f} |")

    # Baselines a search could use for free.
    parent = [r["parent_acc"] for r in rows]
    size = [r["added"] + r["removed"] for r in rows]
    lines += ["", "ROC AUC over all children (leave-one-run-out for the model):", "",
              "| predictor | broken | improves |", "| --- | --- | --- |",
              f"| model (patch + parent + proposal) | {auc(y_b, p_b):.3f} | {auc(y_i, p_i):.3f} |",
              f"| parent accuracy alone (higher = less broken, fewer improve) | {auc(y_b, [-x for x in parent]):.3f} "
              f"| {auc(y_i, [-x for x in parent]):.3f} |",
              f"| patch size alone (lines changed) | {auc(y_b, size):.3f} | {auc(y_i, [-x for x in size]):.3f} |",
              f"| patch removes from the main file | {auc(y_b, [r['main_removed'] for r in rows]):.3f} | — |",
              f"| patch mentions other providers | {auc(y_b, [r['provider_strings'] for r in rows]):.3f} | — |"]
    lines += skip_table(rows, p_b, best_lineage, "broken")
    lines += skip_table(rows, 1 - p_i, best_lineage, "no improvement")
    lines.append("\n'lineage is hit' counts runs where an ancestor of the agent final_eval selected (or the "
                 "agent itself) would have been skipped: the search might never have reached it.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--output-root", required=True, help="directory holding output_hgm/<run>/")
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--rows-out", help="optionally write the feature rows as JSON lines")
    args = ap.parse_args()

    events = load_events(args.ledger, args.runs)
    rows, best_lineage = [], {}
    for run in args.runs:
        run_dir = os.path.join(args.output_root, run)
        r, nodes = node_rows(run, events.get(run, []), run_dir)
        rows += r
        final = os.path.join(run_dir, "final_eval", "final_result.json")
        if os.path.exists(final):
            best = json.loads(_read(final))["best_node"]
            best_lineage[run] = ancestors(nodes, best) | {best}
    if args.rows_out:
        with open(args.rows_out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(report(rows, best_lineage))


if __name__ == "__main__":
    main()
