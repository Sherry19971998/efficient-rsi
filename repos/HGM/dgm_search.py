"""DGM's search policy, run on HGM's harness (efficient-RSI baseline).

Only the outer loop differs from HGM: the initial agent, self-improvement step,
evaluation harness and models are HGM's, so a comparison between the two isolates
the search policy. The rules follow the DGM paper and its public code
(jennyzzt/dgm, DGM_outer.py and self_improve_step.py):

- Generations: each generation picks `n_parallel` parents (4 for Polyglot and 2 for
  SWE-bench in the DGM paper) and creates one child from each, in parallel; the next
  generation starts only when all children are done.
- Parent selection ("score_child_prop"): weight = sigmoid(10 * (accuracy - 0.5)) /
  (1 + number of children in the archive), sampled with replacement.
- Staged evaluation: a child first runs the small subset (10 tasks); if it solves at
  least 40% of it, it also runs the medium subset (50 tasks).
- Archive ("keep_all"): every child that produced a patch and finished the small
  subset joins the archive.

Deliberate differences from DGM's public code, disclosed in the report:
- Parent accuracy uses every task a node has been evaluated on. DGM's Polyglot mode
  stores the medium-subset result in a separate field and selects parents on the
  10-task score only; its SWE-bench mode and its paper use all evaluated tasks.
- Tasks shared by the small and medium subsets are evaluated once, not twice.
- The run stops when the task-evaluation budget is spent, so that DGM and HGM are
  compared at the same number of task evaluations.
- Optionally (max_hours) it also stops once that many hours have passed since the
  process started, checked between generations: the generation in flight finishes, so
  the run overshoots by at most one generation. DGM keeps the GPU far less busy than
  HGM (4 children per generation, which waits for its slowest evaluation), so at an
  equal number of evaluations it holds the GPU much longer; comparisons are made at
  equal GPU hours, and hours past the cap would not enter them. The stop is recorded in
  <output_dir>/stopped_on_time.json and as a `budget_stop` ledger event.
"""

import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import hgm_utils
import rsi_ledger
from utils.common_utils import load_json_file

TEST_MORE_THRESHOLD = 0.4


def node_performance(output_dir, commit):
    path = os.path.join(output_dir, commit, "metadata.json")
    if not os.path.exists(path):
        return None
    return load_json_file(path).get("overall_performance")


def is_compiled(perf, n_small):
    """DGM's filter: a non-empty patch, and the small subset fully evaluated."""
    if not perf:
        return False
    if len(perf.get("total_resolved_ids", [])) + len(perf.get("total_unresolved_ids", [])) == 0:
        return False
    return perf.get("total_submitted_instances", 0) >= n_small


def choose_parents(output_dir, archive, parent_of, k):
    children = {c: 0 for c in archive}
    for c in archive:
        p = parent_of.get(c)
        if p in children:
            children[p] += 1
    weights = []
    for c in archive:
        acc = (node_performance(output_dir, c) or {}).get("accuracy_score", 0.0)
        weights.append(1 / (1 + math.exp(-10 * (acc - 0.5))) / (1 + children[c]))
    return random.choices(archive, weights=weights, k=k)


def create_and_evaluate(parent, image_name, src_path, small, medium, max_workers):
    child = hgm_utils.sample_child(parent, image_name=image_name)
    if child == "failed":
        rsi_ledger.event("propose_failed", parent=parent)
        return None
    rsi_ledger.event("node_created", node=child, parent=parent)
    scores = hgm_utils.eval_agent(child, tasks=small, init_agent_path=src_path, max_workers=max_workers)
    rsi_ledger.event("eval_result", node=child, score=scores, stage="small")
    if sum(scores) >= TEST_MORE_THRESHOLD * len(small):
        done = set(small)
        rest = [t for t in medium if t not in done]
        # an empty task list would make the harness evaluate the whole benchmark
        if rest:
            more = hgm_utils.eval_agent(child, tasks=rest, init_agent_path=src_path, max_workers=max_workers)
            rsi_ledger.event("eval_result", node=child, score=more, stage="medium")
    return child


def run(output_dir, src_path, image_name, max_task_evals, polyglot, n_parallel=None, max_workers=10, logger=None,
        max_hours=None):
    bench = "polyglot" if polyglot else "swe_bench"
    small = load_json_file(f"./{bench}/subsets/small.json")
    medium = load_json_file(f"./{bench}/subsets/medium.json")
    n_parallel = n_parallel or (4 if polyglot else 2)
    archive, parent_of, generation = ["initial"], {}, 0
    start = time.time()

    while hgm_utils.n_task_evals < max_task_evals:
        hours = (time.time() - start) / 3600
        if max_hours is not None and hours >= max_hours:
            stop = {"max_hours": max_hours, "hours": round(hours, 3), "generation": generation,
                    "n_task_evals": hgm_utils.n_task_evals}
            with open(os.path.join(output_dir, "stopped_on_time.json"), "w") as f:
                json.dump(stop, f, indent=2)
            rsi_ledger.event("budget_stop", reason="max_hours", **stop)
            if logger:
                logger.info(f"DGM stopped after {hours:.2f} h (cap {max_hours} h), "
                            f"{hgm_utils.n_task_evals} task evals")
            break
        parents = choose_parents(output_dir, archive, parent_of, n_parallel)
        for p in parents:
            rsi_ledger.event("select_parent", node=p, generation=generation, n_candidates=len(archive))
        with ThreadPoolExecutor(max_workers=n_parallel) as ex:
            children = list(ex.map(lambda p: create_and_evaluate(p, image_name, src_path, small, medium, max_workers),
                                   parents))
        compiled = []
        for parent, child in zip(parents, children):
            if child is None:
                continue
            parent_of[child] = parent
            if is_compiled(node_performance(output_dir, child), len(small)):
                compiled.append(child)
        archive += compiled
        state = {"generation": generation, "parents": parents, "children": children,
                 "children_compiled": compiled, "archive": archive, "n_task_evals": hgm_utils.n_task_evals}
        with open(os.path.join(output_dir, "dgm_metadata.jsonl"), "a") as f:
            f.write(json.dumps(state) + "\n")
        rsi_ledger.event("generation_done", generation=generation, children=[c for c in children if c],
                         children_compiled=compiled, archive_size=len(archive), n_task_evals=hgm_utils.n_task_evals)
        if logger:
            logger.info(f"DGM generation {generation}: {len(compiled)}/{len(parents)} compiled, "
                        f"archive {len(archive)}, task evals {hgm_utils.n_task_evals}")
        generation += 1
