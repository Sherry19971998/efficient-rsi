"""Evaluate the selected agent of a finished search on held-out tasks (efficient-RSI).

Works for both search policies on this harness (`hgm.py --search hgm|dgm`):
1. read the run's search results from the cost ledger, cut at the equal-budget point
   (--max-hours of GPU occupancy, or else --max-evals task evaluations);
2. pick the final agent with the rule in instrumentation/results.py (same for every method);
3. evaluate it, and the initial agent as a reference, on tasks the search never used.

Held-out tasks: Polyglot = all 225 tasks minus the small and medium subsets (166 tasks);
SWE-bench Verified = the 500 tasks minus those subsets, sampled down with --n-heldout
because a full pass is expensive.

The final evaluation is logged under RSI_RUN=<run>-final so that it never counts toward
the search budget or the search's cost.

Usage (from the repository root, after the search):
    python final_eval.py --output_dir output_hgm/<run> --run hgm-fixed-seed0 --polyglot --max-evals 800
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "instrumentation"))
import analyze_ledger  # noqa: E402
import results  # noqa: E402

from config import load_config  # noqa: E402
from utils.common_utils import load_json_file  # noqa: E402
from utils.evo_utils import get_all_performance, get_model_patch_paths  # noqa: E402


def heldout_tasks(polyglot, n=None, seed=0):
    bench = "polyglot" if polyglot else "swe_bench"
    search = set(load_json_file(f"./{bench}/subsets/small.json")) | set(load_json_file(f"./{bench}/subsets/medium.json"))
    if polyglot:
        every = {e["instance_id"] for e in load_json_file("./polyglot/polyglot_benchmark_metadata.json")}
    else:
        from datasets import load_dataset
        every = {t["instance_id"] for t in load_dataset("princeton-nlp/SWE-bench_Verified")["test"]}
    held = sorted(every - search)
    if n is not None and n < len(held):
        held = sorted(random.Random(seed).sample(held, n))
    return held


def evaluate(node, tasks, run_dir, final_dir, src_path, polyglot, max_workers):
    """{task: 0/1} for one agent; results go to final_dir/<node>, never into the search tree."""
    root = os.path.abspath("./")
    patches = [] if node == "initial" else get_model_patch_paths(root, run_dir, node)
    out = os.path.join(final_dir, node)
    os.makedirs(out, exist_ok=True)
    if polyglot:
        from polyglot.harness import harness as polyglot_harness
        polyglot_harness(test_task_list=tasks, max_workers=min(max_workers, len(tasks)), model_name_or_path=node,
                         model_patch_paths=patches, pred_dname=os.path.join(out, "predictions"),
                         output_dir=out, init_agent_path=src_path)
    else:
        from swe_bench.harness import harness as swe_harness
        from swe_bench.report import make_report
        dnames = swe_harness(test_task_list=tasks, max_workers=min(max_workers, len(tasks)), model_name_or_path=node,
                             model_patch_paths=patches, pred_dname=os.path.join(out, "predictions"),
                             init_agent_path=src_path)
        make_report(dnames, run_ids=[f"{node}_{i}" for i in range(len(dnames))], output_dir=out,
                    num_eval_procs=min(max_workers, len(tasks)))
    _, perf = get_all_performance(node, results_dir=out)
    resolved = set((perf or {}).get("total_resolved_ids", []))
    return {t: int(t in resolved) for t in tasks}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", required=True, help="the search run's output directory")
    ap.add_argument("--run", default=os.getenv("RSI_RUN"), help="the search run's RSI_RUN label in the ledger")
    ap.add_argument("--ledger", default=os.getenv("RSI_LEDGER"))
    ap.add_argument("--max-evals", type=int, default=800)
    ap.add_argument("--rule", choices=["beta_lcb", "mean"], default="beta_lcb")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--polyglot", action="store_true")
    ap.add_argument("--n-heldout", type=int, default=None, help="sample this many held-out tasks (needed for SWE-bench)")
    ap.add_argument("--max_workers", type=int, default=10)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--initial_agent_name", default="default_agent")
    ap.add_argument("--skip-initial", action="store_true", help="do not re-evaluate the initial agent")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="cut the search at this many GPU-occupancy hours instead of at --max-evals")
    ap.add_argument("--name", default="final_eval",
                    help="output subdirectory; use another name (e.g. ckpt-0.33) for checkpoint evaluations")
    ap.add_argument("--allow-short", action="store_true",
                    help="evaluate even if the search ran fewer GPU hours than --max-hours (the result then "
                         "comes from a smaller budget than the other runs')")
    args = ap.parse_args()
    if not args.run or not args.ledger:
        ap.error("--run and --ledger (or RSI_RUN and RSI_LEDGER) are required")
    if not args.polyglot and args.n_heldout is None:
        ap.error("SWE-bench needs --n-heldout: a full pass over the ~440 held-out tasks is expensive")

    cfg = load_config(args.config)
    import polyglot.harness
    import swe_bench.harness
    polyglot.harness.llm = swe_bench.harness.llm = cfg.llm.downstream_llm
    polyglot.harness.timeout = swe_bench.harness.timeout = cfg.execution.evaluation_timeout

    events, cutoff = analyze_ledger.truncate(analyze_ledger.load(args.ledger, args.run), args.max_evals, args.max_hours)
    # truncate() keeps the whole run when it never reaches the cut, which would silently compare this
    # run at a smaller budget than the others. 0.01 h of slack covers the 3-decimal rounding of the
    # shortest run's hours that run_all.sh uses as the common cut.
    short = bool(args.max_hours and cutoff is None and analyze_ledger.gpu_hours(events) < args.max_hours - 0.01)
    if short and not args.allow_short:
        sys.exit(f"{args.run} ran only {analyze_ledger.gpu_hours(events):.3f} GPU hours, short of the "
                 f"--max-hours {args.max_hours} cut; continue the search or pass --allow-short")
    scores = results.search_scores(events)
    best = results.select_best(scores, rule=args.rule, eps=args.eps)
    held = heldout_tasks(args.polyglot, args.n_heldout)
    cut = {"gpu_hours": round(analyze_ledger.gpu_hours(events), 3),
           "task_evals": sum(1 for e in events if e["type"] == "eval" and e.get("phase") == "agent_run"
                             and e.get("node") != "initial")}
    print(f"Search cut at {cut['gpu_hours']} GPU hours, {cut['task_evals']} task evaluations")
    print(f"Selected {best} ({sum(scores[best])}/{len(scores[best])} search tasks solved) "
          f"from {len(scores)} nodes; evaluating on {len(held)} held-out tasks")

    # Keep held-out evaluations out of the search's accounting.
    os.environ["RSI_RUN"] = f"{args.run}-final" if args.name == "final_eval" else f"{args.run}-{args.name}"
    run_dir = os.path.abspath(args.output_dir)
    final_dir = os.path.join(run_dir, args.name)
    src_path = os.path.join("initial_polyglot" if args.polyglot else "initial_swe", args.initial_agent_name, "src")
    best_res = evaluate(best, held, run_dir, final_dir, src_path, args.polyglot, args.max_workers)
    init_res = None if args.skip_initial else evaluate("initial", held, run_dir, final_dir, src_path,
                                                      args.polyglot, args.max_workers)

    out = {
        "run": args.run, "rule": args.rule, "eps": args.eps,
        "max_evals": None if args.max_hours else args.max_evals, "max_hours": args.max_hours, "cut": cut,
        "short_of_cut": short,
        "benchmark": "polyglot" if args.polyglot else "swe_bench_verified",
        "best_node": best, "best_search": {"solved": sum(scores[best]), "evaluated": len(scores[best])},
        "heldout_tasks": held, "best": best_res, "initial": init_res,
        "best_accuracy": sum(best_res.values()) / len(held),
        "initial_accuracy": None if init_res is None else sum(init_res.values()) / len(held),
    }
    with open(os.path.join(final_dir, "final_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ("heldout_tasks", "best", "initial")}, indent=2))


if __name__ == "__main__":
    main()
