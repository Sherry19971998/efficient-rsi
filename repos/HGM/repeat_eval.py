"""Evaluate one agent several times on the same tasks (efficient-RSI, experiments 1 and 2).

1. Noise: the initial agent, K passes over the 59 search tasks. How much does one task's
   outcome vary between passes, and so how many tasks a comparison needs.
2. Test-time scaling: the initial agent, K more passes over the 166 held-out tasks. With
   the held-out pass from `final`, pass@k (k <= K + 1) is what spending the search's GPU
   hours on repeated attempts instead would buy. Polyglot's tests are hidden from the
   agent, so picking the passing attempt needs an oracle: pass@k is an upper bound.

Each pass is logged under RSI_RUN=<label>-rep<i>, so the ledger gives its GPU hours, and
writes output_hgm/<label>/rep<i>/result.json ({task: 0/1}). Passes already written are
skipped, so an interrupted job resumes where it stopped.

Usage (from the repository root):
    python repeat_eval.py --config config_local.yaml --polyglot --tasks search --repeats 3 --label noise-search
    python repeat_eval.py --config config_local.yaml --polyglot --tasks heldout --repeats 3 --label tts-heldout
"""

import argparse
import json
import os

from config import load_config
from final_eval import evaluate, heldout_tasks
from utils.common_utils import load_json_file


def search_tasks(polyglot):
    bench = "polyglot" if polyglot else "swe_bench"
    return sorted(set(load_json_file(f"./{bench}/subsets/small.json")) | set(load_json_file(f"./{bench}/subsets/medium.json")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="output_hgm/<label>/ and the RSI_RUN prefix")
    ap.add_argument("--tasks", choices=["search", "heldout"], required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--node", default="initial", help="agent to evaluate: 'initial' or a node of --run-dir")
    ap.add_argument("--run-dir", help="search run output directory holding --node (not needed for 'initial')")
    ap.add_argument("--polyglot", action="store_true")
    ap.add_argument("--n-heldout", type=int, default=None)
    ap.add_argument("--max_workers", type=int, default=10)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--initial_agent_name", default="default_agent")
    args = ap.parse_args()
    if args.node != "initial" and not args.run_dir:
        ap.error("--run-dir is required for a node other than 'initial'")

    cfg = load_config(args.config)
    import polyglot.harness
    import swe_bench.harness
    polyglot.harness.llm = swe_bench.harness.llm = cfg.llm.downstream_llm
    polyglot.harness.timeout = swe_bench.harness.timeout = cfg.execution.evaluation_timeout

    tasks = search_tasks(args.polyglot) if args.tasks == "search" else heldout_tasks(args.polyglot, args.n_heldout)
    out_root = os.path.abspath(os.path.join("output_hgm", args.label))
    run_dir = os.path.abspath(args.run_dir) if args.run_dir else out_root
    src_path = os.path.join("initial_polyglot" if args.polyglot else "initial_swe", args.initial_agent_name, "src")
    print(f"{args.label}: {args.node} on {len(tasks)} {args.tasks} tasks, {args.repeats} passes")

    for i in range(args.repeats):
        rep_dir = os.path.join(out_root, f"rep{i}")
        result_path = os.path.join(rep_dir, "result.json")
        if os.path.exists(result_path):
            print(f"rep{i}: done already")
            continue
        os.environ["RSI_RUN"] = f"{args.label}-rep{i}"
        res = evaluate(args.node, tasks, run_dir, rep_dir, src_path, args.polyglot, args.max_workers)
        os.makedirs(rep_dir, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump({"label": args.label, "rep": i, "node": args.node, "tasks": args.tasks,
                       "rsi_run": os.environ["RSI_RUN"], "result": res}, f, indent=2)
        print(f"rep{i}: {sum(res.values())}/{len(tasks)} solved")


if __name__ == "__main__":
    main()
