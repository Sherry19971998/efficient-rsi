"""Re-evaluate the initial DGM agent and write a folder that DGM_outer.py can start from.

The bundled initial/ and initial_polyglot/ folders hold results of the upstream
models (Claude 3.5 Sonnet / o3-mini). When the agent runs on another model
(DGM_CODE_MODEL), the root node must be re-evaluated with that model, otherwise
parent selection starts from another model's scores.

Usage (from the repository root):
    DGM_CODE_MODEL=gpt-5-mini python eval_initial_agent.py --polyglot --out initial_polyglot_gpt-5-mini
    DGM_INITIAL_DIR=initial_polyglot_gpt-5-mini DGM_CODE_MODEL=gpt-5-mini python DGM_outer.py --polyglot
"""

import argparse
import json
import os

from polyglot.harness import harness as polyglot_harness
from swe_bench.harness import harness as swe_harness
from swe_bench.report import make_report
from utils.common_utils import load_json_file
from utils.evo_utils import get_all_performance


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polyglot", action="store_true")
    parser.add_argument("--out", required=True, help="output folder, e.g. initial_polyglot_gpt-5-mini")
    parser.add_argument("--tasks", choices=["search", "full"], default="full",
                        help="search: the small+medium subsets used during the run; full: the whole benchmark "
                             "(the bundled folders cover the full Polyglot set and the SWE-bench subsets)")
    parser.add_argument("--max_workers", type=int, default=10)
    args = parser.parse_args()

    if not os.getenv("DGM_CODE_MODEL"):
        parser.error("Set DGM_CODE_MODEL to the model the run will use.")
    if os.path.exists(args.out):
        parser.error(f"{args.out} already exists; remove it or pick another name.")

    bench = "polyglot" if args.polyglot else "swe_bench"
    if args.tasks == "search":
        subsets = ["small", "medium"]
    elif args.polyglot:
        subsets = []  # the whole Polyglot benchmark
    else:
        subsets = ["small", "medium", "big"]
    tasks = [t for s in subsets for t in load_json_file(f"./{bench}/subsets/{s}.json")] or None

    out = os.path.abspath(args.out)
    os.makedirs(out)
    pred_dname = os.path.join(out, "predictions")
    if args.polyglot:
        polyglot_harness(
            test_task_list=tasks,
            num_samples=-1,
            max_workers=args.max_workers,
            model_name_or_path="initial",
            model_patch_paths=[],
            num_evals=1,
            num_evals_parallel=1,
            pred_dname=pred_dname,
            output_dir=out,
        )
    else:
        dnames = swe_harness(
            test_task_list=tasks,
            num_samples=-1,
            max_workers=args.max_workers,
            model_name_or_path="initial",
            model_patch_paths=[],
            num_evals=1,
            num_evals_parallel=1,
            pred_dname=pred_dname,
        )
        make_report(
            dnames,
            run_ids=[f"initial_{i}" for i in range(len(dnames))],
            dataset_name="princeton-nlp/SWE-bench_Verified",
            output_dir=out,
            dnames_workers=5,
        )

    _, overall_performance = get_all_performance("initial", results_dir=out)
    metadata = {
        "run_id": "initial",
        "code_model": os.getenv("DGM_CODE_MODEL"),
        "overall_performance": overall_performance,
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(json.dumps({k: v for k, v in overall_performance.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
