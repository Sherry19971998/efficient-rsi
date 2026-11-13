# Modifications to HGM

This directory is the Huxley-Gödel Machine code from <https://github.com/metauto-ai/HGM>,
commit `013872d`, released under the Apache License 2.0 (`LICENSE`, unchanged). The
efficient-RSI project changed the files listed below. `EFFICIENT_RSI_CHANGES.patch` holds the
full diff of the modified files against that commit; the added files are new. The
`HGM_HIDE_ANSWERS` change and the patch-file race fix came later and are not in that patch; they
are in this repository's git history.

Every change that alters behaviour is off unless a flag or environment variable turns it on, so
the code still runs as upstream HGM by default, except for the `tree.py` fix (see below).

## Modified files

| File | Change |
| --- | --- |
| `tree.py` | Fix: `get_pseudo_decendant_evals` returned a node's own evaluation list, which the caller then extended in place, mixing descendants' results into the node's record. It now returns a copy. `HGM_FIX_DESCENDANT_EVALS=0` restores the upstream behaviour. |
| `hgm.py` | Cost-ledger events; `--search dgm` runs DGM's search policy on this harness (`dgm_search.py`); `--max_hours` stops a search after a number of hours; optional node stopping from `erl/online_stop.py` (`HGM_ERL_STOP=1`, `HGM_ERL_EXPAND=1`). |
| `hgm_utils.py` | Cost-ledger timing of proposals and evaluations; host-side diagnose calls tagged for the metering proxy. Fix: for Polyglot, `sample_child` filters a new child's `model_patch.diff` once (`remove_patch_by_files`) before the child joins the tree, so evaluations no longer need to rewrite it (see `polyglot/harness.py`). |
| `self_improve_step.py` | Diagnose calls tagged for the metering proxy; with `HGM_DIAG_CLIP=1`, a diagnose prompt still over the length limit is cut in the middle, keeping its answer format, and each retry lowers the limit. |
| `prompts/self_improvement_prompt.py` | With `HGM_DIAG_CLIP=1`, clips the parts of the diagnose prompt that upstream leaves unbounded (code, solution patch, test log); upstream's own clipping lines for them are commented out. |
| `llm.py` | When `RSI_PROXY_PREFIX` is set (metering on), vLLM and OpenRouter calls go through the metering proxy; requests are forwarded unchanged. |
| `polyglot/harness.py` (race fix) | Fix: upstream `harness()` filtered every patch of the agent's lineage and wrote it back over the shared file with `open(path, "w")`, which truncates it first. HGM evaluates one agent in several threads at once, so a thread could read a just-truncated file and write it back empty, and the agent then ran as its parent without any sign of it (in v1: 8 first-generation patches emptied, among them hgm-fixed-r1's final agent). `harness()` now filters each patch into a private temporary copy for that call and only reads the shared files; the upstream body is `_harness()`. Test: `instrumentation/tests/test_patch_copies.py`. Always on: the filter is idempotent, so only the race changes. |
| `polyglot/harness.py`, `swe_bench/harness.py` | Cost-ledger timing of each evaluation phase; containers get the proxy address. In `polyglot/harness.py`, with `HGM_HIDE_ANSWERS=1`: once the agent has finished, every process it left in the container is killed, and the commit with the tests is fetched from the task's repo on the host before the usual grading steps. |
| `polyglot/test_spec.py` | With `HGM_HIDE_ANSWERS=1`, the instance image no longer gives the agent the answers. Upstream, the image holds the original task folder (`/polyglot/...`, with its tests and the `.meta/` reference solution), and `/testbed`'s git history keeps the commit with all files (`origin/HEAD`, `ORIG_HEAD`), so `git log --all` or `find /` finds them. The setup script now drops those refs, prunes the unreachable objects and deletes the task folder. These images are named `pb.eval.<arch>.<task>.hidden`, so upstream images are never reused for them. Still open: the agent's container keeps network access (each evaluation `pip install`s the agent's requirements), so the public benchmark could be downloaded. |
| `.dockerignore` | Keeps the added orchestration files out of the agent's container image. |

## Added files

| File | Purpose |
| --- | --- |
| `dgm_search.py` | DGM's search policy (parent selection, staged evaluation, archive) on HGM's harness, so DGM and HGM differ only in how they search. |
| `final_eval.py` | Picks a search's final agent at an equal-budget cut and evaluates it on held-out tasks. |
| `repeat_eval.py` | Repeated evaluations of one agent (noise and test-time-scaling experiments). |
| `config_local.yaml` | Configuration for one local Qwen3-Coder-30B-A3B model in every role. |
| `rsi_ledger.py` | Link to `../../instrumentation/rsi_ledger.py`, the cost-ledger hooks. |
| `MODIFICATIONS.md`, `EFFICIENT_RSI_CHANGES.patch` | This notice and the diff. |
