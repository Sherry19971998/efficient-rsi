# Modifications to HGM

This directory is the Huxley-Gödel Machine code from <https://github.com/metauto-ai/HGM>,
commit `013872d`, released under the Apache License 2.0 (`LICENSE`, unchanged). The
efficient-RSI project changed the files listed below. `EFFICIENT_RSI_CHANGES.patch` holds the
full diff of the modified files against that commit; the added files are new.

Every change that alters behaviour is off unless a flag or environment variable turns it on, so
the code still runs as upstream HGM by default, except for the `tree.py` fix (see below).

## Modified files

| File | Change |
| --- | --- |
| `tree.py` | Fix: `get_pseudo_decendant_evals` returned a node's own evaluation list, which the caller then extended in place, mixing descendants' results into the node's record. It now returns a copy. `HGM_FIX_DESCENDANT_EVALS=0` restores the upstream behaviour. |
| `hgm.py` | Cost-ledger events; `--search dgm` runs DGM's search policy on this harness (`dgm_search.py`); `--max_hours` stops a search after a number of hours; optional node stopping from `erl/online_stop.py` (`HGM_ERL_STOP=1`, `HGM_ERL_EXPAND=1`). |
| `hgm_utils.py` | Cost-ledger timing of proposals and evaluations; host-side diagnose calls tagged for the metering proxy. |
| `self_improve_step.py` | Diagnose calls tagged for the metering proxy; with `HGM_DIAG_CLIP=1`, a diagnose prompt still over the length limit is cut in the middle, keeping its answer format, and each retry lowers the limit. |
| `prompts/self_improvement_prompt.py` | With `HGM_DIAG_CLIP=1`, clips the parts of the diagnose prompt that upstream leaves unbounded (code, solution patch, test log); upstream's own clipping lines for them are commented out. |
| `llm.py` | When `RSI_PROXY_PREFIX` is set (metering on), vLLM and OpenRouter calls go through the metering proxy; requests are forwarded unchanged. |
| `polyglot/harness.py`, `swe_bench/harness.py` | Cost-ledger timing of each evaluation phase; containers get the proxy address. |
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
