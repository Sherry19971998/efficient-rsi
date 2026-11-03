# Changes to DGM (secondary check only)

The main experiments do not use DGM's own code: the DGM baseline is DGM's search policy
re-implemented on HGM's harness (`repos/HGM/dgm_search.py`), so that DGM and HGM share the
initial agent, evaluation code and models and differ only in how they search. These changes to
DGM's own code are kept so that DGM can also be run as itself, as a secondary check.

Upstream: <https://github.com/jennyzzt/dgm>, commit `a565fd2`, Apache License 2.0 (`LICENSE`,
copied unchanged). Only the diff and our one new script are stored here, not the 64 MB repository.

| File | What it is |
| --- | --- |
| `DGM_CHANGES.patch` | Diff of the 9 modified files against `a565fd2` |
| `eval_initial_agent.py` | New script: re-evaluates DGM's initial agent with the experiment's models |
| `LICENSE` | DGM's licence |

## What the patch changes

| File | Change |
| --- | --- |
| `coding_agent.py`, `coding_agent_polyglot.py`, `self_improve_step.py` | The models are set by `DGM_CODE_MODEL`, `DGM_SELF_IMPROVE_MODEL` and `DGM_DIAGNOSE_MODEL`; upstream hard-codes Claude 3.5 Sonnet on Bedrock and o1. Unset, upstream's models are used. |
| `llm.py`, `llm_withtools.py` | `gpt-5` models take the existing `o3` branch (Responses API). |
| `DGM_outer.py` | `DGM_INITIAL_DIR` starts the search from re-evaluated initial results (the bundled ones come from upstream's models); cost-ledger events. |
| `self_improve_step.py`, `polyglot/harness.py`, `swe_bench/harness.py` | Cost-ledger timing; containers get the metering proxy's address and the model variables. |
| `.dockerignore` | Keeps the added files out of the agent's container image. |

## Restoring it

```bash
git clone https://github.com/jennyzzt/dgm.git repos/dgm
git -C repos/dgm checkout a565fd2
git -C repos/dgm apply ../dgm-changes/DGM_CHANGES.patch
cp repos/dgm-changes/eval_initial_agent.py repos/dgm/
ln -s ../../instrumentation/rsi_ledger.py repos/dgm/rsi_ledger.py
```

`instrumentation/README.md` shows how to run it (`eval_initial_agent.py`, then `DGM_outer.py`
with `DGM_INITIAL_DIR`). Note that DGM's own code calls
hosted models only (OpenAI, Anthropic, Bedrock); it has no vLLM branch, so it cannot use the
local Qwen model as it stands.
