# Rerun v2: corrected harness, about 6–7 days on one H100

## Why the first runs are rerun

Every child that HGM or DGM proposes starts with a *diagnose* call: the model reads the
parent agent's code and its log and patch on one failed task, and writes a problem statement.
Upstream HGM puts the parent's solution patch and code into that prompt uncut (its own clipping
lines are commented out). Some parents' patches held tens of millions of characters (up to
59 M), far past Qwen3-Coder-30B's 131k-token context, so vLLM rejected the call, the proposal
failed, and the search picked the same parent again. In the first nine runs, 39–99% of
proposals failed this way; hgm-orig-r1 failed 7428 times, 1061 of them on a single parent.
That wasted worker time and kept some parents from ever having children, unequally across
runs, which muddies every comparison.

v2 changes three things and nothing else:

| change | where | switch |
| --- | --- | --- |
| Clip the diagnose prompt: code 120k, solution patch 50k, test log 30k characters (md log is already clipped at 100k upstream), whole prompt 300k; a prompt still over the limit is cut in the middle so its closing answer format survives, and each retry lowers the limit by a quarter (300k, 225k, 169k) | `repos/HGM/prompts/self_improvement_prompt.py`, `repos/HGM/self_improve_step.py` | `HGM_DIAG_CLIP=1` (off by default) |
| HGM stops after a number of hours, like DGM already does (no new expansion or evaluation is started; those in flight finish) | `repos/HGM/hgm.py` | `--max_hours`, set by `HGM_MAX_HOURS` |
| New run labels ending in `v2` (`hgm-fixed-r0v2`, ...) | `run_all.sh` | `V2_REPEATS` |

All comparisons are cut at 9.646 GPU-occupancy hours, so every search is capped at 10.5 wall
hours (a margin over 9.646) instead of running to 800 evaluations (13–20 h). The evaluation
budget is raised to 2000 so that the hour cap, not the budget, ends each search.

Not rerun: hgm-orig (it only checks the upstream bookkeeping bug; its first-run results stay,
with this caveat), the noise and test-time-scaling experiments (they evaluate the initial agent
only, which the fix does not touch), and hgm-erl without expansion (its first run showed it
creates no more children).

## What runs

`./run_all.sh v2` runs, one at a time:

1. Searches, interleaved by repeat: `hgm-fixed`, `dgm`, `hgm-erlx` for repeats `0v2`, `1v2`,
   `2v2` (9 runs, about 10.5 h each, about 4 days).
2. `hgm-erlxa-r0v2`: hgm-erlx with 10% of stop decisions audited (about 10.5 h).
3. Final held-out evaluations of all 10 runs, each cut at 9.646 GPU hours (about 3.5 h each,
   about 1.5 days). The initial agent's held-out result from the first runs is reused.
   `final_eval.py` refuses a run that has fewer GPU hours than the cut (a crash, a long gap),
   since it would otherwise be compared at a smaller budget; continue that search first, or
   pass `--allow-short` and report it.

Configurations: `hgm-fixed` = HGM with the descendant-evaluation fix; `dgm` = DGM's search
policy on HGM's harness (`repos/HGM/dgm_search.py`); `hgm-erlx` = hgm-fixed plus
prediction-driven stopping of weak children (`erl/online_stop.py`), whose freed slots go to new
children; `hgm-erlxa` = hgm-erlx plus audits.

Knobs (environment variables): `V2_HOURS` (10.5), `V2_REPEATS` ("0v2 1v2 2v2"),
`V2_AUDIT_REPEATS` (first repeat; empty = skip hgm-erlxa), `V2_BUDGET` (2000),
`FINAL_HOURS` (9.646).

## Before starting

On engr-liu01s, in tmux session `erl` (windows: vllm, proxy, gpu, main, vmetrics):

1. The GPU driver is consistent: `nvidia-smi` prints the GPU. On 2026-11-03 it printed
   "Driver/library version mismatch" instead: the NVIDIA kernel module (615.71.09) and user
   library (615.71.10) differ, so no new CUDA process can start until the machine is rebooted.
   The vLLM server that was already running kept working, but if it stops at any point during
   the 6–7 days, nothing can restart it. Ask the admin to reboot before starting, while nothing
   of v2 is running yet; after the reboot, start vLLM, the proxy and the samplers again
   (`./run_all.sh vllm` and the tmux windows listed above).
2. vLLM serves the model: `curl -s localhost:8000/v1/models` lists `vllm-qwen3c-local`.
3. The proxy runs: `pgrep -af llm_proxy.py`.
4. Nothing else is searching: `pgrep -af hgm.py` prints nothing. `run_all.sh v2` refuses to
   start otherwise. To stop the earlier queue: Ctrl-C in the `main` window, and
   `pkill -f tmp/after_erlx.sh` (the watcher that would start hgm-erlxa of the old plan).
5. Disk: at least 30 GB free under `~/efficient-rsi` (each of the 10 runs writes 2–3 GB after compression).
   An earlier run hit the home quota; check with `df -h ~` and a test write.

## Start

In the `main` window:

```bash
cd ~/efficient-rsi && source env.sh
./run_all.sh v2
```

It logs to `runs/run_all.log`. A failed search stops the whole sequence (log line
`v2 stopping: <label> failed`); nothing later starts on a broken setup. To resume after fixing
the cause, run `./run_all.sh v2` again: finished runs (`runs/<label>.complete`) are skipped,
and an interrupted one continues from its output directory.

## Checking progress

```bash
tail -5 runs/run_all.log
# events of the current run (the ledger is large; read only its tail)
tail -c 40000000 runs/ledger.jsonl | grep '"run": "hgm-erlx-r0v2"' | grep -v llm_call \
  | python3 -c 'import sys,json,collections; print(collections.Counter(json.loads(l)["type"] for l in sys.stdin))'
```

Healthy signs: `propose_failed` is now a small share of `select_parent`; `budget_stop` with
`reason: max_hours` appears about 10.5 h after a search starts; `erl_error` never appears.
To check that the fix is active: `grep HGM_DIAG_CLIP runs/<label>.setup.txt` shows `1`, and
diagnose logs (`repos/HGM/output_hgm/<label>/<child>/self_improve.log`) show
`<patch clipped: ...>` where a patch was long.

## After it finishes

```bash
# cost breakdowns and held-out comparison (writes runs/compare.txt, runs/anytime.txt)
CONFIGS="hgm-fixed dgm hgm-erlx" REPEATS="0v2 1v2 2v2" FINAL_HOURS=9.646 ./run_all.sh report
# search-side comparison at the cut (children, evaluations, retired, selected node)
cd erl && ../venv/bin/python erl_compare.py ../runs/ledger.jsonl \
  hgm-fixed-r0v2,hgm-fixed-r1v2,hgm-fixed-r2v2,dgm-r0v2,dgm-r1v2,dgm-r2v2,hgm-erlx-r0v2,hgm-erlx-r1v2,hgm-erlx-r2v2,hgm-erlxa-r0v2 9.646
```

Held-out accuracy per run is in `repos/HGM/output_hgm/<label>/final_eval/final_result.json`
(`best_accuracy`).

## Known limits

- The predictor's history (`runs/erl_history.json`) was built from the first nine runs, under
  the unclipped harness. Its labels (solve rates) are still valid; the mix of parents differs.
- GPU hours are wall time minus gaps over 30 minutes, not GPU busy time. Heavy CPU work on the
  server during a run (other jobs, offline replays) slows evaluation containers and counts
  against that run.
- `--max_hours` measures from the process start; a search continued after a crash restarts
  its hour count.
