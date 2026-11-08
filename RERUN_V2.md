# The main experiment (`./run_all.sh v2`): about 6–7 days on one H100

This is the full comparison: HGM, DGM's search policy and HGM with prediction-driven stopping,
three repeats each, every search capped in time and every comparison cut at the same GPU
budget. It is called v2 because it replaced a first round of runs whose harness had the
problems described below. Set up the machine with `README.md` first.

## What changed from the first round, and why

Every child that HGM or DGM proposes starts with a *diagnose* call: the model reads the
parent agent's code and its log and patch on one failed task, and writes a problem statement.
Upstream HGM puts the parent's solution patch and code into that prompt uncut (its own clipping
lines are commented out). Some parents' patches held tens of millions of characters (up to
59 M), far past Qwen3-Coder-30B's 131k-token context, so vLLM rejected the call, the proposal
failed, and the search picked the same parent again. In the first nine runs, 39–99% of
proposals failed this way. That wasted worker time and kept some parents from ever having
children, unequally across runs, which muddies every comparison.

Later we found that upstream task images also leave each task's tests and reference solution
inside the solving agent's container (the original task folder, and the commit with all files
in `/testbed`'s git history). Agents found and read them in about 0.6% of the solving runs we
checked. `HGM_HIDE_ANSWERS=1` removes them; it is set in `env.sh.example` and must stay on.

| change | where | switch |
| --- | --- | --- |
| Clip the diagnose prompt: code 120k, solution patch 50k, test log 30k characters (md log is already clipped at 100k upstream), whole prompt 300k; a prompt still over the limit is cut in the middle so its closing answer format survives, and each retry lowers the limit by a quarter (300k, 225k, 169k) | `repos/HGM/prompts/self_improvement_prompt.py`, `repos/HGM/self_improve_step.py` | `HGM_DIAG_CLIP=1`, set by `v2` |
| HGM stops after a number of hours, like DGM already does (no new expansion or evaluation is started; those in flight finish) | `repos/HGM/hgm.py` | `--max_hours`, set by `v2` from `V2_HOURS` |
| Tests and reference solutions are kept out of the solving agent's container, and fetched from the host only for grading | `repos/HGM/polyglot/test_spec.py`, `repos/HGM/polyglot/harness.py` | `HGM_HIDE_ANSWERS=1`, set in `env.sh` |

All comparisons are cut at 9.646 GPU-occupancy hours (the shortest of the first nine runs, kept
so that results stay comparable), so every search is capped at 10.5 wall hours, a margin over
9.646. The evaluation budget is raised to 2000 so that the hour cap, not the budget, ends each
search.

## What runs

`./run_all.sh v2` runs, one at a time:

1. Searches, interleaved by repeat: `hgm-fixed`, `dgm`, `hgm-erlx` for repeats `0v2`, `1v2`,
   `2v2` (9 runs, about 10.5 h each, about 4 days).
2. `hgm-erlxa-r0v2`: hgm-erlx with 10% of stop decisions audited (about 10.5 h).
3. Final held-out evaluations of all 10 runs, each cut at 9.646 GPU hours (about 3.5 h each,
   about 1.5 days). The first one also evaluates the initial agent on the held-out tasks; the
   later ones reuse that result. `final_eval.py` refuses a run that has fewer GPU hours than
   the cut (a crash, a long gap), since it would otherwise be compared at a smaller budget;
   continue that search first, or pass `--allow-short` and report it.

Configurations: `hgm-fixed` = HGM with the descendant-evaluation fix; `dgm` = DGM's search
policy on HGM's harness (`repos/HGM/dgm_search.py`); `hgm-erlx` = hgm-fixed plus
prediction-driven stopping of weak children (`erl/online_stop.py`), whose freed slots go to new
children; `hgm-erlxa` = hgm-erlx plus audits.

The stopping predictor can also learn from earlier runs (`runs/erl_history.json`). That file is
a run output, not part of the repository: on a new machine it does not exist, and `hgm-erlx`
then trains its predictor on the current search alone (the log says so). To give later runs a
history, build it from finished runs:

```bash
# the patch features of every node, then the history rows
(cd instrumentation/prescreen && python predict_from_diff.py --ledger ../../runs/ledger.jsonl \
    --output-root ../../repos/HGM/output_hgm --runs <finished run labels> --rows-out ../../runs/prescreen_rows.jsonl)
cd erl && python online_stop.py dump-history --ledger ../runs/ledger.jsonl \
    --prescreen-rows ../runs/prescreen_rows.jsonl --runs <finished run labels> --out ../runs/erl_history.json
```

Knobs (environment variables): `V2_HOURS` (10.5), `V2_REPEATS` ("0v2 1v2 2v2"; use other
labels to start a fresh set, e.g. "0v3 1v3 2v3"), `V2_AUDIT_REPEATS` (first repeat; empty =
skip hgm-erlxa), `V2_BUDGET` (2000), `FINAL_HOURS` (9.646).

## Before starting

1. The GPU works: `nvidia-smi` prints the GPU, not an error such as "Driver/library version
   mismatch" (that one needs a reboot). Nothing but this experiment's vLLM should use it: the
   comparison counts GPU time.
2. The services run (README step 6) and `./run_all.sh check` prints only `ok` lines, including
   `HGM_HIDE_ANSWERS=1`.
3. The initial agent was evaluated with the current settings (`./run_all.sh init`; `v2` refuses
   to start otherwise).
4. Nothing else is searching: `pgrep -af "python.* hgm\.py"` prints nothing. `v2` refuses to
   start otherwise.
5. Disk: at least 30 GB free for the outputs (each of the 10 runs writes 2–3 GB after
   compression). Check quotas too, not only `df`: a test write is the surest check.

## Start

Run it inside tmux (or screen), since it takes days:

```bash
source env.sh
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

Healthy signs: `propose_failed` is a small share of `select_parent`; `budget_stop` with
`reason: max_hours` appears about 10.5 h after a search starts; `erl_error` never appears.
`runs/<label>.setup.txt` records the settings of each run: it should show `HGM_DIAG_CLIP=1`
and `HGM_HIDE_ANSWERS=1`. Diagnose logs (`repos/HGM/output_hgm/<label>/<child>/self_improve.log`)
show `<patch clipped: ...>` where a patch was long.

Heavy CPU work on the machine during a run (other jobs, offline analyses) slows the evaluation
containers and counts against that run's GPU hours: run analyses with `nice -n 19 ionice -c3`.

## After it finishes

```bash
# cost breakdowns and held-out comparison (writes runs/compare.txt, runs/anytime.txt)
CONFIGS="hgm-fixed dgm hgm-erlx" REPEATS="0v2 1v2 2v2" FINAL_HOURS=9.646 ./run_all.sh report
# search-side comparison at the cut (children, evaluations, retired, selected node)
cd erl && python erl_compare.py ../runs/ledger.jsonl \
  hgm-fixed-r0v2,hgm-fixed-r1v2,hgm-fixed-r2v2,dgm-r0v2,dgm-r1v2,dgm-r2v2,hgm-erlx-r0v2,hgm-erlx-r1v2,hgm-erlx-r2v2,hgm-erlxa-r0v2 9.646
```

Held-out accuracy per run is in `repos/HGM/output_hgm/<label>/final_eval/final_result.json`
(`best_accuracy`).

## Known limits

- GPU hours are wall time minus gaps over 30 minutes, not GPU busy time; the GPU sampler's
  busy hours are a separate figure, and need a working `nvidia-smi`.
- `--max_hours` measures from the process start; a search continued after a crash restarts
  its hour count.
- The solving agent's container still has network access (each evaluation installs the agent's
  Python requirements), so an agent could in principle download the public benchmark.
