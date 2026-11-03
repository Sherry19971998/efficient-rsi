# Efficient RSI: cost-measured self-improving coding agents

This repository reproduces the efficient-RSI experiments on the Polyglot benchmark: HGM, DGM's
search policy on HGM's harness, and HGM with prediction-driven node stopping, all driven by one
local Qwen3-Coder-30B-A3B (FP8) model on vLLM, with every GPU hour and token metered. It was
built and first run on one server (engr-liu01s); this README sets it up on another machine.

## Layout

| Path | What it is |
| --- | --- |
| `run_all.sh` | One entry point for everything: services, checks, searches, final evaluations, reports. Its header lists every subcommand and setting. |
| `RERUN_V2.md` | The planned corrected rerun (`./run_all.sh v2`): why, what runs, how long, how to check it. Written for the original server; the steps after "Start" apply anywhere. |
| `repos/HGM/` | HGM (Apache-2.0) with our changes; `repos/HGM/MODIFICATIONS.md` lists them. |
| `repos/dgm-changes/` | Our changes to DGM's own code (Apache-2.0), as a patch: not used by the main experiments, kept for a secondary check. |
| `instrumentation/` | Cost metering: the ledger hooks, the metering proxy, GPU samplers, analysis and comparison scripts. |
| `erl/` | Prediction-driven node stopping (`online_stop.py`) and its offline studies. |
| `runs/erl_history.json` | Training history for the stopping predictor, built from the first nine runs. All other run outputs are ignored by git. |
| `env/` | Exact package versions of the two Python environments on the original server. |

## Requirements

- Linux x86_64, one NVIDIA GPU with 80 GB (an H100 was used), with a driver new enough for
  CUDA 13 wheels (the original server ran 615.71). The GPU must be free for the whole run:
  the cost comparison counts GPU time.
- Docker, with your user in the `docker` group. The 225 Polyglot task images share one base of
  about 6 GB, so images need about 10 GB.
- About 100 GB of free disk for the model (31 GB), Docker images and run outputs (2–3 GB per
  search after compression; the ledger grows to several hundred MB).
- Python 3.10 for the harness and Python 3.11 for vLLM, in separate virtual environments.
- tmux, git, curl. No API keys: every model call goes to the local vLLM server.

## Setup

1. **Settings.**
   ```bash
   cp env.sh.example env.sh   # then edit VLLM_BIN, BRIDGE and, if needed, HF_HOME
   source env.sh
   ```
   `BRIDGE` must be the address of the `docker0` interface (`ip -4 addr show docker0`); the
   metering proxy binds there so that only local containers can reach it.

2. **vLLM environment** (Python 3.11, its own venv):
   ```bash
   python3.11 -m venv ~/venvs/vllm
   ~/venvs/vllm/bin/pip install vllm==0.30.0
   ```
   The original environment had torch 2.13.0+cu130 and transformers 5.18.0;
   `env/requirements-lock-vllm.txt` has every package version, if the plain install resolves
   differently.

3. **Harness environment** (Python 3.10, in this checkout):
   ```bash
   python3.10 -m venv venv && source venv/bin/activate
   pip install -r env/requirements-lock-harness.txt
   # SWE-bench's harness code is imported by HGM even for Polyglot; the pinned commit:
   git clone https://github.com/princeton-nlp/SWE-bench.git repos/HGM/swe_bench/SWE-bench
   git -C repos/HGM/swe_bench/SWE-bench checkout dc4c087c2b9e4cefebf2e3d201d27e36
   pip install -e repos/HGM/swe_bench/SWE-bench
   ```

4. **Polyglot dataset, pinned.** The preparation script clones Aider's benchmark without a
   commit, so clone it first at the commit the experiments used; the script then reuses it.
   ```bash
   cd repos/HGM
   git clone https://github.com/Aider-AI/polyglot-benchmark.git polyglot/polyglot-benchmark
   git -C polyglot/polyglot-benchmark checkout 7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f
   python -m polyglot.prepare_polyglot_dataset     # should report 225 tasks
   cd ../..
   ```

5. **Model weights** (about 31 GB):
   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8')"
   ```

6. **Services**, each in its own tmux window, from this directory with `env.sh` sourced and
   the harness venv active:
   ```bash
   ./run_all.sh vllm     # window 1; ready when it reports the server is listening
   ./run_all.sh proxy    # window 2
   ./run_all.sh gpu      # window 3
   ./run_all.sh check    # all lines should start with "ok"
   ```
   Use a git checkout with everything committed: each run records the commit and any local
   diff of `repos/HGM`, and `check` warns when there are uncommitted changes.

7. **The initial agent's evaluation**, shared by every run. Either evaluate it on this machine
   (about 1 hour):
   ```bash
   ./run_all.sh init
   ```
   or restore the original server's cache (`initial_polyglot/`, about 590 MB, provided
   separately) into `repos/HGM/initial_polyglot/`. Use the cache only if your results will be
   compared with runs from the original server; runs on this machine alone should use their
   own evaluation.

## Running

```bash
./run_all.sh pilot   # optional: one short HGM run (50 task evaluations) to see everything work
./run_all.sh v2      # the full plan in RERUN_V2.md: 10 searches, then final evaluations (~6–7 days)
```

`RERUN_V2.md` explains what `v2` runs, how to watch it, how to resume after an interruption,
and how to produce the comparison afterwards. Run outputs go to `repos/HGM/output_hgm/<label>/`
and `runs/`; the cost ledger is `runs/ledger.jsonl`.

## Tests

```bash
(cd instrumentation && python -m unittest discover -s tests)
(cd instrumentation/early_stop && python -m unittest discover -p 'test_*.py')
(cd erl && python -m unittest discover -p 'test_*.py')
```

## Licences

`repos/HGM/` is Apache-2.0 (its `LICENSE` is kept unchanged; changes are listed in
`repos/HGM/MODIFICATIONS.md`). HGM itself builds on DGM (Apache-2.0); SWE-bench and Aider's
polyglot benchmark are fetched at setup, not redistributed here.
