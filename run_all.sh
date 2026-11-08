#!/usr/bin/env bash
# Efficient-RSI experiments on Polyglot, all local: one Qwen3-Coder-30B-A3B (FP8) on vLLM
# plays every role (self-improve, diagnose, solve). No API calls.
#
# Usage (from the "efficient RSI/" directory, inside tmux):
#   ./run_all.sh vllm      # window 1: start vLLM (foreground)
#   ./run_all.sh proxy     # window 2: start the metering proxy (foreground)
#   ./run_all.sh gpu       # window 3: start the GPU sampler (foreground)
#   ./run_all.sh check     # sanity checks; run once the three services are up
#   ./run_all.sh init      # evaluate the initial agent once with the local model (shared by all runs)
#   ./run_all.sh pilot     # one HGM-fixed run with 50 task evaluations
#   ./run_all.sh search    # 3 configs x 3 repeats x 800 task evaluations
#   ./run_all.sh final     # pick the final agent of every run and evaluate it on the 166 held-out tasks
#   ./run_all.sh checkpoints  # anytime curve: same, at fractions of the common GPU budget (repeat 0 only)
#   ./run_all.sh noise     # experiment 1: the initial agent, NOISE_PASSES more passes over the 59 search tasks
#   ./run_all.sh tts       # experiment 2: the initial agent, TTS_PASSES more passes over the 166 held-out
#                          # tasks (run after final, which holds the first held-out pass)
#   ./run_all.sh report    # per-run cost breakdown + cross-run comparison + anytime curve + noise/TTS
#   ./run_all.sh archive-ledger <label>   # set an aborted run's ledger events aside before rerunning it
#   ./run_all.sh v2        # the corrected rerun (RERUN_V2.md): searches, then final evaluations
#
# Knobs (environment variables):
#   REPEATS="0 1 2"   repeat labels (hgm.py has no --seed; these are independent repeats)
#   CONFIGS="hgm-fixed hgm-orig dgm"   (also: hgm-erl = hgm-fixed + erl/online_stop.py node stopping,
#                     predictor history from runs/erl_history.json if present (a run output, built with
#                     erl/online_stop.py dump-history; not in git); hgm-erlx = hgm-erl where retired
#                     children free their expansion slot; neither is in the default list)
#   HGM_ERL_AUDIT=0.1 HGM_ERL_AUDIT_N=5   hgm-erl*: audit this share of stop decisions (off by default;
#                     hgm-erlxa = hgm-erlx with 10% audited)
#   HGM_MAX_HOURS=    HGM runs also stop (no new expansion or evaluation) after this many hours
#   HGM_DIAG_CLIP=1   clip the diagnose prompt to fit the model's context (v2 fix; see RERUN_V2.md)
#   HGM_HIDE_ANSWERS=1  keep each task's tests and reference solution out of the agent's container
#                     while it solves the task (polyglot/test_spec.py); images get a ".hidden" name
#   FINAL_HOURS=      final: cut every run at these GPU hours instead of the shortest run's
#                     (hgm-erl runs are cut at the first nine runs' 9.646 h)
#   PARALLEL=0        1 = run the configs of one repeat concurrently. The pilot showed one run
#                     already keeps the H100 at ~98%, so concurrency adds queueing, not throughput
#   BUDGET=800        task evaluations per search run (when hgm.py stops; its own count differs
#                     from the ledger's, so runs do not end on exactly the same number)
#   FINAL_CUT=hours   where final/report cut each search: "hours" = the GPU-occupancy hours of the
#                     shortest run (equal GPU budget, the primary cost measure); "evals" = BUDGET
#   DGM_MAX_HOURS=17  DGM runs also stop (between generations) after this many hours: DGM keeps
#                     the GPU idle much of the time, so 800 evaluations take it ~30 h, and comparisons
#                     cut at equal GPU hours never use the rest. Empty = no cap
#   CKPT_REPEATS="0" CKPT_FRACS="0.33 0.67"   checkpoints: which repeats, which budget fractions
#   NOISE_PASSES=3 TTS_PASSES=3   passes for the noise and test-time-scaling experiments
#   VLLM_BIN=vllm     vLLM executable
#   BRIDGE=172.17.0.1 docker0 bridge address the proxy binds to (differs on some Docker setups)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HGM="$ROOT/repos/HGM"
INST="$ROOT/instrumentation"
CONFIG="config_local.yaml"
MODEL_HF="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
MODEL_NAME="vllm-qwen3c-local"   # must contain "vllm" and must not contain "gpt" (llm.py routing)
VLLM_PORT=8000
PROXY_PORT=8787
BRIDGE="${BRIDGE:-172.17.0.1}"   # docker0 address (ip -4 addr show docker0); containers reach the proxy here

REPEATS="${REPEATS:-0 1 2}"
CONFIGS="${CONFIGS:-hgm-fixed hgm-orig dgm}"
PARALLEL="${PARALLEL:-0}"
BUDGET="${BUDGET:-800}"
FINAL_CUT="${FINAL_CUT:-hours}"
DGM_MAX_HOURS="${DGM_MAX_HOURS-17}"
CKPT_REPEATS="${CKPT_REPEATS:-0}"
CKPT_FRACS="${CKPT_FRACS:-0.33 0.67}"
NOISE_PASSES="${NOISE_PASSES:-3}"
TTS_PASSES="${TTS_PASSES:-3}"
VLLM_BIN="${VLLM_BIN:-vllm}"   # on engr-liu01s: ~/venvs/rsi/bin/vllm (the anaconda one has a numpy ABI clash)

# Server-side settings (TMPDIR and caches off the full system disk, git identity) live in env.sh.
[[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
mkdir -p "$ROOT/runs"
export RSI_LEDGER="$ROOT/runs/ledger.jsonl"
# The proxy binds to docker0 only, so host-side callers must use the bridge address too, not 127.0.0.1.
export RSI_PROXY_URL="http://$BRIDGE:$PROXY_PORT"
export RSI_PROXY_URL_CONTAINER="$RSI_PROXY_URL"
# Any call that slips past the local model fails loudly instead of spending money.
export OPENAI_API_KEY="sk-invalid-local-only"
unset OpenRouter_API_KEY ANTHROPIC_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY || true

# The initial agent's cached results carry no model label; this marker records which model made them
# and whether the answers were hidden (HGM_HIDE_ANSWERS), so a cache made otherwise is never reused.
INIT_MARKER="$HGM/initial_polyglot/default_agent/.evaluated_with"
INIT_KEY="$MODEL_HF hide_answers=${HGM_HIDE_ANSWERS:-0}"
# Path of the final_result.json that holds the initial agent's held-out evaluation (written by `final`).
INIT_HELDOUT="$ROOT/runs/initial_heldout_result.path"

# A running search: a python process whose script is hgm.py (not a shell or grep that mentions it).
SEARCH_PREFIX='^[^ ]*python[0-9.]* hgm\.py'
SEARCH_RE="$SEARCH_PREFIX( |\$)"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$ROOT/runs/run_all.log"; }

record_setup() {  # what produced this run, for the experiment log
  local label=$1 f="$ROOT/runs/$1.setup.txt"
  {
    echo "label: $label"; date
    echo "HGM commit: $(git -C "$HGM" rev-parse HEAD) dirty: $(git -C "$HGM" status --porcelain | wc -l | tr -d ' ') files"
    echo "HGM_FIX_DESCENDANT_EVALS=${HGM_FIX_DESCENDANT_EVALS:-1}"
    echo "HGM_DIAG_CLIP=${HGM_DIAG_CLIP:-0} HGM_MAX_HOURS=${HGM_MAX_HOURS:-} DGM_MAX_HOURS=${DGM_MAX_HOURS:-} HGM_HIDE_ANSWERS=${HGM_HIDE_ANSWERS:-0}"
    echo "HGM_ERL_STOP=${HGM_ERL_STOP:-0} HGM_ERL_EXPAND=${HGM_ERL_EXPAND:-0} HGM_ERL_HISTORY=${HGM_ERL_HISTORY:-} W=${HGM_ERL_W:-auto} ALPHA=${HGM_ERL_ALPHA:-0.2} MIN_EVALS=${HGM_ERL_MIN_EVALS:-3} AUDIT=${HGM_ERL_AUDIT:-0} AUDIT_N=${HGM_ERL_AUDIT_N:-5}"
    [[ -n "${HGM_ERL_HISTORY:-}" ]] && echo "history sha256: $(sha256sum "$HGM_ERL_HISTORY" | cut -c1-16)"
    echo "config: $CONFIG"; cat "$HGM/$CONFIG"
    echo "vllm cmd: $(cat "$ROOT/runs/vllm_cmd.txt" 2>/dev/null || echo unknown)"
  } > "$f"
  # Exact code of an uncommitted tree: diff against HEAD plus the untracked source files.
  git -C "$HGM" diff HEAD > "$ROOT/runs/$label.hgm.diff"
  (cd "$HGM" && git ls-files --others --exclude-standard -- '*.py' '*.yaml' | tar czf "$ROOT/runs/$label.hgm-untracked.tgz" -T -)
  if [[ "${HGM_ERL_STOP:-0}" == 1 ]]; then tar czf "$ROOT/runs/$label.erl.tgz" -C "$ROOT" --exclude __pycache__ erl; fi
}

check_init_marker() {
  if [[ ! -f "$INIT_MARKER" ]] || [[ "$(cat "$INIT_MARKER")" != "$INIT_KEY" ]]; then
    echo "initial_polyglot/default_agent was not evaluated with $INIT_KEY; run './run_all.sh init' first" >&2
    exit 1
  fi
}

search_run() {  # search_run <config> <label> <budget>
  local cfg=$1 label=$2 budget=$3 extra=() fix=1 erl=0 erl_hist="" erlx=0 audit="${HGM_ERL_AUDIT:-0}"
  case "$cfg" in
    hgm-fixed) ;;
    hgm-orig)  fix=0 ;;
    hgm-erl)   erl=1 erl_hist="$ROOT/runs/erl_history.json" ;;
    hgm-erlx)  erl=1 erl_hist="$ROOT/runs/erl_history.json" erlx=1 ;;   # + retired children free expansion slots
    hgm-erlxa) erl=1 erl_hist="$ROOT/runs/erl_history.json" erlx=1 audit=0.1 ;;   # + audit 10% of stops
    dgm)       extra=(--search dgm) ;;
    *) echo "unknown config $cfg" >&2; return 1 ;;
  esac
  # The history is a run output (not in git): build it from finished runs with
  # `python erl/online_stop.py dump-history ...`. Without it the predictor trains on this search alone.
  if [[ -n "$erl_hist" && ! -f "$erl_hist" ]]; then
    log "$label: no $erl_hist; the stopping predictor trains on this search only (see erl/README.md)"
    erl_hist=""
  fi
  [[ "$cfg" == dgm && -n "$DGM_MAX_HOURS" ]] && extra+=(--max_hours "$DGM_MAX_HOURS")
  [[ "$cfg" != dgm && -n "${HGM_MAX_HOURS:-}" ]] && extra+=(--max_hours "$HGM_MAX_HOURS")
  local out="output_hgm/$label"
  if [[ -f "$ROOT/runs/$label.complete" ]]; then log "$label: already complete, skipping"; return 0; fi
  # Handover from a previous run_all.sh that was stopped while this run's hgm.py kept going:
  # wait for it, then take the run as it finished instead of starting it again.
  if pgrep -f -- "$SEARCH_PREFIX .*--output_dir $out( |\$)" > /dev/null; then
    log "$label: still running under a previous runner, waiting for it"
    while pgrep -f -- "$SEARCH_PREFIX .*--output_dir $out( |\$)" > /dev/null; do sleep 60; done
  fi
  if finished "$label" "$out" "$budget"; then
    touch "$ROOT/runs/$label.complete"
    log "$label: done ($(evals_done "$label") task evaluations$(stop_note "$out"); finished under a previous runner)"
    compress_outputs "$HGM/$out"
    return 0
  fi
  if [[ -f "$HGM/$out/hgm_outer.log" ]]; then
    log "$label: output exists, continuing it"
    extra+=(--continue_from "$out")
  fi
  HGM_ERL_STOP=$erl HGM_ERL_HISTORY=$erl_hist HGM_ERL_EXPAND=$erlx HGM_ERL_AUDIT=$audit HGM_FIX_DESCENDANT_EVALS=$fix record_setup "$label"
  local cap=""
  [[ "$cfg" == dgm && -n "$DGM_MAX_HOURS" ]] && cap=" or $DGM_MAX_HOURS h"
  [[ "$cfg" != dgm && -n "${HGM_MAX_HOURS:-}" ]] && cap=" or $HGM_MAX_HOURS h"
  log "$label: start ($cfg, budget $budget$cap)"
  local rc=0
  (cd "$HGM" && HGM_ERL_STOP=$erl HGM_ERL_HISTORY=$erl_hist HGM_ERL_EXPAND=$erlx HGM_ERL_AUDIT=$audit HGM_FIX_DESCENDANT_EVALS=$fix RSI_RUN="$label" python hgm.py --config "$CONFIG" --polyglot \
      --max_task_evals "$budget" --output_dir "$out" "${extra[@]}" 2>&1 \
      | capped_log "$ROOT/runs/$label.stdout.log") || rc=$?
  # A run counts as finished only if it spent its budget (task evaluations, or the DGM hour cap);
  # otherwise (crash, full disk quota) stop instead of starting the next run on a broken setup.
  if (( rc != 0 )) || ! finished "$label" "$out" "$budget"; then
    log "$label: FAILED (exit $rc, $(evals_done "$label")/$budget task evaluations); see runs/$label.stdout.log"
    return 1
  fi
  touch "$ROOT/runs/$label.complete"
  log "$label: done ($(evals_done "$label") task evaluations$(stop_note "$out"))"
  compress_outputs "$HGM/$out"
}

evals_done() {  # evals_done <label> [phase]: eval events of a run in the ledger (default phase agent_run, as analyze_ledger counts)
  python3 - "$RSI_LEDGER" "$1" "${2:-agent_run}" <<'EOF'
import json, sys
n = 0
for line in open(sys.argv[1]):
    e = json.loads(line)
    n += (e.get("run") == sys.argv[2] and e.get("type") == "eval" and e.get("phase") == sys.argv[3]
          and e.get("node") != "initial")
print(n)
EOF
}

finished() {  # finished <label> <out> <budget>: the run spent its budget (evaluations or the DGM hour cap)
  local n
  # Count started evaluations (setup phase): one whose container fails before the agent runs is
  # budgeted and written as setup but has no agent_run event (76 of 800 in hgm-orig-r2). 95% still
  # catches a run that crashed or stopped early.
  n=$(evals_done "$1" setup)
  (( n * 100 >= $3 * 95 )) || [[ -f "$HGM/$2/stopped_on_time.json" ]]
}

stop_note() {  # ", stopped at the DGM hour cap after X h" when the run ended on time
  [[ -f "$HGM/$1/stopped_on_time.json" ]] || return 0
  python3 -c 'import json, sys
s = json.load(open(sys.argv[1]))
print(", stopped at the %s h cap after %s h" % (s["max_hours"], s["hours"]), end="")' "$HGM/$1/stopped_on_time.json"
}

common_hours() {  # GPU-occupancy hours of the shortest search run: the equal-GPU-budget cut for all of them
  local labels=()
  for r in $REPEATS; do for c in $CONFIGS; do labels+=("$c-r$r"); done; done
  (cd "$INST" && python3 - "$RSI_LEDGER" "${labels[@]}" <<'EOF'
import sys
import analyze_ledger
events = analyze_ledger.load(sys.argv[1])
hours = [analyze_ledger.gpu_hours([e for e in events if e.get("run") == r]) for r in sys.argv[2:]]
if min(hours) == 0:
    sys.exit("some search run has no events in the ledger")
print(f"{min(hours):.3f}")
EOF
  )
}

cut_args() {  # final_eval.py / analyze_ledger.py flags for the FINAL_CUT budget
  local h
  if [[ -n "${FINAL_HOURS:-}" ]]; then echo "--max-hours $FINAL_HOURS"
  elif [[ "$FINAL_CUT" == hours ]]; then h=$(common_hours) || return 1; echo "--max-hours $h"
  else echo "--max-evals $BUDGET"; fi
}

capped_log() {  # keep at most two ~200 MB files of a run's console output (<file> and <file>.1)
  python3 -u -c '
import os, sys
path, cap = sys.argv[1], 200 * 2**20
f = open(path, "ab")
for line in sys.stdin.buffer:
    f.write(line)
    if f.tell() > cap:
        f.close(); os.replace(path, path + ".1"); f = open(path, "ab")
' "$1"
}

compress_outputs() {  # gzip a finished run's large per-task logs; the search no longer reads them
  find "$1" -path '*/predictions/*' -type f -size +1M ! -name '*.gz' -exec gzip -q {} + || true
  log "$(basename "$1"): predictions compressed, now $(du -sh "$1" | cut -f1)"
}

case "${1:-}" in
  vllm)
    cmd=("$VLLM_BIN" serve "$MODEL_HF" --port "$VLLM_PORT" --served-model-name "$MODEL_NAME"
         --enable-auto-tool-choice --tool-call-parser qwen3_coder
         --max-model-len 131072 --gpu-memory-utilization 0.92 --enable-prefix-caching
         --enable-prompt-tokens-details)   # reports cached (prefix-hit) prompt tokens in usage
    # DeepGEMM's FP8 MoE kernels JIT-compile against a CUDA toolkit; engr-liu01s only has 12.4
    # while this vLLM is built for cu13, so fall back to vLLM's other FP8 kernels.
    # FlashInfer's sampler is JIT-built with nvcc too, which the server lacks (same setting as RSI-Text-to-SQL).
    export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    echo "VLLM_USE_DEEP_GEMM=$VLLM_USE_DEEP_GEMM VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER ${cmd[*]}" \
      > "$ROOT/runs/vllm_cmd.txt"
    exec "${cmd[@]}"
    ;;

  proxy)
    exec python "$INST/llm_proxy.py" --bind "$BRIDGE" --port "$PROXY_PORT" \
      --vllm-upstream "http://127.0.0.1:$VLLM_PORT"
    ;;

  gpu)
    pids=$(pgrep -f '^[^ ]*python[0-9.]* [^ ]*vllm serve ' || true)
    [[ -n "$pids" ]] || { echo "vLLM is not running" >&2; exit 1; }
    args=(); for p in $pids; do args+=(--expect-pid "$p"); done
    exec python "$INST/gpu_sampler.py" --interval 5 "${args[@]}"
    ;;

  check)
    curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" | grep -q "$MODEL_NAME" \
      && echo "ok  vLLM serves $MODEL_NAME" || { echo "FAIL vLLM not serving $MODEL_NAME"; exit 1; }
    curl -s -o /dev/null "http://$BRIDGE:$PROXY_PORT/" && echo "ok  proxy reachable on $BRIDGE" \
      || { echo "FAIL proxy not reachable on $BRIDGE:$PROXY_PORT"; exit 1; }
    docker run --rm --add-host=host.docker.internal:host-gateway curlimages/curl \
      -s -o /dev/null -w "%{http_code}\n" "http://$BRIDGE:$PROXY_PORT/" >/dev/null \
      && echo "ok  containers reach the proxy" || { echo "FAIL containers cannot reach the proxy (firewall?)"; exit 1; }
    (cd "$INST" && python3 -m unittest discover -s tests -q) && echo "ok  instrumentation tests"
    [[ "${HGM_HIDE_ANSWERS:-0}" == 1 ]] && echo "ok  HGM_HIDE_ANSWERS=1 (tests and reference solutions hidden)" \
      || echo "WARN HGM_HIDE_ANSWERS is not 1: the solving agent can read each task's tests and solution (see env.sh.example)"
    [[ -z "$(git -C "$HGM" status --porcelain)" ]] && echo "ok  HGM tree is committed" \
      || echo "WARN HGM has uncommitted changes; commit them so every run maps to a commit"
    ;;

  init)
    # Evaluates the initial agent on all 59 search tasks once; every run then copies this cache.
    # Done up front so concurrent runs do not race to create it. Its cost is logged under
    # RSI_RUN=initial-eval and is shared by all methods, so it is left out of the comparison.
    if [[ -f "$INIT_MARKER" && "$(cat "$INIT_MARKER")" == "$INIT_KEY" ]]; then
      log "initial agent already evaluated with $INIT_KEY"; exit 0
    fi
    if [[ -d "$HGM/initial_polyglot/default_agent" ]]; then
      bak="$HGM/initial_polyglot/default_agent.bak.$(date +%s)"
      log "moving stale initial results to $bak"
      mv "$HGM/initial_polyglot/default_agent" "$bak"
    fi
    record_setup initial-eval
    log "initial-eval: start"
    # --search dgm with a zero budget runs only initialize_run(), which evaluates the initial agent.
    (cd "$HGM" && RSI_RUN=initial-eval python hgm.py --config "$CONFIG" --polyglot --search dgm \
        --max_task_evals 0 --output_dir output_hgm/initial-eval > "$ROOT/runs/initial-eval.stdout.log" 2>&1)
    echo "$INIT_KEY" > "$INIT_MARKER"
    log "initial-eval: done"
    ;;

  pilot)
    check_init_marker
    search_run hgm-fixed pilot-hgm-fixed 50
    python "$INST/analyze_ledger.py" "$RSI_LEDGER" --run pilot-hgm-fixed
    # Every LLM call should have gone to vLLM.
    python3 - "$RSI_LEDGER" <<'EOF'
import json, sys, collections
c = collections.Counter()
for line in open(sys.argv[1]):
    e = json.loads(line)
    if e.get("type") == "llm_call" and e.get("run") == "pilot-hgm-fixed":
        c[e.get("provider", "?")] += 1
print("llm_call by provider:", dict(c))
if set(c) - {"vllm"}:
    sys.exit("FAIL: some calls did not go to the local vLLM")
EOF
    ;;

  search)
    check_init_marker
    for r in $REPEATS; do
      if [[ "$PARALLEL" == 1 ]]; then
        pids=()
        for c in $CONFIGS; do search_run "$c" "$c-r$r" "$BUDGET" & pids+=($!); done
        for p in "${pids[@]}"; do wait "$p" || log "a run of repeat $r exited with an error"; done
      else
        for c in $CONFIGS; do
          search_run "$c" "$c-r$r" "$BUDGET" || { log "stopping: $c-r$r failed"; exit 1; }
        done
      fi
    done
    ;;

  v2)
    # Corrected rerun (RERUN_V2.md): the diagnose prompt is clipped to fit the model, every search
    # stops after V2_HOURS hours (all comparisons cut at 9.646 GPU hours), labels end in "v2".
    check_init_marker
    if pgrep -f -- "$SEARCH_RE" > /dev/null; then echo "an hgm.py run is still going; stop it first" >&2; exit 1; fi
    export HGM_DIAG_CLIP=1 HGM_MAX_HOURS="${V2_HOURS:-10.5}" DGM_MAX_HOURS="${V2_HOURS:-10.5}"
    v2_repeats="${V2_REPEATS:-0v2 1v2 2v2}"
    budget="${V2_BUDGET:-2000}"   # high enough that the hour cap, not the budget, ends every search
    for r in $v2_repeats; do
      for c in hgm-fixed dgm hgm-erlx; do
        search_run "$c" "$c-r$r" "$budget" || { log "v2 stopping: $c-r$r failed"; exit 1; }
      done
    done
    first="${v2_repeats%% *}"
    [[ -n "${V2_AUDIT_REPEATS-$first}" ]] && for r in ${V2_AUDIT_REPEATS-$first}; do
      search_run hgm-erlxa "hgm-erlxa-r$r" "$budget" || { log "v2 stopping: hgm-erlxa-r$r failed"; exit 1; }
    done
    log "v2: searches done; final evaluations at ${FINAL_HOURS:-9.646} GPU hours"
    for r in $v2_repeats; do for c in hgm-fixed dgm hgm-erlx; do
      FINAL_HOURS="${FINAL_HOURS:-9.646}" REPEATS="$r" CONFIGS="$c" "$0" final || { log "v2: final $c-r$r failed"; exit 1; }
    done; done
    for r in ${V2_AUDIT_REPEATS-$first}; do
      FINAL_HOURS="${FINAL_HOURS:-9.646}" REPEATS="$r" CONFIGS=hgm-erlxa "$0" final || { log "v2: final hgm-erlxa-r$r failed"; exit 1; }
    done
    log "v2: done"
    ;;

  archive-ledger)
    # ./run_all.sh archive-ledger <label>: move a run's events to runs/ledger.aborted-<label>.jsonl, so a
    # rerun under the same label starts from a clean ledger (nothing is deleted).
    label="${2:?usage: ./run_all.sh archive-ledger <label>}"
    python3 - "$RSI_LEDGER" "$label" "$ROOT/runs/ledger.aborted-$label.jsonl" <<'EOF'
import json, os, sys
src, label, dst = sys.argv[1:]
keep, moved = [], []
for line in open(src):
    (moved if json.loads(line).get("run") == label else keep).append(line)
with open(dst, "a") as f:
    f.writelines(moved)
with open(src + ".tmp", "w") as f:
    f.writelines(keep)
os.replace(src + ".tmp", src)
print(f"{label}: moved {len(moved)} events to {dst}, kept {len(keep)}")
EOF
    ;;

  final)
    # The initial agent is the same for every run, so only the first final evaluation scores it on
    # the held-out tasks; later ones pass --skip-initial and compare.py reuses that result (--initial).
    cutstr=$(cut_args) || exit 1
    read -r -a cut <<< "$cutstr"
    log "final: search runs cut at ${cut[*]}"
    for r in $REPEATS; do for c in $CONFIGS; do
      label="$c-r$r"
      [[ -f "$HGM/output_hgm/$label/final_eval/final_result.json" ]] && { log "$label: final done already"; continue; }
      skip=()
      [[ -f "$INIT_HELDOUT" ]] && skip=(--skip-initial)
      log "$label: final eval ${skip[*]}"
      (cd "$HGM" && python final_eval.py --config "$CONFIG" --output_dir "output_hgm/$label" --run "$label" \
          --polyglot "${cut[@]}" "${skip[@]}" 2>&1 | capped_log "$ROOT/runs/$label.final.log")
      compress_outputs "$HGM/output_hgm/$label/final_eval"
      [[ -f "$INIT_HELDOUT" ]] || echo "$HGM/output_hgm/$label/final_eval/final_result.json" > "$INIT_HELDOUT"
    done; done
    # The --rule mean sensitivity check is not run here; it would need its own --name.
    ;;

  checkpoints)
    # Anytime curve: held-out accuracy of the agent the selection rule picks at fractions of the
    # common GPU budget. With the initial agent (0 h) and final (100%) this gives four points per run.
    # Each point is one held-out evaluation (166 tasks), so by default only repeat 0 gets them.
    total=$(common_hours)
    for r in $CKPT_REPEATS; do for c in $CONFIGS; do for f in $CKPT_FRACS; do
      label="$c-r$r" name="ckpt-$f"
      [[ -f "$HGM/output_hgm/$label/$name/final_result.json" ]] && { log "$label: $name done already"; continue; }
      hours=$(python3 -c "print(round($total * $f, 3))")
      log "$label: $name at $hours of $total GPU hours"
      (cd "$HGM" && python final_eval.py --config "$CONFIG" --output_dir "output_hgm/$label" --run "$label" \
          --polyglot --max-hours "$hours" --name "$name" --skip-initial 2>&1 | capped_log "$ROOT/runs/$label.$name.log")
      compress_outputs "$HGM/output_hgm/$label/$name"
    done; done; done
    ;;

  noise)
    # Experiment 1: how much one task's outcome varies between passes of the same agent. The
    # search runs reuse one cached pass of the initial agent, so these are passes 2.. of it.
    log "noise-search: $NOISE_PASSES passes of the initial agent over the search tasks"
    (cd "$HGM" && python repeat_eval.py --config "$CONFIG" --polyglot --tasks search \
        --repeats "$NOISE_PASSES" --label noise-search 2>&1 | capped_log "$ROOT/runs/noise-search.log")
    compress_outputs "$HGM/output_hgm/noise-search"
    log "noise-search: done"
    ;;

  tts)
    # Experiment 2: what the search's GPU hours would buy as repeated attempts of the initial agent.
    [[ -f "$INIT_HELDOUT" ]] || { echo "run ./run_all.sh final first (it holds the first held-out pass)" >&2; exit 1; }
    log "tts-heldout: $TTS_PASSES more passes of the initial agent over the held-out tasks"
    (cd "$HGM" && python repeat_eval.py --config "$CONFIG" --polyglot --tasks heldout \
        --repeats "$TTS_PASSES" --label tts-heldout 2>&1 | capped_log "$ROOT/runs/tts-heldout.log")
    compress_outputs "$HGM/output_hgm/tts-heldout"
    log "tts-heldout: done"
    ;;

  report)
    cutstr=$(cut_args) || exit 1
    read -r -a cut <<< "$cutstr"
    for r in $REPEATS; do for c in $CONFIGS; do
      label="$c-r$r"
      python "$INST/analyze_ledger.py" "$RSI_LEDGER" --run "$label" "${cut[@]}" \
        --nodes-csv "$ROOT/runs/$label-nodes.csv" > "$ROOT/runs/$label.cost.txt"
    done; done
    args=()
    for c in $CONFIGS; do for r in $REPEATS; do
      f="$HGM/output_hgm/$c-r$r/final_eval/final_result.json"
      [[ -f "$f" ]] && args+=(--result "$c=$f")
    done; done
    init_arg=()
    [[ -f "$INIT_HELDOUT" ]] && init_arg=(--initial "$(cat "$INIT_HELDOUT")")
    (cd "$INST" && python compare.py "${args[@]}" "${init_arg[@]}" --reference dgm --ledger "$RSI_LEDGER") \
      | tee "$ROOT/runs/compare.txt"
    # Anytime curve from whatever checkpoint and final results exist (initial agent = 0 GPU hours).
    python3 - "$HGM/output_hgm" "$(cat "$INIT_HELDOUT" 2>/dev/null)" $CONFIGS <<'EOF' | tee "$ROOT/runs/anytime.txt"
import glob, json, os, sys
out, init_path, configs = sys.argv[1], sys.argv[2], sys.argv[3:]
init = json.load(open(init_path))["initial_accuracy"] if init_path else None
print("\nAnytime curve: held-out accuracy of the selected agent vs GPU-occupancy hours of search\n")
print("| run | point | GPU hours | task evaluations | held-out accuracy |")
print("| --- | --- | --- | --- | --- |")
for run in sorted(os.path.basename(d) for d in glob.glob(f"{out}/*-r*") if os.path.basename(d).rsplit("-r", 1)[0] in configs):
    points = []
    for f in glob.glob(f"{out}/{run}/*/final_result.json"):
        r = json.load(open(f))
        name = os.path.basename(os.path.dirname(f))
        if name == "final_eval" or name.startswith("ckpt-"):
            points.append((r.get("cut", {}).get("gpu_hours", float("nan")), name, r))
    if not points:
        continue
    if init is not None:
        print(f"| {run} | initial | 0 | 0 | {100 * init:.1f}% |")
    for h, name, r in sorted(points, key=lambda p: p[0]):
        print(f"| {run} | {name} | {h:.2f} | {r.get('cut', {}).get('task_evals', '?')} | {100 * r['best_accuracy']:.1f}% |")
EOF
    # Experiments 1 and 2, if they ran.
    rs=()
    [[ -d "$HGM/output_hgm/noise-search" ]] && rs+=(--noise "$HGM/output_hgm/noise-search")
    if [[ -d "$HGM/output_hgm/tts-heldout" ]]; then
      rs+=(--tts "$HGM/output_hgm/tts-heldout")
      [[ -f "$INIT_HELDOUT" ]] && rs+=(--tts-extra "$(cat "$INIT_HELDOUT")")
      rs+=("${args[@]}")
    fi
    if (( ${#rs[@]} )); then
      (cd "$INST" && python repeat_stats.py --ledger "$RSI_LEDGER" "${rs[@]}") | tee "$ROOT/runs/noise_tts.txt"
    fi
    ;;

  *)
    sed -n '2,35p' "$0"; exit 1 ;;
esac
