"""efficient-rsi entry point for little-coder (harness-rsi integration, 2026-11-16).

HGM runs this file inside its containers, with the same command line as efficient-rsi's
coding_agent_polyglot.py:

  solve:        python /hgm/lc/hgm_agent.py --problem_statement P --git_dir /testbed/ ...
  self-improve: python /hgm/lc/hgm_agent.py --problem_statement P --git_dir /hgm/lc/ ... --self_improve

It runs one little-coder attempt in --git_dir, then writes
  <outdir>/model_patch.diff  git diff against --base_commit, untracked files included, files named
                             in HGM_PROTECTED_FILES (the visible read-only tests) excluded
  <outdir>/rsi_attempt.json  rsi_record's attempt summary (calls, divergence point, live calls and
                             seconds, tokens) and the protocol guard's findings
  --chat_history_file        a readable log in efficient-rsi's "Tool Used / Tool Input / Tool Result"
                             format, which its diagnosis and early-stop parsers read
and exits 0 (efficient-rsi scores a non-zero exit as "incomplete").

Recording and coupled evaluation are set by the harness through the environment:
  LC_RSI_MODE     record | replay | off      (default off)
  LC_RECORD_TO    record prefix written by this attempt (<prefix>.jsonl, <prefix>.req.jsonl.gz)
  LC_REPLAY_FROM  parent record prefix to replay from (replay mode)
  LC_ON_DIVERGE   live | stop               (default live: continue with the model after the
                                             divergence point; this is the coupled evaluation)
  HGM_TASK_ID     task id used as the record key

This file, rsi_record.py and providers.py are frozen: the self-improvement patch filter on the host
(efficient-rsi repos/HGM/lc_integration.py) rejects a patch that touches them, so a proposer cannot
change how runs are recorded, metered or diffed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_NET = re.compile(r"\b(curl|wget|pip\s+download|git\s+clone|http[s]?://)", re.I)


def _knobs():
    try:
        return json.loads((HERE / "harness_knobs.json").read_text())
    except Exception:
        return {}


def _base_url(model):
    """Same route as efficient-rsi llm.py create_client() for vLLM models."""
    proxy = os.getenv("RSI_PROXY_PREFIX")
    if proxy:
        return f"{proxy}/p/vllm/v1"
    if os.getenv("OPENAI_BASE_URL"):
        return os.environ["OPENAI_BASE_URL"]
    return f"http://{model[11:]}:8000/v1"


def diff_versus_commit(git_dir, commit, exclude=()):
    """git diff against commit, untracked files included (as efficient-rsi utils/git_utils.py)."""
    out = subprocess.run(["git", "-C", git_dir, "diff", commit], capture_output=True, text=True).stdout
    untracked = subprocess.run(["git", "-C", git_dir, "ls-files", "--others", "--exclude-standard"],
                               capture_output=True, text=True).stdout.splitlines()
    for f in untracked:
        if f in exclude or f.startswith("__pycache__") or "/__pycache__/" in f or f.endswith(".pyc"):
            continue
        r = subprocess.run(["git", "-C", git_dir, "diff", "--no-index", "/dev/null", f],
                           capture_output=True, text=True)
        out += r.stdout
    if exclude:   # drop diff blocks of protected files that git tracks
        blocks = re.split(r"(?=^diff --git )", out, flags=re.M)
        out = "".join(b for b in blocks
                      if not any(re.match(rf"diff --git a/{re.escape(x)} ", b) for x in exclude))
    return out


def build_prompt(args):
    if args.self_improve:
        # The host writes the whole instruction (lc_integration.proposer_instruction): it points
        # at the workspace files instead of pasting logs into the prompt.
        return args.problem_statement
    tests = ""
    if os.getenv("HGM_PROTECTED_FILES"):
        tests = ("\nThe task's test files are in the repository and are read-only: "
                 + ", ".join(os.environ["HGM_PROTECTED_FILES"].split(",")) + ".\n")
    return (f"I have uploaded a code repository in the directory {args.git_dir}. Help solve the "
            f"following problem.\n\n<problem_description>\n{args.problem_statement}\n</problem_description>\n\n"
            f"Your task is to make changes to the files in the {args.git_dir} directory to address the "
            f"<problem_description>. I have already taken care of the required dependencies.\n{tests}"
            + (f"\n{args.test_description}\n" if args.test_description else "")
            + "\nPlease analyze the problem description carefully. Then make edits to the code files "
              "to complete the instruction.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem_statement", required=True)
    p.add_argument("--git_dir", required=True)
    p.add_argument("--base_commit", required=True)
    p.add_argument("--chat_history_file", required=True)
    p.add_argument("--outdir", default="/hgm/")
    p.add_argument("--test_description", default=None)
    p.add_argument("--self_improve", action="store_true")
    p.add_argument("--language", default="python")
    p.add_argument("--model", default="vllm-qwen3c-local")
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    t0 = time.time()
    deadline = t0 + 0.9 * args.timeout          # efficient-rsi's agent stops at 90% of the timeout

    os.environ["LLAMACPP_BASE_URL"] = _base_url(args.model)
    os.chdir(args.git_dir)

    import agent
    import rsi_record
    from config import load_config
    from context import build_system_prompt
    from local.config import get_model_profile, is_small_model

    lc_model = "llamacpp/" + args.model
    cfg = load_config()
    cfg["model"] = lc_model
    cfg["permission_mode"] = "accept-all"
    if is_small_model(lc_model):
        for k in get_model_profile(lc_model):
            cfg.pop(k, None)                    # the model profile is authoritative (see benchmarks)
    system = build_system_prompt(cfg)

    knobs = _knobs()
    max_turns = int(knobs.get("max_turns_self_improve" if args.self_improve else "max_turns") or 40)
    protected = [x for x in os.getenv("HGM_PROTECTED_FILES", "").split(",") if x]
    task = os.getenv("HGM_TASK_ID") or ("self_improve" if args.self_improve else "task")
    mode = os.getenv("LC_RSI_MODE", "off")
    rec_to = os.getenv("LC_RECORD_TO") or None
    rep_from = os.getenv("LC_REPLAY_FROM") or None
    if rec_to:
        Path(rec_to).parent.mkdir(parents=True, exist_ok=True)
    rsi_record.begin(task, 1, mode=mode, record=rec_to,
                     replay=(rep_from + ".jsonl") if (mode == "replay" and rep_from) else None,
                     replay_log=(rec_to + ".replay") if rec_to else None,
                     on_diverge=os.getenv("LC_ON_DIVERGE", "live"), record_in_replay=True)

    log = open(args.chat_history_file, "w")
    state = agent.AgentState()
    guard, error, stop_reason = [], None, "normal"
    try:
        for e in agent.run(build_prompt(args), state, cfg, system):
            if isinstance(e, agent.TextChunk):
                log.write(e.text)
            elif isinstance(e, agent.ToolStart):
                s = json.dumps(e.inputs, default=str)
                log.write(f"\nTool Used: {e.name}\nTool Input: {s}\n")
                if e.name in ("WebSearch", "WebFetch") or (e.name == "Bash" and _NET.search(s)):
                    guard.append({"tool": e.name, "input": s[:300]})
            elif isinstance(e, agent.ToolEnd):
                log.write(f"Tool Result: {str(e.result)[:20000]}\n")
            log.flush()
            if state.turn_count >= max_turns:
                stop_reason = "turn_limit"
                break
            if time.time() > deadline:
                stop_reason = "soft_deadline"
                break
    except rsi_record.StopReplay as ex:
        stop_reason = "replay_stop"
        error = f"replay_stop: {ex}"
    except Exception as ex:                     # the attempt still ends with a diff and exit 0
        stop_reason = "agent_exception"
        error = f"{type(ex).__name__}: {ex}"
        log.write(f"\nAGENT ERROR: {error}\n")
    summary = rsi_record.end()
    log.close()

    patch = diff_versus_commit(args.git_dir, args.base_commit, exclude=protected)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "model_patch.diff").write_text(patch)
    (Path(args.outdir) / "rsi_attempt.json").write_text(json.dumps(
        {"rsi": summary, "turns": state.turn_count, "error": error, "guard": guard,
         "scope": "attempt", "stop_reason": stop_reason, "max_turns": max_turns,
         "seconds": round(time.time() - t0, 2), "self_improve": args.self_improve}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())  # startup failures must remain observable to the host
