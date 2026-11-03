"""Parse DGM / HGM coding-agent chat logs into step sequences with outcome labels.

A predictions directory holds, per task, `<task>.md` (the agent's chat history, with
"Tool Used / Tool Input / Tool Result" blocks) and `<task>.json` (the outcome in
`eval_result`: resolved / unresolved / empty_patch / ...). DGM and HGM share this layout.
Finished efficient-RSI runs gzip large files, so `<task>.md.gz` / `<task>.json.gz` are read too.

Two log formats occur: the DGM repo's OpenAI Responses API (`Response(... input_tokens=,
output_tokens= ...)`, first call on a `Response(` line) and Chat Completions through vLLM
(`ChatCompletion(... completion_tokens=, prompt_tokens=, ... cached_tokens= ...)`, every
call on a `Tool Response:` line, the first one before any tool is used).
"""

import gzip
import json
import os
import re
from dataclasses import dataclass, field

USAGE_RE = re.compile(r"input_tokens=(\d+).*?output_tokens=(\d+)")
CHAT_IN_RE = re.compile(r"\bprompt_tokens=(\d+)")
CHAT_OUT_RE = re.compile(r"\bcompletion_tokens=(\d+)")
CACHED_RE = re.compile(r"\bcached_tokens=(\d+)")
MARKERS = ("Tool Used: ", "Tool Input: ", "Tool Result: ", "Tool Response: ")


@dataclass
class Step:
    tool: str
    input: str
    result: str
    # tokens of the LLM call made right after this step (the call that chose the next action)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # prefix-cache hits among input_tokens (vLLM); ~free on the GPU


@dataclass
class Trajectory:
    agent: str          # node / commit id of the agent that produced it
    task: str
    outcome: str        # raw eval_result
    steps: list = field(default_factory=list)
    first_call_tokens: tuple = (0, 0)   # (input, output) of the call that chose the first action
    first_call_cached: int = 0

    @property
    def solved(self):
        return self.outcome == "resolved"

    @property
    def total_tokens(self):
        return (self.first_call_tokens[0] + sum(s.input_tokens for s in self.steps),
                self.first_call_tokens[1] + sum(s.output_tokens for s in self.steps))


def _usage(line):
    """(input, output, cached) tokens of one logged call, in either log format."""
    cached = CACHED_RE.search(line)
    cached = int(cached.group(1)) if cached else 0
    m = USAGE_RE.search(line)
    if m:
        return int(m.group(1)), int(m.group(2)), cached
    i, o = CHAT_IN_RE.search(line), CHAT_OUT_RE.search(line)
    if i and o:
        return int(i.group(1)), int(o.group(1)), cached
    return 0, 0, 0


def parse_log(text):
    """Line-based parse: a Tool Result runs until the next marker line or a ===== section header.

    Returns (steps, (first-call input, output), first-call cached tokens)."""
    steps, first_call, first_cached, field_name, buf = [], (0, 0), 0, None, []
    current = None

    def flush():
        if current is not None and field_name in ("input", "result"):
            setattr(current, field_name, "\n".join(buf).strip())

    for line in text.split("\n"):
        if line.startswith("=" * 10):
            flush(); field_name, buf = None, []
            continue
        marker = next((m for m in MARKERS if line.startswith(m)), None)
        if marker is None:
            if field_name in ("input", "result"):
                buf.append(line)
            elif current is None and line.startswith("Response(") and first_call == (0, 0):
                *first_call, first_cached = _usage(line)
                first_call = tuple(first_call)
            continue
        flush(); buf = []
        value = line[len(marker):]
        if marker == "Tool Used: ":
            current = Step(value.strip(), "", "")
            steps.append(current)
            field_name = None
        elif marker == "Tool Input: ":
            field_name, buf = "input", [value]
        elif marker == "Tool Result: ":
            field_name, buf = "result", [value]
        else:  # Tool Response: usage of the call made after this step
            if current is not None:
                current.input_tokens, current.output_tokens, current.cached_tokens = _usage(value)
            elif first_call == (0, 0):  # Chat Completions logs: the call that chose the first action
                *first_call, first_cached = _usage(value)
                first_call = tuple(first_call)
            field_name = None
    flush()
    return steps, first_call, first_cached


def _find(pred_dir, name):
    """Path of name or name.gz in pred_dir, or None."""
    for n in (name, name + ".gz"):
        p = os.path.join(pred_dir, n)
        if os.path.exists(p):
            return p
    return None


def _read(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", errors="replace") as f:
            return f.read()
    with open(path, errors="replace") as f:
        return f.read()


def load_predictions(pred_dir, agent=None):
    """All trajectories in one predictions directory that have both a log and an outcome."""
    agent = agent or os.path.basename(os.path.normpath(pred_dir))
    tasks = sorted({n[:-len(ext)] for n in os.listdir(pred_dir) for ext in (".json", ".json.gz") if n.endswith(ext)})
    out = []
    for task in tasks:
        res_path, log_path = _find(pred_dir, task + ".json"), _find(pred_dir, task + ".md")
        if not res_path or not log_path:
            continue
        try:
            outcome = json.loads(_read(res_path)).get("eval_result", "")
        except ValueError:
            continue
        steps, first_call, first_cached = parse_log(_read(log_path))
        out.append(Trajectory(agent, task, outcome, steps, first_call, first_cached))
    return out


def load_run(run_dir):
    """Trajectories of every node of a search run: <run_dir>/<node>/predictions/<node>_<i>/."""
    out = []
    for node in sorted(os.listdir(run_dir)):
        pred_root = os.path.join(run_dir, node, "predictions")
        if not os.path.isdir(pred_root):
            continue
        for sub in sorted(os.listdir(pred_root)):
            if os.path.isdir(os.path.join(pred_root, sub)):
                out += load_predictions(os.path.join(pred_root, sub), agent=node)
    return out
