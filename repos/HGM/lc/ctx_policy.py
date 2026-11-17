"""Context policies for harness-rsi (CLM: "context policy" as a search target of the harness).

A policy is a pure function from the stored message list to the message list sent to the model.
The stored state is never changed, so a policy is deterministic, replayable, and covered by the
request hash in rsi_record. LC_CTX_POLICY selects one:

  none      : send the stored messages unchanged (the baseline harness).
  trunc200  : planted defect. Every tool result is cut to 200 characters.
  last3     : tool results older than the last 3 tool rounds are replaced by a one-line stub.
  testfb    : test runner output (pytest) is reduced to the failing tests, the assertion lines
              and the summary line; the retry prompt gets the same reduction.
  knobs     : (default when LC_CTX_POLICY is unset) read harness_knobs.json next to this file
              (or the path in LC_KNOBS). This is the file a proposer edits first.

harness_knobs.json (all optional; a missing or null knob means "off"):
  tool_result_max_chars : int   cut every tool result to this many characters
  keep_tool_rounds      : int   replace tool results older than the last N tool rounds by a stub
  compress_test_output  : bool  reduce pytest output as in testfb
"""
import json
import os
import re
from pathlib import Path

KEEP_ROUNDS = 3
TRUNC = 200
_PRESETS = {
    "none": {},
    "trunc200": {"tool_result_max_chars": 200},
    "last3": {"keep_tool_rounds": 3},
    "testfb": {"compress_test_output": True},
}
_KNOB_KEYS = ("tool_result_max_chars", "keep_tool_rounds", "compress_test_output")


def name():
    return os.environ.get("LC_CTX_POLICY", "knobs")


def knobs():
    """The active knob values: a preset from LC_CTX_POLICY, else harness_knobs.json."""
    p = name()
    if p in _PRESETS:
        return dict(_PRESETS[p])
    if p != "knobs":
        raise ValueError(f"unknown LC_CTX_POLICY={p}")
    path = Path(os.environ.get("LC_KNOBS") or Path(__file__).with_name("harness_knobs.json"))
    try:
        k = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    return {x: k.get(x) for x in _KNOB_KEYS if k.get(x) not in (None, False, 0)}


def apply(messages):
    k = knobs()
    if not k:
        return messages
    out = messages
    if k.get("keep_tool_rounds"):
        out = _last_rounds(out, int(k["keep_tool_rounds"]))
    if k.get("compress_test_output"):
        out = [_with(m, compress_test_output(m["content"])) if _is_tool(m) else m for m in out]
    if k.get("tool_result_max_chars"):
        n = int(k["tool_result_max_chars"])
        out = [_with(m, _trunc(m["content"], n)) if _is_tool(m) else m for m in out]
    return out


def compress_tests():
    """True when test feedback (tool results and retry prompts) is to be reduced."""
    return bool(knobs().get("compress_test_output"))


def _is_tool(m):
    return m.get("role") == "tool" and isinstance(m.get("content"), str)


def _with(m, content):
    if content == m["content"]:
        return m
    m2 = dict(m)
    m2["content"] = content
    return m2


def _trunc(s, n=TRUNC):
    if len(s) <= n:
        return s
    return s[:n] + f"\n[truncated {len(s) - n} chars]"


def _last_rounds(messages, keep):
    # A tool round is one assistant message with tool calls plus the tool results after it.
    rounds = [i for i, m in enumerate(messages) if m.get("role") == "assistant" and m.get("tool_calls")]
    if len(rounds) <= keep:
        return messages
    cutoff = rounds[-keep]                     # tool results before this index are old
    out = []
    for i, m in enumerate(messages):
        if i < cutoff and _is_tool(m) and len(m["content"]) > 120:
            out.append(_with(m, f"[earlier {m.get('name', 'tool')} output removed by context policy: "
                                f"{len(m['content'])} chars]"))
        else:
            out.append(m)
    return out


_PYTEST = re.compile(r"(test session starts|^=+ .*\b(passed|failed|error)s?\b.* in [\d.]+s)", re.M)
_HDR = re.compile(r"^(platform |cachedir|rootdir|plugins|collecting|configfile|test session starts)")
_RESULT = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")


def compress_test_output(s, limit=2000):
    """Keep every failing test name, then assertion lines (E ...), file:line of the failure and the
    summary line. Only the detail lines count against `limit`; the name list is never cut."""
    if not isinstance(s, str) or not _PYTEST.search(s):
        return s
    names, short, detail, n_pass, e_in_block = [], [], [], 0, 0
    for raw in s.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        r = _RESULT.match(line)
        if r:
            if r.group(2) == "PASSED":
                n_pass += 1
            else:
                names.append(f"{r.group(1)} {r.group(2)}")
            continue
        st = line.strip("= ").strip()
        if _HDR.match(st):
            continue
        if re.match(r"^_{3,} .* _{3,}$", line):           # failure block header
            detail.append("-- " + line.strip("_ "))
            e_in_block = 0
            continue
        if line.startswith("E "):
            e_in_block += 1
            if e_in_block <= 6:
                detail.append(line)
            continue
        if re.match(r"^\S+\.py:\d+: ", line):              # file:line: in func
            detail.append(line)
            continue
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            short.append(line[:160])                       # summary lines; used when -v lines are absent
            continue
        if re.match(r"^=+ .* in [\d.]+s", line):
            detail.append(st)
    head = ([f"[{n_pass} passing tests not shown]"] if n_pass else []) + (names or short)
    head_s = "\n".join(head)
    body = "\n".join(detail)
    room = max(limit - len(head_s) - 1, 400)
    if len(body) > room:
        body = body[:room] + "\n[... more failure details cut]"
    out = (head_s + "\n" + body).strip()
    return out if out else s[:limit]
