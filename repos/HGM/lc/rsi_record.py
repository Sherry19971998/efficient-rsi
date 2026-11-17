"""Record / replay of LLM calls for harness-rsi (coupled evaluation).

The harness calls begin(...) before each attempt and end() after it. agent.run() sends every
model call through wrapped_stream(). Three modes, chosen per attempt:

  off     : pass through.
  record  : pass through; append one JSON line per call to <record>.jsonl and the full
            canonical request to <record>.req.jsonl.gz.
  replay  : serve the recorded responses in order instead of calling the model. Tools still
            run for real, so the next request is rebuilt by the harness. Each request is
            hashed and compared with the recorded one. On the first mismatch (the divergence
            point) the attempt either stops (on_diverge="stop") or continues with the live
            model (on_diverge="live"). Results go to <replay_log>.jsonl.

Two hashes per request: raw (byte-identical) and norm (volatile substrings such as test
durations masked). Self-replay of an unchanged harness should match on norm at every call.

2026-11-16 (experiment 03):
- The recorded norm hash is recomputed at replay time from the stored request text
  (<record>.req.jsonl.gz) with the current masking rules, so a new mask also applies to old
  records. Records without stored requests fall back to the stored norm_sha.
- gen_sha hashes the generation parameters (temperature, max_tokens, thinking budget). A call
  matches only if gen_sha matches too; records written before this field skip the check.
- After the divergence point with on_diverge="live", every live call is logged (event "live")
  with its duration and tokens, and attempt_end reports live_calls / live_s.

2026-11-16 (efficient-rsi integration):
- record_in_replay=True writes a complete record of the replayed attempt: served calls are copied
  from the parent record (served=True, no model time), live calls are recorded as usual. A child
  node's record then serves its own children, so coupling chains down the search tree.
"""
import gzip
import hashlib
import json
import re
import time

from providers import AssistantTurn, TextChunk

_VOLATILE = [
    (re.compile(r"\b\d+(\.\d+)?\s?(ms|s|sec|secs|seconds)\b"), "<T>"),          # durations
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?"), "<TS>"),     # timestamps
    (re.compile(r"0x[0-9a-fA-F]{6,}"), "<ADDR>"),                                # object addresses
    (re.compile(r"\bid='?\d{8,}'?"), "id=<ID>"),                                 # object ids in mock reprs
    # `ls -l` owner and group (the ICRN network filesystem shows nobody/nogroup or the uid at
    # random) and the column padding that depends on them
    (re.compile(r"([-dlcbps][rwxsStT-]{9}[+.@]?) +(\d+) +[^\s\\]+ +[^\s\\]+ +(\d+) +"), r"\1 \2 <OWN> \3 "),
    # `ls -l` modification times, e.g. "Nov 16 19:51" or "Nov 16  2025"
    (re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(\d{1,2}:\d{2}|\d{4})\b"), "<LSTIME>"),
]

_S = {"mode": "off"}
GEN_KEYS = ("temperature", "max_tokens", "thinking_budget", "top_p", "top_k")


class StopReplay(Exception):
    """Raised inside agent.run() when replay reaches the divergence point with on_diverge=stop."""


def canonical(model, system, messages, tool_schemas):
    return json.dumps({"model": model, "system": system, "messages": messages,
                       "tools": tool_schemas}, sort_keys=True, ensure_ascii=False, default=str)


def normalize(s):
    for pat, rep in _VOLATILE:
        s = pat.sub(rep, s)
    return s


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _gen_sha(config):
    return _sha(json.dumps({k: config.get(k) for k in GEN_KEYS}, sort_keys=True, default=str))


def begin(task, attempt, mode="off", record=None, replay=None, replay_log=None, on_diverge="stop",
          record_in_replay=False):
    _S.clear()
    _S.update(task=task, attempt=attempt, mode=mode, k=0, diverged=None, on_diverge=on_diverge,
              record_in_replay=record_in_replay,
              record=record, replay_log=replay_log, served=[], reqs={}, live=[],
              tokens={"in": 0, "out": 0, "cached": 0, "calls": 0, "think_exceeded": 0})
    if mode == "replay":
        with open(replay) as f:
            _S["served"] = [json.loads(l) for l in f if l.strip()]
            _S["served"] = [e for e in _S["served"] if e["task"] == task and e["attempt"] == attempt]
        gz = (replay[:-len(".jsonl")] if replay.endswith(".jsonl") else replay) + ".req.jsonl.gz"
        try:
            with gzip.open(gz, "rt", encoding="utf-8") as f:
                for l in f:
                    e = json.loads(l)
                    if e["task"] == task and e["attempt"] == attempt:
                        _S["reqs"][e["k"]] = e["req"]
        except FileNotFoundError:
            pass


def end():
    live = _S.get("live", [])
    out = {"task": _S.get("task"), "attempt": _S.get("attempt"), "mode": _S.get("mode"),
           "calls": _S.get("k", 0), "diverged_at": _S.get("diverged"),
           "recorded_calls": len(_S.get("served", [])), **_S.get("tokens", {}),
           "live_calls": len(live), "live_s": round(sum(x["t1"] - x["t0"] for x in live), 2)}
    if _S.get("mode") == "replay" and _S.get("replay_log"):
        with open(_S["replay_log"], "a") as f:
            f.write(json.dumps({"event": "attempt_end", **out}) + "\n")
    _S["mode"] = "off"
    return out


def _append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _append_gz(path, obj):
    with gzip.open(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _count(turn):
    t = _S["tokens"]
    t["calls"] += 1
    t["in"] += turn.in_tokens or 0
    t["out"] += turn.out_tokens or 0
    t["cached"] += getattr(turn, "cached_tokens", 0) or 0
    t["think_exceeded"] += int(bool(getattr(turn, "think_exceeded", False)))


def wrapped_stream(stream_fn, model, system, messages, tool_schemas, config):
    mode = _S.get("mode", "off")
    if mode == "off":
        for ev in stream_fn(model=model, system=system, messages=messages, tool_schemas=tool_schemas, config=config):
            if isinstance(ev, AssistantTurn):
                if "tokens" in _S:
                    _count(ev)
            yield ev
        return

    k = _S["k"]
    _S["k"] += 1
    req = canonical(model, system, messages, tool_schemas)
    raw, norm, gen = _sha(req), _sha(normalize(req)), _gen_sha(config)

    if mode == "replay" and _S["diverged"] is None:
        rec = _S["served"][k] if k < len(_S["served"]) else None
        if rec is None and k == len(_S["served"]) and _S["served"]:
            # One call past the end of the recording. In the recording this call was started and then
            # cut by the harness's turn cap after its first chunk (never recorded). Probe with an
            # empty chunk: if the consumer closes us now, this is the same cut, not a divergence.
            try:
                yield TextChunk("")
            except GeneratorExit:
                _append(_S["replay_log"], {"event": "cut", "task": _S["task"], "attempt": _S["attempt"], "k": k})
                raise
        rec_norm = None
        if rec is not None:
            rec_norm = _sha(normalize(_S["reqs"][k])) if k in _S["reqs"] else rec["norm_sha"]
        match_gen = bool(rec) and rec.get("gen_sha", gen) == gen
        match_raw = bool(rec) and rec["raw_sha"] == raw and match_gen
        match_norm = bool(rec) and rec_norm == norm and match_gen
        _append(_S["replay_log"], {"event": "call", "task": _S["task"], "attempt": _S["attempt"], "k": k,
                                   "match_raw": match_raw, "match_norm": match_norm, "match_gen": match_gen,
                                   "recorded": rec is not None, "req_chars": len(req)})
        _append_gz(_S["replay_log"] + ".req.jsonl.gz", {"task": _S["task"], "attempt": _S["attempt"], "k": k, "req": req})
        if match_norm:
            if _S.get("record_in_replay") and _S.get("record"):
                _append(_S["record"] + ".jsonl", {**rec, "k": k, "raw_sha": raw, "norm_sha": norm, "gen_sha": gen,
                                                  "req_chars": len(req), "t0": 0.0, "t1": 0.0, "served": True})
                _append_gz(_S["record"] + ".req.jsonl.gz",
                           {"task": _S["task"], "attempt": _S["attempt"], "k": k, "req": req})
            r = rec["resp"]
            if r.get("text"):
                yield TextChunk(r["text"])
            turn = AssistantTurn(r.get("text", ""), r.get("tool_calls", []), r.get("in", 0), r.get("out", 0))
            turn.cached_tokens = r.get("cached", 0)
            yield turn
            return
        _S["diverged"] = k
        if _S["on_diverge"] == "stop":
            raise StopReplay(f"diverged at call {k}")
        # live: fall through to the real model for this and all later calls

    t0 = time.time()
    turn = None
    for ev in stream_fn(model=model, system=system, messages=messages, tool_schemas=tool_schemas, config=config):
        if isinstance(ev, AssistantTurn):
            turn = ev
        yield ev
    t1 = time.time()
    if turn is not None:
        _count(turn)
    if mode == "replay" and turn is not None:
        x = {"event": "live", "task": _S["task"], "attempt": _S["attempt"], "k": k, "t0": t0, "t1": t1,
             "in": turn.in_tokens, "out": turn.out_tokens, "cached": getattr(turn, "cached_tokens", 0),
             "think_exceeded": bool(getattr(turn, "think_exceeded", False)), "req_chars": len(req)}
        _S["live"].append(x)
        if _S.get("replay_log"):
            _append(_S["replay_log"], x)
            _append_gz(_S["replay_log"] + ".req.jsonl.gz",
                       {"task": _S["task"], "attempt": _S["attempt"], "k": k, "req": req, "live": True})
    recording = mode == "record" or (mode == "replay" and _S.get("record_in_replay"))
    if recording and turn is not None and _S.get("record"):
        entry = {"task": _S["task"], "attempt": _S["attempt"], "k": k, "raw_sha": raw, "norm_sha": norm,
                 "gen_sha": gen,
                 "req_chars": len(req), "t0": t0, "t1": t1,
                 "resp": {"text": turn.text, "tool_calls": turn.tool_calls, "in": turn.in_tokens,
                          "out": turn.out_tokens, "cached": getattr(turn, "cached_tokens", 0),
                          "think_exceeded": bool(getattr(turn, "think_exceeded", False))}}
        _append(_S["record"] + ".jsonl", entry)
        _append_gz(_S["record"] + ".req.jsonl.gz", {"task": _S["task"], "attempt": _S["attempt"], "k": k, "req": req})


SUMMARY_PREFIX = "[Previous conversation summary]\n"


def side_stream(stream_fn, purpose, **kw):
    """Model calls outside the agent's main loop (compaction summaries).

    record: pass through; append timing and tokens to <record>.side.jsonl.
    replay, before the divergence point: serve the summary text from the next recorded request
            (compaction puts it there as the first message), so old records replay too. If it is
            not there, call the model; the next request then diverges.
    live calls are logged like other live calls (event "live", purpose set).
    """
    mode = _S.get("mode", "off")
    if mode == "replay" and _S.get("diverged") is None:
        req = _S.get("reqs", {}).get(_S.get("k"))
        if req:
            for m in json.loads(req)["messages"]:
                c = m.get("content")
                if isinstance(c, str) and c.startswith(SUMMARY_PREFIX):
                    _append(_S["replay_log"], {"event": "side_served", "purpose": purpose, "task": _S["task"],
                                               "attempt": _S["attempt"], "k_next": _S["k"]})
                    yield TextChunk(c[len(SUMMARY_PREFIX):])
                    return
        if _S.get("k", 0) >= len(_S.get("served", [])) and _S.get("on_diverge") == "stop":
            # Past the last recorded call: in the recording this compaction was followed by a call
            # that the turn cap cut. Stop-mode replay has no model; give an empty summary. The
            # next call is then probed: a cut ends the attempt, anything else is a divergence.
            _append(_S["replay_log"], {"event": "side_tail", "purpose": purpose, "task": _S["task"],
                                       "attempt": _S["attempt"], "k_next": _S["k"]})
            yield TextChunk("")
            return
    t0 = time.time()
    turn = None
    for ev in stream_fn(**kw):
        if isinstance(ev, AssistantTurn):
            turn = ev
        yield ev
    t1 = time.time()
    if mode == "off" or turn is None:
        return
    _count(turn)
    x = {"event": "live", "purpose": purpose, "task": _S["task"], "attempt": _S["attempt"], "k": _S.get("k"),
         "t0": t0, "t1": t1, "in": turn.in_tokens, "out": turn.out_tokens,
         "cached": getattr(turn, "cached_tokens", 0)}
    if mode == "record":
        _append(_S["record"] + ".side.jsonl", x)
    elif mode == "replay":
        _S["live"].append(x)
        if _S.get("replay_log"):
            _append(_S["replay_log"], x)
