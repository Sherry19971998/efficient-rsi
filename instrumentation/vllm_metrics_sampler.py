"""Sample vLLM's own load metrics, the GPU-side record the cost ledger cannot reconstruct.

nvidia-smi shows how busy the GPU is; when it is unavailable (or in addition), vLLM's
Prometheus endpoint says what the GPU was doing: how many requests were running or
queued, how full the KV cache was, how often requests were preempted (their prefill is
then recomputed, pure waste), and cumulative prompt / generated / prefix-cache-hit tokens,
whose differences give throughput over time. Every --interval seconds it appends one
line to --out (a file of its own, so the search's ledger is never touched); every
--host-every samples it also records the host load average and the number of running
docker containers.

    python vllm_metrics_sampler.py --out runs/vllm_metrics.jsonl --interval 10
    python vllm_metrics_sampler.py --summary --out runs/vllm_metrics.jsonl --ledger runs/ledger.jsonl --run hgm-orig-r0

Only the standard library is used.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

# metric name in vLLM -> field in the sample; counters and gauges are summed over label sets
GAUGES = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
          "vllm:kv_cache_usage_perc": "kv_cache_usage"}
COUNTERS = {"vllm:num_preemptions_total": "preemptions", "vllm:prompt_tokens_total": "prompt_tokens",
            "vllm:generation_tokens_total": "generation_tokens",
            "vllm:prefix_cache_hits_total": "prefix_cache_hits", "vllm:prefix_cache_queries_total": "prefix_cache_queries"}


def parse(text):
    """Prometheus text format -> {field: value} for the metrics above, plus queue reasons."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_labels, _, value = line.rpartition(" ")
        name, _, labels = name_labels.partition("{")
        try:
            v = float(value)
        except ValueError:
            continue
        field = GAUGES.get(name) or COUNTERS.get(name)
        if field:
            out[field] = out.get(field, 0.0) + v
        elif name == "vllm:num_requests_waiting_by_reason":
            reason = labels.split('reason="', 1)[-1].split('"', 1)[0] if 'reason="' in labels else "?"
            out[f"waiting_{reason}"] = out.get(f"waiting_{reason}", 0.0) + v
    return out


def host_stats():
    stats = {"load1": round(os.getloadavg()[0], 2)}
    try:
        ps = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, timeout=20, check=True)
        stats["containers"] = len(ps.stdout.split())
    except (subprocess.SubprocessError, FileNotFoundError):
        stats["containers"] = None
    return stats


def run_sampler(url, out, interval, host_every):
    n = 0
    with open(out, "a") as f:
        while True:
            start = time.time()
            rec = {"ts": round(start, 3)}
            try:
                with urllib.request.urlopen(url, timeout=max(1.0, interval / 2)) as r:
                    rec.update(parse(r.read().decode()))
            except OSError as e:
                rec["error"] = str(e)[:200]
            if n % host_every == 0:
                rec.update(host_stats())
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            time.sleep(max(0.0, interval - (time.time() - start)))


def window(ledger, run):
    """(first, last) ts of a run in the cost ledger."""
    ts = []
    with open(ledger) as f:
        for line in f:
            if f'"{run}"' in line:
                e = json.loads(line)
                if e.get("run") == run and "ts" in e:
                    ts.append(e["ts"])
    if not ts:
        sys.exit(f"{run}: no events in {ledger}")
    return min(ts), max(ts)


def summarize(samples):
    """What the GPU did over a list of samples (in time order)."""
    ok = [s for s in samples if "error" not in s and "running" in s]
    if len(ok) < 2:
        return {"samples": len(samples), "note": "fewer than two usable samples"}
    hours = (ok[-1]["ts"] - ok[0]["ts"]) / 3600

    def delta(k):
        return ok[-1].get(k, 0.0) - ok[0].get(k, 0.0)

    queries = delta("prefix_cache_queries")
    return {
        "samples": len(samples), "errors": len(samples) - len(ok), "hours": round(hours, 2),
        # no request in flight = the GPU had nothing to do (agents in setup/tests, or between steps)
        "idle_share": round(sum(s["running"] == 0 for s in ok) / len(ok), 3),
        "mean_running": round(sum(s["running"] for s in ok) / len(ok), 1),
        "queued_share": round(sum(s.get("waiting", 0) > 0 for s in ok) / len(ok), 3),
        "mean_kv_cache_usage": round(sum(s.get("kv_cache_usage", 0) for s in ok) / len(ok), 3),
        "kv_cache_full_share": round(sum(s.get("kv_cache_usage", 0) >= 0.95 for s in ok) / len(ok), 3),
        "preemptions_per_hour": round(delta("preemptions") / hours, 1) if hours else None,
        "prompt_tokens_per_s": round(delta("prompt_tokens") / (hours * 3600)) if hours else None,
        "generation_tokens_per_s": round(delta("generation_tokens") / (hours * 3600), 1) if hours else None,
        "prefix_cache_hit_rate": round(delta("prefix_cache_hits") / queries, 3) if queries else None,
        "max_containers": max((s["containers"] for s in samples if s.get("containers") is not None), default=None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    ap.add_argument("--out", default="runs/vllm_metrics.jsonl")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--host-every", type=int, default=6, help="record load and container count every N samples")
    ap.add_argument("--summary", action="store_true", help="summarise --out instead of sampling")
    ap.add_argument("--ledger", help="with --summary and --run: restrict to that run's time window")
    ap.add_argument("--run")
    args = ap.parse_args()
    if not args.summary:
        run_sampler(args.url, args.out, args.interval, args.host_every)
        return
    lo, hi = window(args.ledger, args.run) if args.ledger and args.run else (float("-inf"), float("inf"))
    with open(args.out) as f:
        samples = [s for s in map(json.loads, f) if lo <= s["ts"] <= hi]
    print(json.dumps(summarize(samples), indent=2))


if __name__ == "__main__":
    main()
