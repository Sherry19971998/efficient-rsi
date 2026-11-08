"""Sample GPU utilisation into the cost ledger while a run is going.

Every --interval seconds it appends one `gpu_sample` event per GPU with utilisation,
memory and the processes using it. analyze_ledger.py turns these into GPU-busy hours
(sum of utilisation x interval) and flags samples where a process other than the
expected vLLM server was on the GPU, since that contaminates the timing.

Usage:
    RSI_LEDGER=runs/ledger.jsonl python gpu_sampler.py --interval 5 --expect-pid <vLLM pid>
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rsi_ledger  # noqa: E402


def query(fields, kind="gpu"):
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    out = subprocess.run(["nvidia-smi", f"{flag}={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def with_descendants(pids, proc="/proc"):
    """The given pids and every process below them. vLLM's engine runs in a child process
    (EngineCore), and that child, not the `vllm serve` process, is what holds the GPU."""
    children = {}
    try:
        entries = os.listdir(proc)
    except OSError:   # no /proc (not Linux): only the pids themselves
        return set(pids)
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(proc, name, "stat")) as f:
                stat = f.read()
        except OSError:   # the process ended meanwhile
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])   # field 4; the name in () may hold spaces
        children.setdefault(ppid, []).append(int(name))
    out, todo = set(), list(pids)
    while todo:
        p = todo.pop()
        if p not in out:
            out.add(p)
            todo.extend(children.get(p, []))
    return out


def sample(interval, expect_pids):
    gpus = query("index,uuid,utilization.gpu,memory.used,memory.total")
    apps = query("gpu_uuid,pid,used_memory", kind="apps")
    expected = with_descendants(expect_pids) if expect_pids else set()
    for index, uuid, util, mem_used, mem_total in gpus:
        procs = [{"pid": int(pid), "mem_mib": int(mem)} for g, pid, mem in apps if g == uuid]
        foreign = [p["pid"] for p in procs if expect_pids and p["pid"] not in expected]
        rsi_ledger.event("gpu_sample", gpu=int(index), util_pct=float(util), mem_used_mib=int(mem_used),
                         mem_total_mib=int(mem_total), interval_s=interval, procs=procs,
                         foreign_pids=foreign or None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--expect-pid", type=int, action="append", default=[],
                    help="PIDs allowed on the GPU, with all their child processes (the vLLM server); others are flagged")
    args = ap.parse_args()
    if not rsi_ledger.enabled():
        sys.exit("Set RSI_LEDGER before starting the sampler.")
    expect = set(args.expect_pid)
    while True:
        start = time.time()
        try:
            sample(args.interval, expect)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            rsi_ledger.event("gpu_sample_error", error=str(e))
        time.sleep(max(0.0, args.interval - (time.time() - start)))


if __name__ == "__main__":
    main()
