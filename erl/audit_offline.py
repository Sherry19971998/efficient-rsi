"""How often would the online audit find a false stop? Measured on the recorded runs.

The recorded HGM runs stopped no child, so every child the stopper would retire still has
its later results. Here OnlineStopper runs over each recorded stream with every stop
audited (audit_p = 1) and the rollback off (audit_max_false = 1), and reports, per run,
how many audits finished, how many were false stops (the child's own results alone kept its
upper bound above the incumbent after audit_n more tasks), and the rate. w = 0 (plain
statistics) is the reference: its stops use no prediction.

    python audit_offline.py --ledger ../runs/ledger.jsonl --prescreen-rows ../runs/prescreen_rows.jsonl \
        --runs hgm-fixed-r0 ... --history ../runs/erl_history.json
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "instrumentation"))

import analyze_ledger  # noqa: E402
import online_stop  # noqa: E402
import replay as rp  # noqa: E402


def run_stream(events, stream, diffs, w, history, audit_n, rule):
    log = []
    st = online_stop.OnlineStopper(w=w, history_path=history, audit_p=1.0, audit_n=audit_n, audit_max_false=1.0,
                                   audit_rule=rule,
                                   log=lambda kind, **kw: log.append((kind, kw)))
    created = sorted((e["ts"], e["node"], e.get("parent")) for e in events if e["type"] == "node_created" and e.get("node"))
    ci = 0
    for node, res, cost, _, ts, *_ in stream:
        while ci < len(created) and created[ci][0] <= ts:
            n, p = created[ci][1], created[ci][2]
            st._state(n, p, diffs.get(n))
            ci += 1
        if node == "initial" and node not in st.states:
            st.seed("initial", None, res)
            continue
        if st.allow(node):
            st.result(node, res, cost)
    ends = [kw for k, kw in log if k == "erl_audit_end"]
    started = sum(k == "erl_audit_start" for k, _ in log)
    errors = sum(k == "erl_error" for k, _ in log)
    return started, ends, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--prescreen-rows", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--history", help="history rows (erl_history.json); note it includes these runs themselves")
    ap.add_argument("--max-hours", type=float, default=9.646)
    ap.add_argument("--audit-n", type=int, default=5)
    ap.add_argument("--rule", default="mean", choices=["mean", "ucb"])
    args = ap.parse_args()

    diff_rows = {}
    with open(args.prescreen_rows) as f:
        for line in f:
            r = json.loads(line)
            diff_rows[(r["run"], r["node"])] = r
    by_run = rp.load(args.ledger, args.runs)
    print(f"Cut at {args.max_hours} GPU hours; every stop audited with {args.audit_n} more tasks, "
          f"rule {args.rule}; "
          f"history: {args.history or 'none'}.\n")
    print("| run | setting | stops audited (finished) | false stops | false-stop rate |")
    print("| --- | --- | --- | --- | --- |")
    tot = {}
    for run in args.runs:
        events, _ = analyze_ledger.truncate(by_run[run], max_hours=args.max_hours)
        stream = rp.result_stream(events)
        diffs = {n: r for (rr, n), r in diff_rows.items() if rr == run}
        for name, w, hist in (("w0", 0, None), ("auto", "auto", None), ("auto + history", "auto", args.history)):
            if hist is None and name == "auto + history":
                continue
            started, ends, errors = run_stream(events, stream, diffs, w, hist, args.audit_n, args.rule)
            false = sum(e["false_stop"] for e in ends)
            a = tot.setdefault(name, [0, 0])
            a[0] += len(ends); a[1] += false
            rate = f"{100 * false / len(ends):.0f}%" if ends else "—"
            print(f"| {run} | {name} | {started} ({len(ends)}) | {false} | {rate} |" + (f" errors {errors}" if errors else ""),
                  flush=True)
    print()
    for name, (n, f) in tot.items():
        print(f"{name}: {f}/{n} false stops ({100 * f / max(n, 1):.0f}%)")


if __name__ == "__main__":
    main()
