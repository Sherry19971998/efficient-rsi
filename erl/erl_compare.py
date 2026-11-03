"""Compare hgm-erl runs with hgm-fixed at the common GPU-hour cut (search-set results only)."""
import json, sys
from collections import Counter, defaultdict
sys.path.insert(0, "../instrumentation")
import analyze_ledger as a
from results import select_best

ledger, runs, hours = sys.argv[1], sys.argv[2].split(","), float(sys.argv[3])
by = defaultdict(list)
with open(ledger) as f:
    for line in f:
        if '"hgm-' not in line[:200] and '"run": "hgm-' not in line:
            continue
        e = json.loads(line)
        if e.get("run") in runs:
            by[e["run"]].append(e)
print(f"cut at {hours} GPU hours\n")
print("| run | evals | children | evals on initial | evals on top-3 nodes | retired | selected (solved/n) | best other n>=20 |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for run in runs:
    ev, _ = a.truncate(by[run], max_hours=hours)
    res = defaultdict(list)
    for e in ev:
        if e["type"] == "eval_result":
            res[e["node"]] += e["score"]
    n_eval = sum(len(v) for n, v in res.items() if n != "initial" or True) - 0
    kids = sum(e["type"] == "node_created" for e in ev)
    retired = sum(e["type"] == "erl_retire" for e in ev)
    init_cached = next((len(e["score"]) for e in ev if e["type"] == "eval_result" and e.get("source") == "initial_cache"), 0)
    searched = {n: len(v) - (init_cached if n == "initial" else 0) for n, v in res.items()}
    total = sum(searched.values())
    top3 = sum(sorted((c for n, c in searched.items() if n != "initial"), reverse=True)[:3])
    best = select_best({n: v for n, v in res.items() if v}, eps=0.05)
    b = res[best]
    others = sorted(((sum(v) / len(v), len(v), n) for n, v in res.items() if len(v) >= 20 and n != best), reverse=True)[:1]
    o = f"{others[0][0]:.2f} ({others[0][1]})" if others else "-"
    print(f"| {run} | {total} | {kids} | {searched.get('initial', 0)} | {100 * top3 / max(total, 1):.0f}% | {retired} | "
          f"{best[-12:]} {sum(b)}/{len(b)} = {sum(b) / len(b):.2f} | {o} |")
