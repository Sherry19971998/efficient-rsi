"""Summarise a cost ledger written by rsi_ledger.py and llm_proxy.py.

Prints a Markdown report with:
  1. API spend by stage (propose / diagnose / eval) and the models seen
  2. wall time and container CPU by stage and phase (setup / agent_run / test)
  3. waste metrics: evaluation spent on nodes that were never chosen as parents,
     on broken nodes (accuracy 0), and on lineages that never beat their parent
  4. optionally, a per-node CSV (--nodes-csv)

Usage:
    python analyze_ledger.py runs/ledger.jsonl [--run hgm-fixed-seed0] [--nodes-csv nodes.csv]
"""

import argparse
import csv
import json
import re
from collections import defaultdict


def load(path, run=None):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if run is None or e.get("run") == run:
                events.append(e)
    return events


def truncate_at_evals(events, max_evals):
    """Keep only what happened up to the max_evals-th task evaluation of the search.

    One task evaluation = one `eval` event with phase `agent_run` (every harness writes
    exactly one per task). The root's cached evaluation is not part of the search budget.
    Calls and phases still in flight at the cut-off are dropped, which slightly
    undercounts the cost of the last few evaluations for both methods alike.
    """
    runs = sorted((e for e in events if e["type"] == "eval" and e.get("phase") == "agent_run"
                   and e.get("node") != "initial"), key=lambda e: e["ts"])
    if len(runs) <= max_evals:
        return events, None
    cutoff = runs[max_evals - 1]["ts"]
    return [e for e in events if e["ts"] <= cutoff], cutoff


# A stretch this long without any ledger event is a stopped search (a crash before
# --continue_from), not search time: the run did not hold the GPU then.
IDLE_GAP_S = 1800


def _active_times(events):
    """(ts, search seconds elapsed by ts) for each event, leaving idle gaps out."""
    ts = sorted(e["ts"] for e in events if "ts" in e and e.get("node") != "initial")
    out, elapsed = [], 0.0
    for prev, t in zip([None] + ts, ts):
        if prev is not None and t - prev <= IDLE_GAP_S:
            elapsed += t - prev
        out.append((t, elapsed))
    return out


def gpu_hours(events):
    """GPU occupancy of a search in hours: wall time from its first to its last event, minus idle gaps.

    Runs go one at a time on one GPU (PARALLEL=0), so a run holds the GPU for its whole wall
    time, including setup and tests that only use the CPU. That is what a rented GPU costs;
    how busy the GPU was meanwhile is the GPU sampler's separate busy-hours figure.
    """
    times = _active_times(events)
    return times[-1][1] / 3600 if times else 0.0


def truncate_at_hours(events, max_hours):
    """Keep only what happened in the first max_hours of GPU occupancy (equal-GPU-budget cut)."""
    for t, elapsed in _active_times(events):
        if elapsed >= max_hours * 3600:
            return [e for e in events if e.get("ts", 0) <= t], t
    return events, None


def truncate(events, max_evals=None, max_hours=None):
    """Cut a run at a GPU-hour budget if one is given, else at a number of task evaluations."""
    if max_hours:
        return truncate_at_hours(events, max_hours)
    if max_evals:
        return truncate_at_evals(events, max_evals)
    return events, None


# Both HGM timeouts (self-improvement and evaluation) are 3600 s; a phase this long hit the limit.
TIMEOUT_S = 3590


def broken_cause(n):
    """Most likely reason a node scored 0, from the signals the ledger holds (first match wins)."""
    if n.auth_failures:
        return "called a model it has no credentials for"
    if n.propose_timed_out:
        return "self-modification hit the time limit"
    if n.n_evals and n.eval_timeouts == n.n_evals:
        return "every evaluation hit the time limit"
    if n.ctx_overflows:
        return "exceeded the model's context length"
    return "other (the modified agent fails the tasks)"


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def pct(part, whole):
    return f"{100 * part / whole:.1f}%" if whole else "—"


def fmt(x, digits=2):
    return "—" if x is None else f"{x:,.{digits}f}"


class Node:
    def __init__(self, id):
        self.id = id
        self.parent = None
        self.children = []
        self.scores = []          # per-task 0/1 results (HGM)
        self.accuracy = None      # summary accuracy (DGM)
        self.times_parent = 0
        self.eval_usd = 0.0
        self.eval_wall = 0.0
        self.eval_cpu = 0.0
        self.eval_tasks = set()
        self.propose_usd = 0.0
        self.propose_wall = 0.0
        # failure signals, used to say why a node scored 0
        self.auth_failures = 0    # calls refused for missing credentials: the agent reached for a model it has no key for
        self.ctx_overflows = 0    # calls refused for exceeding the model's context length
        self.eval_timeouts = 0
        self.eval_timeout_wall = 0.0
        self.n_evals = 0
        self.propose_timed_out = False

    @property
    def acc(self):
        if self.scores:
            return mean(self.scores)
        return self.accuracy


def build(events):
    nodes = {}

    def node(i):
        if i not in nodes:
            nodes[i] = Node(i)
        return nodes[i]

    usd = defaultdict(float)
    tokens = defaultdict(lambda: defaultdict(int))
    models = defaultdict(lambda: {"calls": 0, "priced": 0})
    wall = defaultdict(float)
    cpu = defaultdict(float)
    failed_calls = 0
    by_provider = defaultdict(lambda: {"calls": 0, "usd": 0.0, "tokens": 0})
    gpu = {"samples": 0, "sampled_s": 0.0, "busy_s": 0.0, "foreign_samples": 0}

    # HGM's harness names an evaluation "<commit>_<i>" (hgm_utils.eval_agent run_ids), while the
    # search events name the node "<commit>"; map the former back onto the node.
    known = {e.get("node") for e in events
             if e["type"] in ("node_created", "eval_result", "select_parent", "node_evaluated")}

    def canonical(i):
        if i and i not in known:
            m = re.fullmatch(r"(.+)_\d+", i)
            if m and m.group(1) in known:
                return m.group(1)
        return i

    for e in events:
        t = e["type"]
        n = canonical(e.get("node"))
        if t == "gpu_sample":
            dt = e.get("interval_s") or 0
            gpu["samples"] += 1
            gpu["sampled_s"] += dt
            gpu["busy_s"] += dt * (e.get("util_pct") or 0) / 100
            gpu["foreign_samples"] += bool(e.get("foreign_pids"))
            continue
        if t == "llm_call":
            tag = e.get("tag") or "untagged"
            m = e.get("model") or "?"
            bp = by_provider[e.get("provider") or "openai"]
            bp["calls"] += 1
            bp["usd"] += e.get("usd") or 0
            bp["tokens"] += (e.get("input_tokens") or 0) + (e.get("output_tokens") or 0)
            models[m]["calls"] += 1
            if e.get("status") != 200:
                failed_calls += 1
                if n and e.get("status") in (401, 403):
                    node(n).auth_failures += 1
                elif n and e.get("status") == 400 and "context length" in (e.get("error_body") or ""):
                    node(n).ctx_overflows += 1
            if e.get("usd") is not None:
                models[m]["priced"] += 1
                usd[tag] += e["usd"]
                if n and tag == "eval":
                    node(n).eval_usd += e["usd"]
                elif n and tag in ("propose", "diagnose"):
                    node(n).propose_usd += e["usd"]
            for k in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
                tokens[tag][k] += e.get(k) or 0
        elif t in ("eval", "propose"):
            key = (t, e.get("phase"))
            wall[key] += e.get("wall_s") or 0
            cpu[key] += e.get("container_cpu_s") or 0
            timed_out = e.get("phase") == "agent_run" and (e.get("wall_s") or 0) >= TIMEOUT_S
            if n and t == "eval":
                node(n).eval_wall += e.get("wall_s") or 0
                node(n).eval_cpu += e.get("container_cpu_s") or 0
                if e.get("task"):
                    node(n).eval_tasks.add(e["task"])
                if e.get("phase") == "agent_run":
                    node(n).n_evals += 1
                    node(n).eval_timeouts += timed_out
                    node(n).eval_timeout_wall += (e.get("wall_s") or 0) if timed_out else 0
            elif n and t == "propose":
                node(n).propose_wall += e.get("wall_s") or 0
                node(n).propose_timed_out |= timed_out
        elif t == "eval_report":
            wall[("eval", "report")] += e.get("wall_s") or 0
            if n:
                node(n).eval_wall += e.get("wall_s") or 0
        elif t == "select_parent" and n:
            node(n).times_parent += 1
        elif t == "node_created" and n:
            node(n).parent = e.get("parent")
        elif t == "eval_result" and n:
            node(n).scores.extend(e.get("score") or [])
        elif t == "node_evaluated" and n:
            node(n).parent = node(n).parent or e.get("parent")
            node(n).accuracy = e.get("accuracy")

    for nd in list(nodes.values()):
        if nd.parent:
            node(nd.parent).children.append(nd.id)
    return nodes, usd, tokens, models, wall, cpu, failed_calls, by_provider, gpu


def subtree_best(nodes, i, memo):
    if i in memo:
        return memo[i]
    nd = nodes[i]
    best = nd.acc if nd.acc is not None else -1
    for c in nd.children:
        best = max(best, subtree_best(nodes, c, memo))
    memo[i] = best
    return best


def report(events):
    nodes, usd, tokens, models, wall, cpu, failed_calls, by_provider, gpu = build(events)
    out = []
    total_usd = sum(usd.values())

    out.append("## 1. API spend by stage\n")
    out.append("| stage | USD | share | input tok | cached tok | uncached input tok (share) | output tok (share) |")
    out.append("| --- | --- | --- | --- | --- | --- | --- |")
    uncached = {tag: tk["input_tokens"] - tk["cached_tokens"] for tag, tk in tokens.items()}
    tot_unc, tot_out = sum(uncached.values()), sum(tk["output_tokens"] for tk in tokens.values())
    for tag in sorted(set(usd) | set(tokens)):
        tk = tokens[tag]
        out.append(f"| {tag} | {fmt(usd[tag])} | {pct(usd[tag], total_usd)} | {tk['input_tokens']:,} | "
                   f"{tk['cached_tokens']:,} | {uncached.get(tag, 0):,} ({pct(uncached.get(tag, 0), tot_unc)}) | "
                   f"{tk['output_tokens']:,} ({pct(tk['output_tokens'], tot_out)}) |")
    out.append(f"| **total** | **{fmt(total_usd)}** | | | | | |\n")
    out.append("On a local GPU, prefix-cache hits cost little: GPU time follows uncached input (prefill) and "
               "output (decode) tokens, not total input. The two share columns bracket each stage's GPU share.")
    out.append("Models seen: " + (", ".join(
        f"{m} ({v['calls']} calls, {v['priced']} priced)" for m, v in sorted(models.items())) or "none"))
    if any(v["priced"] < v["calls"] for v in models.values()):
        out.append("\nSome calls have no USD: the model is missing from prices.json, the call failed, or it streamed.")
    out.append(f"\nFailed or non-200 calls: {failed_calls}\n")
    out.append("| provider | calls | tokens | USD |")
    out.append("| --- | --- | --- | --- |")
    for prov, v in sorted(by_provider.items()):
        out.append(f"| {prov} | {v['calls']:,} | {v['tokens']:,} | {fmt(v['usd'])} |")
    out.append("\nLocal vLLM calls carry no USD; their cost is the GPU time in section 4.\n")

    out.append("## 2. Time by stage and phase\n")
    out.append("| stage | phase | wall h | share of stage | container CPU h |")
    out.append("| --- | --- | --- | --- | --- |")
    for stage in ("propose", "eval"):
        stage_total = sum(v for (s, _), v in wall.items() if s == stage)
        for (s, phase), v in sorted(wall.items()):
            if s == stage:
                out.append(f"| {stage} | {phase} | {fmt(v / 3600)} | {pct(v, stage_total)} | {fmt(cpu[(s, phase)] / 3600)} |")
    eval_total = sum(v for (s, _), v in wall.items() if s == "eval")
    out.append(f"\nFixed overhead in evaluation (setup phase): {pct(wall[('eval', 'setup')], eval_total)} of evaluation wall time.")
    n_evals = sum(n.n_evals for n in nodes.values())
    n_to = sum(n.eval_timeouts for n in nodes.values())
    out.append(f"Evaluations that hit the {TIMEOUT_S + 10} s limit: {n_to} of {n_evals} ({pct(n_to, n_evals)}), "
               f"holding {pct(sum(n.eval_timeout_wall for n in nodes.values()), eval_total)} of evaluation wall time.\n")

    out.append("## 3. Waste\n")
    evaluated = [n for n in nodes.values() if n.eval_wall > 0 or n.eval_usd > 0]
    tot_w = sum(n.eval_wall for n in evaluated)
    tot_u = sum(n.eval_usd for n in evaluated)

    def share(pred):
        sel = [n for n in evaluated if pred(n)]
        return (len(sel), pct(sum(n.eval_wall for n in sel), tot_w), pct(sum(n.eval_usd for n in sel), tot_u))

    memo = {}
    rows = [
        ("never chosen as parent", lambda n: n.times_parent == 0),
        ("broken (accuracy 0)", lambda n: n.acc == 0),
        ("lineage never beats its parent", lambda n: n.parent in nodes and nodes[n.parent].acc is not None
         and subtree_best(nodes, n.id, memo) <= nodes[n.parent].acc),
    ]
    out.append("| evaluation spent on nodes that are … | nodes | share of eval wall time | share of eval USD |")
    out.append("| --- | --- | --- | --- |")
    for label, pred in rows:
        k, w, u = share(pred)
        out.append(f"| {label} | {k} / {len(evaluated)} | {w} | {u} |")
    broken = [n for n in evaluated if n.acc == 0]
    if broken:
        out.append("\nWhy the broken nodes score 0 (first matching signal in the ledger):\n")
        out.append("| cause | nodes | share of eval wall time |")
        out.append("| --- | --- | --- |")
        causes = defaultdict(list)
        for n in broken:
            causes[broken_cause(n)].append(n)
        for cause, ns in sorted(causes.items(), key=lambda kv: -len(kv[1])):
            out.append(f"| {cause} | {len(ns)} | {pct(sum(n.eval_wall for n in ns), tot_w)} |")
    if gpu["samples"]:
        out.append("\n## 4. GPU\n")
        out.append(f"Sampled {fmt(gpu['sampled_s'] / 3600)} h; GPU busy {fmt(gpu['busy_s'] / 3600)} h "
                   f"(utilisation x time). Samples with a process other than the expected vLLM server on the GPU: "
                   f"{gpu['foreign_samples']} of {gpu['samples']} ({pct(gpu['foreign_samples'], gpu['samples'])}); "
                   f"timing from those samples is contaminated.")
    created = [n for n in nodes.values() if n.parent]
    out.append(f"\nNodes created: {len(created)}; broken among them: "
               f"{sum(1 for n in created if n.acc == 0)}; never evaluated: {sum(1 for n in created if n not in evaluated)}.")
    return "\n".join(out), nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ledger")
    parser.add_argument("--run", help="only events whose RSI_RUN label matches")
    parser.add_argument("--nodes-csv", help="write per-node statistics to this CSV")
    parser.add_argument("--max-evals", type=int, help="cut the run at this many task evaluations (equal-budget comparison)")
    parser.add_argument("--max-hours", type=float, help="cut the run at this many GPU-occupancy hours (overrides --max-evals)")
    args = parser.parse_args()

    events = load(args.ledger, args.run)
    if args.max_hours or args.max_evals:
        events, cutoff = truncate(events, args.max_evals, args.max_hours)
        what = f"{args.max_hours} GPU hours" if args.max_hours else f"task evaluation {args.max_evals}"
        print(f"Cut at {what}" + (f" (ts {cutoff:.0f})" if cutoff else " (run had less)") + "\n")
    n_evals = sum(1 for e in events if e["type"] == "eval" and e.get("phase") == "agent_run"
                  and e.get("node") != "initial")
    print(f"GPU occupancy: {gpu_hours(events):.2f} h for {n_evals} task evaluations\n")
    text, nodes = report(events)
    print(text)
    if args.nodes_csv:
        with open(args.nodes_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node", "parent", "children", "times_parent", "accuracy", "n_eval_tasks",
                        "eval_usd", "eval_wall_s", "eval_container_cpu_s", "propose_usd", "propose_wall_s"])
            for n in nodes.values():
                w.writerow([n.id, n.parent or "", len(n.children), n.times_parent,
                            "" if n.acc is None else round(n.acc, 4), len(n.eval_tasks),
                            round(n.eval_usd, 4), round(n.eval_wall, 1), round(n.eval_cpu, 1),
                            round(n.propose_usd, 4), round(n.propose_wall, 1)])


if __name__ == "__main__":
    main()
