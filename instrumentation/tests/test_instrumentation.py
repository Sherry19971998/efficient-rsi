"""Tests for the cost ledger, the LLM proxy, the analyzer and the HGM tree fix.

Run from the instrumentation directory:
    python -m unittest discover -s tests -v
"""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HGM_DIR = os.path.join(os.path.dirname(ROOT), "repos", "HGM")
sys.path.insert(0, ROOT)

import analyze_ledger  # noqa: E402
import llm_proxy  # noqa: E402
import rsi_ledger  # noqa: E402


def read_ledger(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class LedgerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = os.path.join(self.tmp.name, "ledger.jsonl")
        self.env = {k: os.environ.get(k) for k in ("RSI_LEDGER", "RSI_RUN", "RSI_PROXY_URL", "RSI_PROXY_URL_CONTAINER")}
        os.environ["RSI_LEDGER"] = self.ledger
        os.environ["RSI_RUN"] = "test-run"
        os.environ.pop("RSI_PROXY_URL", None)
        os.environ.pop("RSI_PROXY_URL_CONTAINER", None)

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()


class TestLedger(LedgerCase):
    def test_disabled_is_noop(self):
        os.environ.pop("RSI_LEDGER")
        rsi_ledger.event("x", a=1)
        with rsi_ledger.span("y"):
            pass
        self.assertEqual(rsi_ledger.proxy_env("eval", "n", "t"), {})
        self.assertFalse(os.path.exists(self.ledger))

    def test_event_and_span(self):
        rsi_ledger.event("select_parent", node="n1", skipped=None)
        with self.assertRaises(ValueError):
            with rsi_ledger.span("eval_report", node="n1"):
                raise ValueError("boom")
        e1, e2 = read_ledger(self.ledger)
        self.assertEqual(e1["type"], "select_parent")
        self.assertEqual(e1["run"], "test-run")
        self.assertNotIn("skipped", e1)
        self.assertEqual(e2["status"], "error: ValueError")
        self.assertIn("wall_s", e2)

    def test_phases_with_container_cpu(self):
        class FakeContainer:
            cpu = 10.0

            def stats(self, stream=False, one_shot=True):
                FakeContainer.cpu += 2.5
                return {"cpu_stats": {"cpu_usage": {"total_usage": FakeContainer.cpu * 1e9}}}

        ph = rsi_ledger.Phases("eval", node="n1", task="t1")
        ph.phase("setup")
        ph.attach(FakeContainer())
        ph.phase("agent_run")
        ph.done()
        ph.done()  # second call is a no-op
        events = read_ledger(self.ledger)
        self.assertEqual([e["phase"] for e in events], ["setup", "agent_run"])
        self.assertEqual(events[1]["container_cpu_s"], 2.5)
        self.assertTrue(all(e["node"] == "n1" and e["task"] == "t1" for e in events))

    def test_proxy_env_tags_and_container_host(self):
        os.environ["RSI_PROXY_URL"] = "http://127.0.0.1:8787"
        env = rsi_ledger.proxy_env("eval", "20251001_120000_1", "django__django-1/x")
        self.assertEqual(env["OPENAI_BASE_URL"],
                         "http://host.docker.internal:8787/r/test-run/t/eval/20251001_120000_1/django__django-1%2Fx/v1")
        self.assertEqual(env["RSI_PROXY_PREFIX"],
                         "http://host.docker.internal:8787/r/test-run/t/eval/20251001_120000_1/django__django-1%2Fx")
        self.assertEqual(llm_proxy.split_tagged_path("/t/eval/20251001_120000_1/django__django-1%2Fx/v1/responses"),
                         ("eval", "20251001_120000_1", "django__django-1/x", "openai", "/v1/responses"))
        self.assertEqual(llm_proxy.split_tagged_path("/t/propose/n1/-/p/vllm/v1/chat/completions"),
                         ("propose", "n1", "-", "vllm", "/v1/chat/completions"))

    def test_tag_client_uses_bound_node(self):
        os.environ["RSI_PROXY_URL"] = "http://127.0.0.1:8787"

        class FakeClient:
            def __init__(self, base_url):
                self.base_url = base_url

            def with_options(self, base_url):
                return base_url

        with rsi_ledger.bind(node="child-7"):
            url = rsi_ledger.tag_client(FakeClient("https://api.openai.com/v1"), "diagnose", task="t3")
            vllm = rsi_ledger.tag_client(FakeClient("http://172.17.0.1:8000/v1"), "diagnose", task="t3")
            router = rsi_ledger.tag_client(FakeClient("https://openrouter.ai/api/v1"), "diagnose")
        self.assertEqual(url, "http://127.0.0.1:8787/r/test-run/t/diagnose/child-7/t3/v1")
        self.assertEqual(vllm, "http://127.0.0.1:8787/r/test-run/t/diagnose/child-7/t3/p/vllm/v1")
        self.assertEqual(router, "http://127.0.0.1:8787/r/test-run/t/diagnose/child-7/-/p/openrouter/v1")


class FakeUpstream(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeUpstream.seen.append((self.path, self.headers.get("Authorization"), body))
        if self.path.endswith("/responses"):
            reply = {"model": "gpt-5-mini-2025-08-07",
                     "usage": {"input_tokens": 1_000_000, "output_tokens": 100_000,
                               "input_tokens_details": {"cached_tokens": 400_000},
                               "output_tokens_details": {"reasoning_tokens": 60_000}}}
            status = 200
        else:
            reply = {"error": {"message": "rate limited"}}
            status = 429
        data = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TestProxy(LedgerCase):
    def setUp(self):
        super().setUp()
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        fake = llm_proxy.urlsplit(f"http://127.0.0.1:{self.upstream.server_port}")
        llm_proxy.ProxyHandler.upstreams = {"openai": fake, "vllm": fake}
        llm_proxy.ProxyHandler.prices = llm_proxy.load_prices(os.path.join(ROOT, "prices.json"))
        self.proxy = ThreadingHTTPServer(("127.0.0.1", 0), llm_proxy.ProxyHandler)
        threading.Thread(target=self.proxy.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.proxy.server_port}"

    def tearDown(self):
        for server in (self.proxy, self.upstream):
            server.shutdown()
            server.server_close()
        super().tearDown()

    def post(self, path, payload):
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer sk-test"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def wait_for_events(self, n):
        for _ in range(50):
            if os.path.exists(self.ledger) and len(read_ledger(self.ledger)) >= n:
                return read_ledger(self.ledger)
            time.sleep(0.05)
        self.fail("ledger events not written")

    def test_forwards_and_records_cost(self):
        status, reply = self.post("/t/eval/node1/task1/v1/responses", {"model": "gpt-5-mini", "input": "hi"})
        self.assertEqual(status, 200)
        self.assertEqual(reply["model"], "gpt-5-mini-2025-08-07")
        path, auth, _ = FakeUpstream.seen[-1]
        self.assertEqual(path, "/v1/responses")
        self.assertEqual(auth, "Bearer sk-test")
        (e,) = self.wait_for_events(1)
        self.assertEqual((e["type"], e["tag"], e["node"], e["task"]), ("llm_call", "eval", "node1", "task1"))
        # 600k uncached * 0.25 + 400k cached * 0.025 + 100k output * 2.00, per 1M tokens
        self.assertAlmostEqual(e["usd"], 0.15 + 0.01 + 0.2)
        self.assertEqual(e["reasoning_tokens"], 60_000)

    def test_upstream_error_is_passed_through(self):
        status, reply = self.post("/t/propose/n2/-/v1/chat/completions", {"model": "gpt-5-mini", "messages": []})
        self.assertEqual(status, 429)
        self.assertIn("error", reply)
        (e,) = self.wait_for_events(1)
        self.assertEqual(e["status"], 429)
        self.assertNotIn("usd", e)
        self.assertIn("error", e["error_body"])

    def test_routes_by_provider(self):
        status, _ = self.post("/t/eval/n3/t9/p/vllm/v1/responses", {"model": "qwen3-coder"})
        self.assertEqual(status, 200)
        self.assertEqual(FakeUpstream.seen[-1][0], "/v1/responses")
        (e,) = self.wait_for_events(1)
        self.assertEqual((e["provider"], e["node"], e["task"]), ("vllm", "n3", "t9"))
        status, _ = self.post("/t/eval/n3/t9/p/nowhere/v1/responses", {"model": "x"})
        self.assertEqual(status, 400)

    def test_run_label_comes_from_the_url(self):
        # The proxy process has no RSI_RUN of its own; the caller's label rides in /r/<run>.
        status, _ = self.post("/r/hgm-fixed-r0/t/eval/n3/t9/p/vllm/v1/responses", {"model": "qwen3-coder"})
        self.assertEqual(status, 200)
        self.assertEqual(FakeUpstream.seen[-1][0], "/v1/responses")
        (e,) = self.wait_for_events(1)
        self.assertEqual((e["run"], e["provider"], e["node"]), ("hgm-fixed-r0", "vllm", "n3"))
        self.assertEqual(llm_proxy.split_run("/t/eval/n/t/v1"), (None, "/t/eval/n/t/v1"))
        self.assertEqual(llm_proxy.split_run("/r/a%2Fb/t/eval/n/t/v1"), ("a/b", "/t/eval/n/t/v1"))

    def test_openrouter_calls_use_openrouter_prices(self):
        llm_proxy.ProxyHandler.upstreams["openrouter"] = llm_proxy.ProxyHandler.upstreams["openai"]
        llm_proxy.ProxyHandler.openrouter_prices = {"gpt-5-mini-2025-08-07": {"input": 1.0, "output": 1.0}}
        try:
            self.post("/t/propose/n4/-/p/openrouter/v1/responses", {"model": "x"})
            (e,) = self.wait_for_events(1)
        finally:
            llm_proxy.ProxyHandler.openrouter_prices = {}
        # 1M input + 100k output tokens at USD 1 per 1M each (cached tokens fall back to the input rate)
        self.assertAlmostEqual(e["usd"], 1.1)

    def test_price_prefix_prefers_longest(self):
        prices = {"gpt-5": {"input": 1}, "gpt-5-mini": {"input": 2}}
        self.assertEqual(llm_proxy.price_for("gpt-5-mini-2025", prices)["input"], 2)
        self.assertEqual(llm_proxy.price_for("gpt-5-2025", prices)["input"], 1)
        self.assertIsNone(llm_proxy.price_for("claude", prices))


class TestAnalyzer(unittest.TestCase):
    def test_waste_metrics(self):
        events = [
            {"type": "select_parent", "node": "initial"},
            {"type": "node_created", "node": "a", "parent": "initial"},
            {"type": "node_created", "node": "b", "parent": "initial"},
            {"type": "eval_result", "node": "initial", "score": [1, 0]},
            {"type": "eval_result", "node": "a", "score": [0, 0]},
            {"type": "eval_result", "node": "b", "score": [1, 1]},
            {"type": "eval", "phase": "setup", "node": "a", "task": "t1", "wall_s": 30},
            {"type": "eval", "phase": "agent_run", "node": "a", "task": "t1", "wall_s": 70},
            {"type": "eval", "phase": "agent_run", "node": "b", "task": "t1", "wall_s": 100},
            {"type": "eval", "phase": "agent_run", "node": "initial", "task": "t1", "wall_s": 100},
            {"type": "llm_call", "tag": "eval", "node": "a", "model": "gpt-5-mini", "usd": 1.0, "status": 200},
            {"type": "llm_call", "tag": "eval", "node": "b", "model": "gpt-5-mini", "usd": 3.0, "status": 200},
        ]
        text, nodes = analyze_ledger.report(events)
        self.assertEqual(nodes["initial"].children, ["a", "b"])
        # a and b were never parents: they hold 200 of 300 s and all 4 USD
        self.assertIn("| never chosen as parent | 2 / 3 | 66.7% | 100.0% |", text)
        self.assertIn("| broken (accuracy 0) | 1 / 3 | 33.3% | 25.0% |", text)
        # a (0.0) never beats initial (0.5); b (1.0) does
        self.assertIn("| lineage never beats its parent | 1 / 3 | 33.3% | 25.0% |", text)
        self.assertIn("Fixed overhead in evaluation (setup phase): 10.0%", text)

    def test_harness_eval_ids_map_to_nodes(self):
        # HGM's harness tags evaluations "<commit>_<i>"; the search events use "<commit>".
        events = [
            {"type": "node_created", "node": "20261017_073532_1", "parent": "initial"},
            {"type": "eval_result", "node": "20261017_073532_1", "score": [1]},
            {"type": "eval", "phase": "agent_run", "node": "20261017_073532_1_0", "task": "t1", "wall_s": 70},
            {"type": "llm_call", "tag": "eval", "node": "20261017_073532_1_0", "model": "gpt-5-mini",
             "usd": 1.0, "status": 200},
            {"type": "eval", "phase": "agent_run", "node": "unknown_0", "task": "t1", "wall_s": 5},
        ]
        _, nodes = analyze_ledger.report(events)
        n = nodes["20261017_073532_1"]
        self.assertEqual((n.eval_wall, n.eval_usd, n.eval_tasks), (70, 1.0, {"t1"}))
        self.assertNotIn("20261017_073532_1_0", nodes)
        self.assertIn("unknown_0", nodes)  # no matching node: left as is

    def test_broken_causes_and_timeouts(self):
        ev = []
        for i in "abcde":
            ev += [{"type": "node_created", "node": i, "parent": "initial"},
                   {"type": "eval_result", "node": i, "score": [0]}]
        ev += [
            {"type": "llm_call", "tag": "propose", "node": "a", "provider": "openai", "status": 401},
            {"type": "propose", "phase": "agent_run", "node": "b", "wall_s": 3600},
            {"type": "llm_call", "tag": "eval", "node": "d", "provider": "vllm", "status": 400,
             "error_body": "This model's maximum context length is 131072 tokens."},
        ]
        for i, w in zip("abcde", (100, 100, 3600, 100, 100)):
            ev.append({"type": "eval", "phase": "agent_run", "node": i, "task": "t", "wall_s": w})
        text, _ = analyze_ledger.report(ev)
        self.assertIn("| called a model it has no credentials for | 1 | 2.5% |", text)
        self.assertIn("| self-modification hit the time limit | 1 | 2.5% |", text)
        self.assertIn("| every evaluation hit the time limit | 1 | 90.0% |", text)
        self.assertIn("| exceeded the model's context length | 1 | 2.5% |", text)
        self.assertIn("| other (the modified agent fails the tasks) | 1 | 2.5% |", text)
        self.assertIn("limit: 1 of 5 (20.0%), holding 90.0% of evaluation wall time", text)


class TestTruncation(unittest.TestCase):
    def test_cut_at_nth_search_evaluation(self):
        ev = [{"type": "eval", "phase": "agent_run", "node": "initial", "ts": 0}]
        ev += [{"type": "eval", "phase": "agent_run", "node": f"n{i}", "ts": i} for i in range(1, 11)]
        ev += [{"type": "llm_call", "ts": 3.5}, {"type": "llm_call", "ts": 9.5}]
        kept, cutoff = analyze_ledger.truncate_at_evals(ev, 4)
        self.assertEqual(cutoff, 4)
        self.assertEqual(sum(1 for e in kept if e["type"] == "eval" and e["node"] != "initial"), 4)
        self.assertIn(3.5, [e["ts"] for e in kept])
        self.assertNotIn(9.5, [e["ts"] for e in kept])
        self.assertEqual(analyze_ledger.truncate_at_evals(ev, 50), (ev, None))

    @staticmethod
    def _crashed_run():
        """An event every 10 min for 0-2 h, a crash, then 10-10.5 h after --continue_from."""
        h, step = 3600, 600
        ts = list(range(0, 2 * h + 1, step)) + list(range(10 * h, 10 * h + h // 2 + 1, step))
        return [{"type": "llm_call", "ts": t} for t in ts]

    def test_gpu_hours_skip_idle_gaps_and_the_initial_agent(self):
        ev = [{"type": "eval", "phase": "agent_run", "node": "initial", "ts": -5 * 3600}]  # cached, not search
        ev += self._crashed_run()
        self.assertAlmostEqual(analyze_ledger.gpu_hours(ev), 2.5)
        self.assertEqual(analyze_ledger.gpu_hours([]), 0.0)

    def test_cut_at_gpu_hours(self):
        h, ev = 3600, self._crashed_run()
        kept, cutoff = analyze_ledger.truncate_at_hours(ev, 2.2)  # the 8 h gap does not count
        self.assertEqual(cutoff, 10 * h + 1200)
        self.assertEqual(len(kept), 13 + 3)
        kept, cutoff = analyze_ledger.truncate_at_hours(ev, 1)
        self.assertEqual((cutoff, len(kept)), (1 * h, 7))
        self.assertEqual(analyze_ledger.truncate_at_hours(ev, 9), (ev, None))
        # truncate(): hours take precedence over evaluations
        self.assertEqual(analyze_ledger.truncate(ev, max_evals=1, max_hours=1)[1], 1 * h)
        self.assertEqual(analyze_ledger.truncate(ev), (ev, None))


@unittest.skipUnless(os.path.exists(os.path.join(HGM_DIR, "tree.py")), "HGM repo not cloned")
class TestHGMTreeFix(unittest.TestCase):
    """Runs in a subprocess so the env flag is read fresh at import time."""

    SCRIPT = r"""
import sys, types
hu = types.ModuleType("hgm_utils"); hu.nodes = {}; hu.eval_agent = hu.sample_child = None
sys.modules["hgm_utils"] = hu
sys.path.insert(0, sys.argv[1])
from tree import Node
root = Node("initial", utility_measures=[1, 0, 1, 0])
child = Node("c1", parent_id=0, utility_measures=[1, 1, 1]); root.add_child(child)
gc = Node("c2", parent_id=1, utility_measures=[0, 0]); child.add_child(gc)
for _ in range(3):
    for n in list(hu.nodes.values()):
        n.get_decendant_evals(num_pseudo=10000)
print(root.num_evals, child.num_evals, len(root.get_decendant_evals(num_pseudo=10000)))
"""

    def run_tree(self, flag):
        env = dict(os.environ, HGM_FIX_DESCENDANT_EVALS=flag)
        out = subprocess.run([sys.executable, "-c", self.SCRIPT, HGM_DIR], env=env,
                             capture_output=True, text=True, check=True)
        return [int(x) for x in out.stdout.split()]

    def test_fixed_keeps_own_records(self):
        self.assertEqual(self.run_tree("1"), [4, 3, 9])

    def test_upstream_behaviour_still_available(self):
        root_n, child_n, _ = self.run_tree("0")
        self.assertGreater(root_n, 4)
        self.assertGreater(child_n, 3)


if __name__ == "__main__":
    unittest.main()
