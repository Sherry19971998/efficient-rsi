"""Tests for vllm_metrics_sampler.py (parsing vLLM's Prometheus text and the run summary)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vllm_metrics_sampler as vms  # noqa: E402

# Trimmed from the real endpoint on engr-liu01s (vLLM v1).
TEXT = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="vllm-qwen3c-local"} 19.0
vllm:num_requests_waiting{engine="0",model_name="vllm-qwen3c-local"} 1.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="vllm-qwen3c-local",reason="capacity"} 1.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="vllm-qwen3c-local",reason="deferred"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="vllm-qwen3c-local"} 0.9327268804832447
vllm:prefix_cache_queries_total{engine="0",model_name="vllm-qwen3c-local"} 2.271046722e+09
vllm:prefix_cache_hits_total{engine="0",model_name="vllm-qwen3c-local"} 1.892069232e+09
vllm:num_preemptions_total{engine="0",model_name="vllm-qwen3c-local"} 357.0
vllm:prompt_tokens_total{engine="0",model_name="vllm-qwen3c-local"} 2.271046722e+09
vllm:generation_tokens_total{engine="0",model_name="vllm-qwen3c-local"} 3.826841e+07
vllm:request_success_total{engine="0",finished_reason="stop",model_name="vllm-qwen3c-local"} 5.0
"""


class TestParse(unittest.TestCase):
    def test_real_endpoint_text(self):
        m = vms.parse(TEXT)
        self.assertEqual((m["running"], m["waiting"], m["preemptions"]), (19.0, 1.0, 357.0))
        self.assertAlmostEqual(m["kv_cache_usage"], 0.9327, places=3)
        self.assertEqual((m["waiting_capacity"], m["waiting_deferred"]), (1.0, 0.0))
        self.assertEqual(m["generation_tokens"], 3.826841e7)
        self.assertNotIn("request_success", m)


class TestSummary(unittest.TestCase):
    def test_idle_queue_preemption_and_throughput(self):
        s = [
            {"ts": 0, "running": 0, "waiting": 0, "kv_cache_usage": 0.1, "preemptions": 10, "prompt_tokens": 0,
             "generation_tokens": 0, "prefix_cache_hits": 0, "prefix_cache_queries": 0, "containers": 21},
            {"ts": 1800, "running": 20, "waiting": 2, "kv_cache_usage": 0.97, "preemptions": 12},
            {"ts": 2700, "error": "connection refused"},
            {"ts": 3600, "running": 10, "waiting": 0, "kv_cache_usage": 0.5, "preemptions": 15,
             "prompt_tokens": 3.6e6, "generation_tokens": 36000, "prefix_cache_hits": 3e6,
             "prefix_cache_queries": 3.6e6, "containers": 78},
        ]
        r = vms.summarize(s)
        self.assertEqual((r["samples"], r["errors"], r["hours"]), (4, 1, 1.0))
        self.assertAlmostEqual(r["idle_share"], 1 / 3, places=3)
        self.assertAlmostEqual(r["queued_share"], 1 / 3, places=3)
        self.assertAlmostEqual(r["kv_cache_full_share"], 1 / 3, places=3)
        self.assertEqual(r["preemptions_per_hour"], 5.0)
        self.assertEqual((r["prompt_tokens_per_s"], r["generation_tokens_per_s"]), (1000, 10.0))
        self.assertAlmostEqual(r["prefix_cache_hit_rate"], 0.833, places=3)
        self.assertEqual(r["max_containers"], 78)

    def test_too_few_samples(self):
        self.assertIn("note", vms.summarize([{"ts": 0, "running": 1}]))


if __name__ == "__main__":
    unittest.main()
