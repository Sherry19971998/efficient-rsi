"""Logging proxy for OpenAI API calls made by DGM / HGM.

Clients use a base URL of the form
    http://<proxy>/t/<tag>/<node>/<task>/v1                    (OpenAI)
    http://<proxy>/t/<tag>/<node>/<task>/p/<provider>/v1       (openrouter, vllm)
optionally preceded by /r/<run> (the caller's RSI_RUN, since the proxy has none)
(built by rsi_ledger.proxy_env / rsi_ledger.tag_client). The proxy strips the
/r/<run>/t/<tag>/<node>/<task> prefix, forwards the request to the upstream API
unchanged, and appends one `llm_call` event with token usage and USD cost to
the ledger. Streaming responses are forwarded but their usage is not parsed.

Usage:
    RSI_LEDGER=runs/ledger.jsonl python llm_proxy.py --port 8787 --prices prices.json
"""

import argparse
import http.client
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit, urlunsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rsi_ledger  # noqa: E402

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
    "accept-encoding",
}


def load_prices(path):
    with open(path) as f:
        table = json.load(f)
    return {k: v for k, v in table.items() if not k.startswith("_")}


def fetch_openrouter_prices(url="https://openrouter.ai/api/v1/models", timeout=30):
    """Per-model list prices from OpenRouter's public models API, as USD per 1M tokens.

    OpenRouter quotes per-token strings; tiered `overrides` (long-prompt rates) are ignored,
    and the provider OpenRouter picks may bill a different rate, so treat these as list prices.
    """
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = json.load(r)["data"]
    prices = {}
    for m in data:
        p = m.get("pricing") or {}
        try:
            entry = {"input": float(p["prompt"]) * 1e6, "output": float(p["completion"]) * 1e6}
        except (KeyError, TypeError, ValueError):
            continue
        if p.get("input_cache_read"):
            entry["cached_input"] = float(p["input_cache_read"]) * 1e6
        prices[m["id"]] = entry
    return prices


def price_for(model, prices):
    """Longest-prefix match, so 'gpt-5-mini-2025-08-07' uses the 'gpt-5-mini' entry."""
    best = None
    for name in prices:
        if model.startswith(name) and (best is None or len(name) > len(best)):
            best = name
    return prices[best] if best else None


def parse_usage(body):
    """Token counts from a Chat Completions or Responses API reply."""
    usage = body.get("usage") or {}
    if "input_tokens" in usage:  # Responses API
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
    else:  # Chat Completions API
        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return {
        "input_tokens": inp or 0,
        "cached_tokens": cached or 0,
        "output_tokens": out or 0,
        "reasoning_tokens": reasoning or 0,
    }


def cost_usd(usage, price):
    """Price entries are USD per 1M tokens; reasoning tokens are already inside output_tokens."""
    if not price:
        return None
    uncached = usage["input_tokens"] - usage["cached_tokens"]
    cached_rate = price.get("cached_input", price["input"])
    return round(
        (uncached * price["input"] + usage["cached_tokens"] * cached_rate
         + usage["output_tokens"] * price["output"]) / 1e6,
        6,
    )


def split_run(path):
    """'/r/<run>/t/...' -> ('<run>', '/t/...'); paths without a run segment come back unchanged."""
    if path.startswith("/r/"):
        _, _, run, rest = path.split("/", 3)
        return unquote(run), "/" + rest
    return None, path


def split_tagged_path(path):
    """'/t/<tag>/<node>/<task>[/p/<provider>]/v1/...' -> (tag, node, task, provider, '/v1/...')."""
    parts = path.split("/")
    if len(parts) >= 6 and parts[1] == "t":
        tag, node, task = (unquote(p) for p in parts[2:5])
        rest = parts[5:]
        provider = "openai"
        if len(rest) >= 2 and rest[0] == "p":
            provider, rest = rest[1], rest[2:]
        return tag, node, task, provider, "/" + "/".join(rest)
    return None, None, None, "openai", path


class ProxyHandler(BaseHTTPRequestHandler):
    upstreams = {}  # provider -> urlsplit result, set in main()
    prices = {}
    openrouter_prices = {}  # exact OpenRouter model id -> price
    timeout = 900

    def log_message(self, fmt, *args):
        pass

    def _json_error(self, status, message):
        """Errors raised by the proxy itself, in the OpenAI error shape so SDK clients can report them."""
        data = json.dumps({"error": {"message": message, "type": "rsi_proxy_error"}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def do_DELETE(self):
        self._forward()

    def _forward(self):
        run, tagged = split_run(self.path)
        tag, node, task, provider, path = split_tagged_path(tagged)
        upstream = self.upstreams.get(provider)
        if upstream is None:
            self._json_error(400, f"unknown provider {provider!r}; configure it with --{provider}-upstream")
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["Accept-Encoding"] = "identity"

        requested_model = None
        stream = False
        if body:
            try:
                req = json.loads(body)
                requested_model = req.get("model")
                stream = bool(req.get("stream"))
            except ValueError:
                pass

        start = time.time()
        conn_cls = http.client.HTTPSConnection if upstream.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(upstream.netloc, timeout=self.timeout)
        status, resp_body, error = 502, b"", None
        try:
            conn.request(self.command, upstream.path.rstrip("/") + path, body=body, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            resp_body = resp.read()
            self.send_response(status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        except Exception as e:
            error = f"{e.__class__.__name__}: {e}"
            try:
                self._json_error(502, error)
            except Exception:
                pass
        finally:
            conn.close()

        if self.command != "POST":
            return
        record = {
            "tag": tag, "node": node, "task": task, "provider": provider, "path": path, "status": status,
            "latency_s": round(time.time() - start, 3), "model": requested_model,
            "stream": stream or None, "error": error,
        }
        if run:
            record["run"] = run
        if status != 200 and resp_body:  # why the upstream refused (e.g. vLLM's 400s give no reason in its log)
            record["error_body"] = resp_body[:500].decode("utf-8", errors="replace")
        if status == 200 and not stream:
            try:
                reply = json.loads(resp_body)
                record["model"] = reply.get("model") or requested_model
                usage = parse_usage(reply)
                record.update(usage)
                price = (self.openrouter_prices.get(record["model"] or "") if provider == "openrouter"
                         else price_for(record["model"] or "", self.prices))
                record["usd"] = cost_usd(usage, price)
            except ValueError:
                pass
        rsi_ledger.event("llm_call", **record)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--bind", default="0.0.0.0", help="containers reach the proxy via host.docker.internal")
    parser.add_argument("--upstream", "--openai-upstream", dest="upstream", default="https://api.openai.com")
    parser.add_argument("--openrouter-upstream", default="https://openrouter.ai/api")
    parser.add_argument("--vllm-upstream", default=None, help="e.g. http://127.0.0.1:8000 (the local vLLM server)")
    parser.add_argument("--prices", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.json"))
    args = parser.parse_args()

    if not rsi_ledger.enabled():
        sys.exit("Set RSI_LEDGER to the ledger path before starting the proxy.")
    ProxyHandler.upstreams = {"openai": urlsplit(args.upstream), "openrouter": urlsplit(args.openrouter_upstream)}
    if args.vllm_upstream:
        ProxyHandler.upstreams["vllm"] = urlsplit(args.vllm_upstream)
    try:
        ProxyHandler.openrouter_prices = fetch_openrouter_prices()
        print(f"Loaded OpenRouter list prices for {len(ProxyHandler.openrouter_prices)} models")
    except Exception as e:  # metering still works; OpenRouter calls just get no USD
        print(f"Could not load OpenRouter prices ({e}); OpenRouter calls will have no USD")
    ProxyHandler.prices = load_prices(args.prices)
    server = ThreadingHTTPServer((args.bind, args.port), ProxyHandler)
    routes = ", ".join(f"{k} -> {urlunsplit(v)}" for k, v in ProxyHandler.upstreams.items())
    print(f"LLM proxy on {args.bind}:{args.port} ({routes}), ledger {os.getenv('RSI_LEDGER')}")
    server.serve_forever()


if __name__ == "__main__":
    main()
