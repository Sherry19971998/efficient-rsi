"""Cost ledger for self-improving agent runs (DGM / HGM).

Every hook is a no-op unless the RSI_LEDGER environment variable points to a
JSONL file. Events are appended one JSON object per line; the LLM proxy
(llm_proxy.py) appends `llm_call` events to the same file.

Environment variables:
    RSI_LEDGER                  path of the JSONL ledger; unset = instrumentation off
    RSI_RUN                     free-form run label copied into every event (e.g. "hgm-fixed-seed0")
    RSI_PROXY_URL               proxy address as seen from the host, e.g. http://127.0.0.1:8787
    RSI_PROXY_URL_CONTAINER     proxy address as seen from containers,
                                default: RSI_PROXY_URL with the host replaced by host.docker.internal
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from urllib.parse import quote, urlsplit, urlunsplit

_lock = threading.Lock()
_context = threading.local()


@contextmanager
def bind(**fields):
    """Attach fields (e.g. node=<child id>) to host-side LLM calls tagged inside this block."""
    previous = getattr(_context, "fields", {})
    _context.fields = {**previous, **fields}
    try:
        yield
    finally:
        _context.fields = previous


def bound(name):
    return getattr(_context, "fields", {}).get(name)


def enabled():
    return bool(os.getenv("RSI_LEDGER"))


def event(type, **fields):
    """Append one event to the ledger."""
    path = os.getenv("RSI_LEDGER")
    if not path:
        return
    record = {"ts": time.time(), "type": type, "run": os.getenv("RSI_RUN", "")}
    record.update({k: v for k, v in fields.items() if v is not None})
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _lock:
        # O_APPEND keeps whole lines intact when the proxy process writes concurrently
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)


@contextmanager
def span(type, **fields):
    """Record the wall time of a block as one event, also when the block raises."""
    start = time.time()
    status = "ok"
    try:
        yield fields
    except BaseException as e:
        status = f"error: {type_name(e)}"
        raise
    finally:
        event(type, wall_s=round(time.time() - start, 3), status=status, **fields)


def type_name(e):
    return e.__class__.__name__


def container_cpu_s(container):
    """Cumulative CPU seconds used by a running container, or None."""
    if not enabled() or container is None:
        return None
    try:
        try:
            stats = container.stats(stream=False, one_shot=True)
        except TypeError:  # docker-py < 6 has no one_shot
            stats = container.stats(stream=False)
        return stats["cpu_stats"]["cpu_usage"]["total_usage"] / 1e9
    except Exception:
        return None


class Phases:
    """Times consecutive phases of one evaluation or proposal (setup -> agent_run -> test).

    Each call to phase() closes the previous phase and writes one event with its
    wall time and, when a container is attached, the container CPU it used.
    """

    def __init__(self, type, container=None, **fields):
        self.type = type
        self.fields = fields
        self.container = container
        self.name = None
        self.start = None
        self.cpu_start = None

    def attach(self, container):
        self.container = container
        if self.name is not None and self.cpu_start is None:
            self.cpu_start = container_cpu_s(container)

    def phase(self, name):
        self._close()
        self.name = name
        self.start = time.time()
        self.cpu_start = container_cpu_s(self.container)

    def done(self, **result):
        self._close(**result)

    def _close(self, **result):
        if self.name is None:
            return
        cpu_end = container_cpu_s(self.container)
        cpu_s = None
        if self.cpu_start is not None and cpu_end is not None:
            cpu_s = round(cpu_end - self.cpu_start, 3)
        event(self.type, phase=self.name, wall_s=round(time.time() - self.start, 3),
              container_cpu_s=cpu_s, **self.fields, **result)
        self.name = None


PROVIDERS = ("openai", "openrouter", "vllm")


def _tagged_prefix(base, tag, node, task):
    parts = [quote(str(x or "-"), safe="") for x in (tag, node, task)]
    # The proxy is a separate process with no RSI_RUN of its own, so the run label rides in the URL.
    run = os.getenv("RSI_RUN")
    head = f"/r/{quote(run, safe='')}" if run else ""
    return f"{base.rstrip('/')}{head}/t/{parts[0]}/{parts[1]}/{parts[2]}"


def _tagged_url(base, tag, node, task, provider="openai"):
    prefix = _tagged_prefix(base, tag, node, task)
    return f"{prefix}/v1" if provider == "openai" else f"{prefix}/p/{provider}/v1"


def provider_of(base_url):
    """Which upstream a client talks to, judged from its base URL."""
    url = str(base_url)
    if "openrouter.ai" in url:
        return "openrouter"
    if "api.openai.com" in url:
        return "openai"
    return "vllm"


def _container_proxy_base():
    explicit = os.getenv("RSI_PROXY_URL_CONTAINER")
    if explicit:
        return explicit
    host_url = os.getenv("RSI_PROXY_URL")
    if not host_url:
        return None
    u = urlsplit(host_url)
    netloc = "host.docker.internal" + (f":{u.port}" if u.port else "")
    return urlunsplit((u.scheme, netloc, u.path, "", ""))


def proxy_env(tag, node=None, task=None):
    """Environment variables that route a container's LLM calls through the proxy.

    OPENAI_BASE_URL covers clients built with openai.OpenAI(); RSI_PROXY_PREFIX lets
    agent code that builds its own base URL (vLLM, OpenRouter) append /p/<provider>/v1.
    """
    if not enabled():
        return {}
    base = _container_proxy_base()
    if not base:
        return {}
    return {"OPENAI_BASE_URL": _tagged_url(base, tag, node, task),
            "RSI_PROXY_PREFIX": _tagged_prefix(base, tag, node, task)}


def tag_client(client, tag, node=None, task=None):
    """Route a host-side OpenAI client through the proxy, tagged with tag/node/task."""
    base = os.getenv("RSI_PROXY_URL")
    if not enabled() or not base or not hasattr(client, "with_options"):
        return client
    node = node if node is not None else bound("node")
    task = task if task is not None else bound("task")
    provider = provider_of(getattr(client, "base_url", ""))
    return client.with_options(base_url=_tagged_url(base, tag, node, task, provider))
