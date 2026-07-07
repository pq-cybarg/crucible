from __future__ import annotations
# Ollama's NATIVE /api/chat, used only to honor RESOURCE LIMITS the OpenAI-compat endpoint ignores:
# `keep_alive` (unload the model after a reply so it stops pinning ~15 GB of RAM between turns) and
# `options.num_ctx` / `num_predict` / `num_gpu` (KV-cache + generation + offload caps). This is what
# lets a big local model run "just below where it freezes the machine" — trading RAM for compute time.
# Same shape as EndpointModel (__call__ + stream) so the agent loop is agnostic; same tools-drop and
# model-resolve recovery so a no-native-tools model still degrades to the text tool protocol.
import json
from typing import Iterable, Iterator

from crucible.agent import _tools_unsupported


def ollama_base(chat_url: str) -> str:
    """The Ollama server root from any of its URLs (…, …/v1, …/v1/chat/completions) → the host:port."""
    u = chat_url.rstrip("/")
    for suffix in ("/v1/chat/completions", "/api/chat", "/v1"):
        if u.endswith(suffix):
            return u[: -len(suffix)].rstrip("/")
    return u


def is_ollama(base: str, timeout: float = 2.0) -> bool:
    """Does this endpoint speak Ollama's native API? (GET /api/version). Cheap, never raises."""
    import httpx
    try:
        return httpx.get(base.rstrip("/") + "/api/version", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def parse_ollama_stream(lines: Iterable[str]) -> Iterator[tuple[str, object]]:
    """Accumulate Ollama's native NDJSON stream: each line is {message:{content,...}, done}. Yields
    ('token', delta) per content chunk then one ('final', message) with assembled content + any
    tool_calls — the exact shape the agent loop consumes (mirrors parse_openai_stream)."""
    content = ""
    tool_calls: list[dict] = []
    for line in lines:
        line = (line or "").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        msg = obj.get("message") or {}
        piece = msg.get("content") or ""
        if piece:
            content += piece
            yield ("token", piece)
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            tool_calls.append({"id": tc.get("id") or f"call_{i}", "type": "function",
                               "function": {"name": fn.get("name", ""),
                                            "arguments": json.dumps(fn.get("arguments", {}))
                                            if isinstance(fn.get("arguments"), (dict, list)) else (fn.get("arguments") or "")}})
        if obj.get("done"):
            break
    final: dict = {"role": "assistant", "content": content}
    if tool_calls:
        final["tool_calls"] = tool_calls
    yield ("final", final)


class OllamaNativeModel:
    """Drive Ollama's /api/chat with resource limits. Interface-compatible with EndpointModel."""

    def __init__(self, chat_url: str, token: str = "", model_name: str = "local",
                 max_tokens: int = 1024, served_model: str | None = None,
                 num_ctx: int = 0, keep_alive: str = "", max_output_tokens: int = 0, num_gpu: int = -1):
        self.base = ollama_base(chat_url)
        self.url = self.base + "/api/chat"
        self.models_url = self.base + "/v1/models"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.model_name = served_model or model_name
        self._explicit = served_model is not None
        self.supports_tools = True
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.num_gpu = num_gpu
        # a per-request output cap: explicit resource limit wins, else the model's default budget
        self.max_output_tokens = max_output_tokens or max_tokens

    def _options(self) -> dict:
        opts: dict = {}
        if self.num_ctx > 0:
            opts["num_ctx"] = self.num_ctx
        if self.max_output_tokens > 0:
            opts["num_predict"] = self.max_output_tokens
        if self.num_gpu >= 0:
            opts["num_gpu"] = self.num_gpu
        return opts

    def _payload(self, messages: list[dict], tools: list[dict], stream: bool) -> dict:
        p: dict = {"model": self.model_name, "messages": messages, "stream": stream}
        opts = self._options()
        if opts:
            p["options"] = opts
        if self.keep_alive != "":
            p["keep_alive"] = self.keep_alive
        if tools and self.supports_tools:
            p["tools"] = tools
        return p

    def _resolve_served_model(self) -> bool:
        import httpx
        try:
            r = httpx.get(self.models_url, headers=self.headers, timeout=10)
            r.raise_for_status()
            ids = [m.get("id") for m in (r.json().get("data") or []) if m.get("id")]
        except (httpx.HTTPError, ValueError, KeyError):
            ids = []
        if ids and self.model_name not in ids:
            self.model_name = ids[0]
            return True
        return False

    def _recover(self, status: int, body: str, tools: list[dict]) -> bool:
        if status == 404 and not self._explicit and self._resolve_served_model():
            return True
        if status == 400 and tools and self.supports_tools and _tools_unsupported(body):
            self.supports_tools = False
            return True
        return False

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        import httpx
        for _ in range(3):
            r = httpx.post(self.url, json=self._payload(messages, tools, False),
                           headers=self.headers, timeout=600)
            if r.status_code >= 400 and self._recover(r.status_code, r.text, tools):
                continue
            r.raise_for_status()
            msg = (r.json().get("message") or {})
            out: dict = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                # normalize native tool_calls into OpenAI shape via the stream parser's logic
                final = list(parse_ollama_stream([json.dumps({"message": msg, "done": True})]))[-1][1]
                if isinstance(final, dict) and final.get("tool_calls"):
                    out["tool_calls"] = final["tool_calls"]
            return out
        r.raise_for_status()
        return {"role": "assistant", "content": ""}

    def stream(self, messages: list[dict], tools: list[dict]) -> Iterator[tuple[str, object]]:
        import httpx
        for _ in range(3):
            with httpx.stream("POST", self.url, json=self._payload(messages, tools, True),
                              headers=self.headers, timeout=600) as r:
                if r.status_code >= 400:
                    r.read()
                    if self._recover(r.status_code, r.text, tools):
                        continue
                r.raise_for_status()
                yield from parse_ollama_stream(r.iter_lines())
                return
