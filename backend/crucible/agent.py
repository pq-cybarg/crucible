from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Literal

from crucible.audit import AuditLog
from crucible.permissions import PermissionPolicy
from crucible.tools.base import ToolRegistry

EventType = Literal["assistant", "assistant_delta", "tool_call", "tool_result", "done", "error"]
Model = Callable[[list[dict], list[dict]], dict]


@dataclass
class AgentEvent:
    type: EventType
    data: dict


class Agent:
    def __init__(self, model: Model, tools: ToolRegistry,
                 permissions: PermissionPolicy, audit: AuditLog, max_iters: int = 10,
                 stream: bool = True):
        self.model = model
        self.tools = tools
        self.permissions = permissions
        self.audit = audit
        self.max_iters = max_iters
        # When True and the model exposes .stream(), emit token-level assistant_delta events.
        self.stream = stream

    def run(self, messages: list[dict]) -> Iterator[AgentEvent]:
        convo = list(messages)
        streamer = getattr(self.model, "stream", None)
        use_stream = bool(self.stream) and callable(streamer)
        for _ in range(self.max_iters):
            try:
                if use_stream:
                    msg = {"role": "assistant", "content": ""}
                    for kind, payload in streamer(convo, self.tools.schemas()):
                        if kind == "token":
                            yield AgentEvent("assistant_delta", {"delta": payload})
                        else:  # "final"
                            msg = payload
                else:
                    msg = self.model(convo, self.tools.schemas())
            except Exception as exc:  # network/model failure (esp. BYO endpoints)
                yield AgentEvent("error", {"reason": f"model call failed: {exc}"})
                return
            convo.append(msg)
            if msg.get("content"):
                yield AgentEvent("assistant", {"content": msg["content"], "streamed": use_stream})
            calls = msg.get("tool_calls") or []
            if not calls:
                yield AgentEvent("done", {"content": msg.get("content", "")})
                return
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")
                self.audit.record("tool_call", {"name": name, "args": args})
                yield AgentEvent("tool_call", {"id": call["id"], "name": name, "args": args})
                decision = self.permissions.check(name, args)
                if not decision.allowed:
                    result = {"ok": False, "output": "", "error": decision.reason}
                else:
                    res = self.tools.get(name).run(**args)
                    result = res.model_dump()
                self.audit.record("tool_result", {"name": name, **result})
                yield AgentEvent("tool_result", {"id": call["id"], "name": name, **result})
                convo.append({"role": "tool", "tool_call_id": call["id"],
                              "content": result["output"] or result.get("error", "")})
        yield AgentEvent("error", {"reason": "max_iters exceeded"})


def chat_client_model(client, extract: Callable[[dict], dict]):
    """Adapter: wrap a Phase-1 ChatClient into a Model. `extract` maps the raw
    /v1/chat/completions JSON to an assistant message dict. Network path; not unit-tested."""
    import httpx

    def model(messages: list[dict], tools: list[dict]) -> dict:
        payload = {"model": "local", "messages": messages, "tools": tools, "max_tokens": 1024}
        r = httpx.post(f"{client.endpoint}/v1/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
        return extract(r.json())

    return model


def extract_openai_message(body: dict) -> dict:
    """Map a standard OpenAI /v1/chat/completions response to an assistant message
    dict (content + optional tool_calls) — the shape the Agent loop consumes."""
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    out: dict = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def parse_openai_stream(lines: Iterable[str]) -> Iterator[tuple[str, object]]:
    """Pure accumulator for an OpenAI streaming /v1/chat/completions response.
    Consumes raw SSE lines and yields ('token', text) for each content delta, then a
    single ('final', message_dict) with the assembled content + tool_calls. Handles
    fragmented tool_call deltas (id/name/arguments streamed in pieces, keyed by index)."""
    content_parts: list[str] = []
    tool_frags: dict[int, dict] = {}
    for raw in lines:
        line = raw.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = ((obj.get("choices") or [{}])[0]).get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
            yield ("token", delta["content"])
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_frags.setdefault(idx, {"id": None, "name": "", "args": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["args"] += fn["arguments"]
    msg: dict = {"role": "assistant", "content": "".join(content_parts)}
    if tool_frags:
        msg["tool_calls"] = [
            {"id": slot["id"] or f"call_{i}", "type": "function",
             "function": {"name": slot["name"], "arguments": slot["args"]}}
            for i, slot in sorted(tool_frags.items())
        ]
    yield ("final", msg)


class EndpointModel:
    """A Model for ANY OpenAI-compatible /v1/chat/completions endpoint (Crucible, Ollama,
    llama.cpp, vLLM, a remote node). This is what lets the full Crucible agent tool-loop run
    against a user's BYO backend: Crucible executes the tools locally, the endpoint generates.

    - __call__: blocking, returns the full assistant message (used by evals / non-stream).
    - stream:   yields ('token', delta) then ('final', msg) — token-level streaming for the UI.
    Network paths are not unit-tested; parse_openai_stream() and URL normalization are."""

    def __init__(self, chat_url: str, token: str = "", model_name: str = "local",
                 max_tokens: int = 1024):
        url = chat_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + ("/chat/completions" if url.endswith("/v1") else "/v1/chat/completions")
        self.url = url
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.model_name = model_name
        self.max_tokens = max_tokens

    def _payload(self, messages: list[dict], tools: list[dict], stream: bool) -> dict:
        p: dict = {"model": self.model_name, "messages": messages, "max_tokens": self.max_tokens}
        if tools:
            p["tools"] = tools
        if stream:
            p["stream"] = True
        return p

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        import httpx
        r = httpx.post(self.url, json=self._payload(messages, tools, False),
                       headers=self.headers, timeout=600)
        r.raise_for_status()
        return extract_openai_message(r.json())

    def stream(self, messages: list[dict], tools: list[dict]) -> Iterator[tuple[str, object]]:
        import httpx
        with httpx.stream("POST", self.url, json=self._payload(messages, tools, True),
                          headers=self.headers, timeout=600) as r:
            r.raise_for_status()
            yield from parse_openai_stream(r.iter_lines())


def endpoint_model(chat_url: str, token: str = "", model_name: str = "local",
                   max_tokens: int = 1024) -> EndpointModel:
    """Construct an EndpointModel (kept as a function for call-site compatibility)."""
    return EndpointModel(chat_url, token, model_name, max_tokens)
