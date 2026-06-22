from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Callable, Iterator, Literal

from crucible.audit import AuditLog
from crucible.permissions import PermissionPolicy
from crucible.tools.base import ToolRegistry

EventType = Literal["assistant", "tool_call", "tool_result", "done", "error"]
Model = Callable[[list[dict], list[dict]], dict]


@dataclass
class AgentEvent:
    type: EventType
    data: dict


class Agent:
    def __init__(self, model: Model, tools: ToolRegistry,
                 permissions: PermissionPolicy, audit: AuditLog, max_iters: int = 10):
        self.model = model
        self.tools = tools
        self.permissions = permissions
        self.audit = audit
        self.max_iters = max_iters

    def run(self, messages: list[dict]) -> Iterator[AgentEvent]:
        convo = list(messages)
        for _ in range(self.max_iters):
            msg = self.model(convo, self.tools.schemas())
            convo.append(msg)
            if msg.get("content"):
                yield AgentEvent("assistant", {"content": msg["content"]})
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


def endpoint_model(chat_url: str, token: str = "", model_name: str = "local",
                   max_tokens: int = 1024):
    """Build a Model callable that drives ANY OpenAI-compatible /v1/chat/completions
    endpoint (Crucible, Ollama, llama.cpp, vLLM, a remote node). This is what lets the
    full Crucible agent tool-loop run against a user's BYO backend: Crucible executes the
    tools locally, the named endpoint does the generation. Network path; not unit-tested."""
    import httpx

    url = chat_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + ("/chat/completions" if url.endswith("/v1") else "/v1/chat/completions")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def model(messages: list[dict], tools: list[dict]) -> dict:
        payload: dict = {"model": model_name, "messages": messages, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = tools
        r = httpx.post(url, json=payload, headers=headers, timeout=600)
        r.raise_for_status()
        return extract_openai_message(r.json())

    return model
