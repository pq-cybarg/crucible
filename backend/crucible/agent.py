from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Literal

from crucible.audit import AuditLog
from crucible.permissions import PermissionPolicy
from crucible.tools.base import ToolRegistry

EventType = Literal["assistant", "assistant_delta", "tool_call", "tool_result", "done", "error"]
Model = Callable[[list[dict], list[dict]], dict]

# Housekeeping tools whose per-call chatter should be AGGREGATED by the UI (one high-level summary), not
# streamed as an individual notification each. They're also loop-guarded so a weak model can't spin on them
# forever: an identical repeat is skipped, and there's a per-run budget after which they're short-circuited.
MEMORY_TOOLS = frozenset({"recall_memory", "crystallize_memory", "recrystallize_memory",
                          "consolidate_memory", "link_memory", "prioritize_memory"})
MEMORY_BUDGET = 6                      # max memory-tool RUNS per turn before we tell the model to move on


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
        mem_seen: dict[str, int] = {}      # (name+args) → times seen, to kill duplicate memory loops
        mem_runs = 0                        # memory-tool calls actually executed this turn (budget)
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
                quiet = name in MEMORY_TOOLS       # UI aggregates these instead of notifying per-call
                self.audit.record("tool_call", {"name": name, "args": args})
                yield AgentEvent("tool_call", {"id": call["id"], "name": name, "args": args, "quiet": quiet})
                # Loop guard for housekeeping tools: skip exact repeats and cap the per-turn budget so a
                # weak model can't spin on recall/consolidate forever — short-circuit with a nudge instead.
                guard = None
                if quiet:
                    sig = name + "|" + json.dumps(args, sort_keys=True)
                    mem_seen[sig] = mem_seen.get(sig, 0) + 1
                    if mem_seen[sig] > 1:
                        guard = "already performed this exact memory operation — do not repeat it; continue."
                    elif mem_runs >= MEMORY_BUDGET:
                        guard = "memory-maintenance budget reached for this turn — stop organizing memory and answer the user."
                if guard is not None:
                    result = {"ok": True, "output": guard}
                else:
                    decision = self.permissions.check(name, args)
                    if not decision.allowed:
                        result = {"ok": False, "output": "", "error": decision.reason}
                    else:
                        res = self.tools.get(name).run(**args)
                        result = res.model_dump()
                    if quiet:
                        mem_runs += 1
                self.audit.record("tool_result", {"name": name, **result})
                yield AgentEvent("tool_result", {"id": call["id"], "name": name, "quiet": quiet, **result})
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


def _tools_unsupported(body: str) -> bool:
    """Does this error body mean the model rejected the OpenAI `tools` field? (Ollama/llama.cpp say
    'does not support tools'.) Used to drop tools and fall back to the text tool protocol."""
    b = (body or "").lower()
    return "does not support tools" in b or ("tool" in b and "support" in b and "not" in b)


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
                 max_tokens: int = 1024, served_model: str | None = None):
        url = chat_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + ("/chat/completions" if url.endswith("/v1") else "/v1/chat/completions")
        self.url = url
        self.models_url = url[: -len("/chat/completions")].rstrip("/") + "/models"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        # An explicit served_model is the exact tag the upstream expects (e.g. Ollama's "llama3.2:latest").
        # Absent one, model_name is often a Crucible REGISTRY LABEL like "ollama-localhost-11434" — which
        # the upstream does not know, so a chat 404s. We remember the registry label and, on a 404,
        # auto-resolve the real tag from GET /v1/models and retry — so BYO backends "just work".
        self.model_name = served_model or model_name
        self._explicit = served_model is not None
        self._resolved = served_model is not None
        self.max_tokens = max_tokens
        # Many local models (gpt-oss, plain llama in Ollama) reject the OpenAI `tools` field with a
        # 400 "does not support tools". When that happens we drop tools and retry, and remember it so
        # later calls skip tools — the hybrid loop then degrades to its TEXT tool protocol (the tools
        # are described in the system preamble), so the agent still works without native tool-calling.
        self.supports_tools = True

    def _resolve_served_model(self) -> bool:
        """Ask the upstream what it actually serves (GET /v1/models) and adopt the first model.
        Returns True if it changed the model name. Never raises — a failure just leaves it as-is."""
        import httpx
        try:
            r = httpx.get(self.models_url, headers=self.headers, timeout=10)
            r.raise_for_status()
            data = r.json().get("data") or []
            ids = [m.get("id") for m in data if m.get("id")]
        except (httpx.HTTPError, ValueError, KeyError):
            ids = []
        self._resolved = True
        if ids and self.model_name not in ids:
            self.model_name = ids[0]
            return True
        return False

    def _payload(self, messages: list[dict], tools: list[dict], stream: bool) -> dict:
        p: dict = {"model": self.model_name, "messages": messages, "max_tokens": self.max_tokens}
        if tools and self.supports_tools:
            p["tools"] = tools
        if stream:
            p["stream"] = True
        return p

    def _recover(self, status: int, body: str, tools: list[dict]) -> bool:
        """After a failed request, mutate state so a retry can succeed. Returns True if a retry is
        worth attempting: a 404 → auto-resolve the real served model; a tools-unsupported 400 → drop
        the tools param (remembered). False means the error is genuine and should surface."""
        if status == 404 and not self._explicit and self._resolve_served_model():
            return True
        if status == 400 and tools and self.supports_tools and _tools_unsupported(body):
            self.supports_tools = False
            return True
        return False

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        import httpx
        for _ in range(3):    # initial + up to two recoveries (model-resolve, tools-drop)
            r = httpx.post(self.url, json=self._payload(messages, tools, False),
                           headers=self.headers, timeout=600)
            if r.status_code >= 400 and self._recover(r.status_code, r.text, tools):
                continue
            r.raise_for_status()
            return extract_openai_message(r.json())
        r.raise_for_status()      # exhausted recoveries — surface the last error
        return extract_openai_message(r.json())

    def stream(self, messages: list[dict], tools: list[dict]) -> Iterator[tuple[str, object]]:
        import httpx
        # httpx exposes the status BEFORE the body streams, so a fast-failing 404 (wrong model tag) or
        # tools-unsupported 400 is caught and recovered before any tokens flow — the UI never sees a
        # broken stream. Bounded retries: model-resolve, then tools-drop.
        for _ in range(3):
            with httpx.stream("POST", self.url, json=self._payload(messages, tools, True),
                              headers=self.headers, timeout=600) as r:
                if r.status_code >= 400:
                    r.read()      # a streamed error body must be read before .text is available
                    if self._recover(r.status_code, r.text, tools):
                        continue
                r.raise_for_status()
                yield from parse_openai_stream(r.iter_lines())
                return


def endpoint_model(chat_url: str, token: str = "", model_name: str = "local",
                   max_tokens: int = 1024, served_model: str | None = None) -> EndpointModel:
    """Construct an EndpointModel (kept as a function for call-site compatibility). `served_model`
    pins the exact upstream model tag; without it the tag auto-resolves from /v1/models on a 404."""
    return EndpointModel(chat_url, token, model_name, max_tokens, served_model)
