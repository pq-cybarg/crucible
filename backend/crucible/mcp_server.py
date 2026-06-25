from __future__ import annotations
# Crucible MCP server. Exposes Crucible's capabilities as MCP tools over stdio (JSON-RPC),
# so Claude Code — or any MCP-speaking agent — can DRIVE the whole system: list models,
# diagnose censorship, get the plain-language surgical report, run causal tracing,
# abliterate, run safety suites, chat with a model, and manage the runtime. This is the
# "let the AI evolve the system" handle. The HTTP caller is injectable, so the JSON-RPC
# dispatch and tool schemas are fully unit-tested without a live backend.
import json
import sys
from typing import Callable, Optional

PROTOCOL_VERSION = "2024-11-05"

# Each tool: (name, description, JSON-Schema properties, required, -> (METHOD, path, body-builder))
# body-builder maps the call arguments to the HTTP request body (or None for GET).
TOOLS_SPEC: list[dict] = [
    {"name": "crucible_list_models", "description": "List every model in the registry (base, abliterated, steered) with lineage.",
     "schema": {"type": "object", "properties": {}}, "http": ("GET", "/api/models", None)},
    {"name": "crucible_diagnose", "description": "Diagnose censorship in a model: where refusal lives, per-component impact, surgical verdict, and a plain-language report.",
     "schema": {"type": "object", "properties": {"base_id": {"type": "string"}}, "required": ["base_id"]},
     "http": ("POST", "/api/abliteration/diagnose", lambda a: {"base_id": a["base_id"]})},
    {"name": "crucible_explain", "description": "Plain-language 'surgical' diagnosis (where/how-we-know/target/repair/risk), optionally proven by causal intervention and translated to a language.",
     "schema": {"type": "object", "properties": {"base_id": {"type": "string"}, "language": {"type": "string"}, "include_causal": {"type": "boolean"}}, "required": ["base_id"]},
     "http": ("POST", "/api/abliteration/explain", lambda a: {"base_id": a["base_id"], "language": a.get("language", "en"), "include_causal": a.get("include_causal", False)})},
    {"name": "crucible_causal_trace", "description": "Activation-patching causal trace: prove WHICH layer causes refusal (not just correlates).",
     "schema": {"type": "object", "properties": {"base_id": {"type": "string"}}, "required": ["base_id"]},
     "http": ("POST", "/api/abliteration/causal-trace", lambda a: {"base_id": a["base_id"]})},
    {"name": "crucible_safety_suite", "description": "Run a safety eval suite (xstest_overrefusal, capability_control, or a supplied harmful set). Reports under/over-refusal.",
     "schema": {"type": "object", "properties": {"suite": {"type": "string"}, "model_id": {"type": "string"}, "path": {"type": "string"}, "use_judge": {"type": "boolean"}}, "required": ["suite"]},
     "http": ("POST", "/api/evals/safety-suite", lambda a: {k: a[k] for k in ("suite", "model_id", "path", "use_judge") if k in a})},
    {"name": "crucible_chat", "description": "Send a message to a model (by registry id) and get the reply — drives the full agent.",
     "schema": {"type": "object", "properties": {"message": {"type": "string"}, "model_id": {"type": "string"}}, "required": ["message"]},
     "http": ("POST", "/api/agent/run", lambda a: {"messages": [{"role": "user", "content": a["message"]}], "permissions": {"default": "allow", "modes": {}}, **({"model_id": a["model_id"]} if a.get("model_id") else {})})},
    {"name": "crucible_runtime", "description": "Manage the model runtime: action=status|start|stop, with model_id for start/stop.",
     "schema": {"type": "object", "properties": {"action": {"type": "string", "enum": ["status", "start", "stop"]}, "model_id": {"type": "string"}}, "required": ["action"]},
     "http": ("DYNAMIC", "", None)},  # resolved in dispatch
]

_BY_NAME = {t["name"]: t for t in TOOLS_SPEC}


def tools_list() -> list[dict]:
    return [{"name": t["name"], "description": t["description"], "inputSchema": t["schema"]}
            for t in TOOLS_SPEC]


class CrucibleMcpServer:
    def __init__(self, http: Optional[Callable[[str, str, Optional[dict]], dict]] = None,
                 base: str = "http://127.0.0.1:8400", token: str = ""):
        self.base = base.rstrip("/")
        self.token = token
        self._http = http or self._default_http

    # ---- JSON-RPC dispatch (pure, tested) -------------------------------
    def handle(self, req: dict) -> Optional[dict]:
        """Return a JSON-RPC response dict, or None for notifications (no id)."""
        rid = req.get("id")
        method = req.get("method")
        if method == "initialize":
            return self._ok(rid, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "crucible", "version": "1.0"},
            })
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "tools/list":
            return self._ok(rid, {"tools": tools_list()})
        if method == "tools/call":
            params = req.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name not in _BY_NAME:
                return self._err(rid, -32602, f"unknown tool: {name}")
            try:
                result = self._dispatch(name, args)
                return self._ok(rid, {"content": [{"type": "text", "text": json.dumps(result)}]})
            except Exception as exc:  # surface tool errors as MCP tool errors
                return self._ok(rid, {"content": [{"type": "text", "text": f"error: {exc}"}],
                                      "isError": True})
        return self._err(rid, -32601, f"method not found: {method}")

    def _dispatch(self, name: str, args: dict) -> dict:
        if name == "crucible_runtime":
            action = args.get("action", "status")
            if action == "status":
                return self._http("GET", "/api/runtime", None)
            if action in ("start", "stop"):
                return self._http("POST", f"/api/runtime/{action}", {"model_id": args.get("model_id")})
            raise ValueError(f"bad runtime action: {action}")
        spec = _BY_NAME[name]
        method, path, builder = spec["http"]
        body = builder(args) if builder else None
        return self._http(method, path, body)

    @staticmethod
    def _ok(rid, result) -> dict:
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    @staticmethod
    def _err(rid, code, message) -> dict:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

    # ---- transport (not unit-tested) ------------------------------------
    def _default_http(self, method: str, path: str, body: Optional[dict]) -> dict:
        import httpx
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        url = self.base + path
        if method == "GET":
            r = httpx.get(url, headers=headers, timeout=300)
        else:
            r = httpx.post(url, json=body or {}, headers=headers, timeout=600)
        r.raise_for_status()
        if "text/event-stream" in r.headers.get("content-type", ""):
            return {"stream": r.text}
        return r.json() if r.text else {}

    def serve_stdio(self, stdin=None, stdout=None) -> None:
        """Read JSON-RPC requests (one per line) from stdin, write responses to stdout."""
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle(req)
            if resp is not None:
                stdout.write(json.dumps(resp) + "\n")
                stdout.flush()


def main() -> None:
    import os
    server = CrucibleMcpServer(
        base=os.environ.get("CRUCIBLE_ENDPOINT", "http://127.0.0.1:8400"),
        token=os.environ.get("CRUCIBLE_API_TOKEN", ""),
    )
    server.serve_stdio()


if __name__ == "__main__":
    main()
