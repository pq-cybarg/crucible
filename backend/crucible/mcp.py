from __future__ import annotations
# Minimal MCP (Model Context Protocol) stdio client. Spawns a configured MCP server,
# does the JSON-RPC handshake, lists its tools, and wraps each as a crucible Tool so
# the agent can call them — bringing the crucible CLI to Claude-Code tool parity.
import json
import os
import subprocess

from crucible.tools.base import ToolResult


def build_request(req_id: int, method: str, params=None, notify: bool = False) -> dict:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if not notify:
        msg["id"] = req_id
    if params is not None:
        msg["params"] = params
    return msg


class McpStdioClient:
    def __init__(self, command: str, args=None, env=None):
        self.proc = subprocess.Popen(
            [command, *(args or [])], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, env={**os.environ, **(env or {})})
        self._id = 0
        self._handshake()

    def _rpc(self, method: str, params=None, notify: bool = False):
        self._id += 1
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(build_request(self._id, method, params, notify)) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("mcp server closed the connection")
            try:
                resp = json.loads(line)
            except ValueError:
                continue
            if resp.get("id") == self._id:
                return resp

    def _handshake(self):
        self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                 "clientInfo": {"name": "crucible", "version": "0.1.0"}})
        self._rpc("notifications/initialized", notify=True)

    def list_tools(self) -> list[dict]:
        return ((self._rpc("tools/list") or {}).get("result") or {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        res = ((self._rpc("tools/call", {"name": name, "arguments": arguments}) or {}).get("result") or {})
        text = "\n".join(p.get("text", "") for p in res.get("content", []) if p.get("type") == "text")
        return text, bool(res.get("isError", False))

    def close(self) -> None:
        try:
            self.proc.terminate()
        except OSError:
            pass


class McpTool:
    def __init__(self, client, spec: dict):
        self._client = client
        self._mcp_name = spec["name"]
        self.name = spec["name"]
        self.description = spec.get("description", "")
        self.parameters = spec.get("inputSchema") or {"type": "object", "properties": {}}

    def run(self, **kwargs) -> ToolResult:
        text, is_err = self._client.call_tool(self._mcp_name, kwargs)
        return ToolResult(ok=not is_err, output=text, error=text if is_err else None)


def load_mcp(servers: dict):
    """Spawn each configured MCP server and collect (clients, wrapped_tools)."""
    clients, tools = [], []
    for _name, conf in (servers or {}).items():
        try:
            c = McpStdioClient(conf["command"], conf.get("args"), conf.get("env"))
            clients.append(c)
            tools.extend(McpTool(c, spec) for spec in c.list_tools())
        except (OSError, RuntimeError, KeyError):
            continue
    return clients, tools
