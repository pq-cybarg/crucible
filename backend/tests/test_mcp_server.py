import io
import json

from crucible.mcp_server import CrucibleMcpServer, tools_list


def mkserver():
    calls = []

    def http(method, path, body):
        calls.append((method, path, body))
        if path == "/api/models":
            return [{"id": "qwen", "kind": "base"}]
        if path == "/api/runtime":
            return {"max_resident": 1, "resident": []}
        return {"ok": True, "path": path, "body": body}

    return CrucibleMcpServer(http=http), calls


def test_initialize_handshake():
    s, _ = mkserver()
    r = s.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert r["result"]["protocolVersion"]
    assert r["result"]["serverInfo"]["name"] == "crucible"


def test_notifications_return_none():
    s, _ = mkserver()
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list_has_core_tools():
    s, _ = mkserver()
    r = s.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in r["result"]["tools"]}
    assert {"crucible_list_models", "crucible_diagnose", "crucible_explain",
            "crucible_safety_suite", "crucible_chat", "crucible_runtime"} <= names
    # every tool advertises an input schema
    assert all("inputSchema" in t for t in r["result"]["tools"])


def test_tools_call_list_models_routes_to_http():
    s, calls = mkserver()
    r = s.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                  "params": {"name": "crucible_list_models", "arguments": {}}})
    assert ("GET", "/api/models", None) in calls
    payload = json.loads(r["result"]["content"][0]["text"])
    assert payload[0]["id"] == "qwen"


def test_tools_call_diagnose_builds_body():
    s, calls = mkserver()
    s.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
              "params": {"name": "crucible_diagnose", "arguments": {"base_id": "qwen"}}})
    assert ("POST", "/api/abliteration/diagnose", {"base_id": "qwen"}) in calls


def test_runtime_actions():
    s, calls = mkserver()
    s.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
              "params": {"name": "crucible_runtime", "arguments": {"action": "status"}}})
    assert ("GET", "/api/runtime", None) in calls
    s.handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
              "params": {"name": "crucible_runtime", "arguments": {"action": "start", "model_id": "m"}}})
    assert ("POST", "/api/runtime/start", {"model_id": "m"}) in calls


def test_chat_tool_builds_agent_run_body():
    s, calls = mkserver()
    s.handle({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
              "params": {"name": "crucible_chat", "arguments": {"message": "hi", "model_id": "qwen"}}})
    method, path, body = [c for c in calls if c[1] == "/api/agent/run"][0]
    assert body["messages"][0]["content"] == "hi"
    assert body["model_id"] == "qwen"


def test_unknown_tool_errors():
    s, _ = mkserver()
    r = s.handle({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                  "params": {"name": "nope", "arguments": {}}})
    assert r["error"]["code"] == -32602


def test_tool_exception_is_reported_as_mcp_error():
    def boom(method, path, body):
        raise RuntimeError("backend down")
    s = CrucibleMcpServer(http=boom)
    r = s.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                  "params": {"name": "crucible_list_models", "arguments": {}}})
    assert r["result"]["isError"] is True
    assert "backend down" in r["result"]["content"][0]["text"]


def test_serve_stdio_roundtrip():
    s, _ = mkserver()
    inp = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n\n')
    out = io.StringIO()
    s.serve_stdio(stdin=inp, stdout=out)
    line = out.getvalue().strip()
    assert json.loads(line)["result"]["tools"]
