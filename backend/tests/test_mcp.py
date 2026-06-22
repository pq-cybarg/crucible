from crucible.mcp import McpTool, build_request


def test_build_request_call_and_notify():
    assert build_request(3, "tools/list") == {"jsonrpc": "2.0", "method": "tools/list", "id": 3}
    n = build_request(4, "notifications/initialized", notify=True)
    assert "id" not in n and n["method"] == "notifications/initialized"
    p = build_request(1, "tools/call", {"name": "x"})
    assert p["params"] == {"name": "x"}


class StubClient:
    def __init__(self, text="ok", err=False):
        self.text, self.err, self.called = text, err, None

    def call_tool(self, name, arguments):
        self.called = (name, arguments)
        return self.text, self.err


def test_mcp_tool_wraps_call():
    spec = {"name": "search", "description": "web search",
            "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}}
    client = StubClient("result text")
    tool = McpTool(client, spec)
    assert tool.name == "search" and tool.parameters["properties"]["q"]["type"] == "string"
    res = tool.run(q="hello")
    assert res.ok and res.output == "result text"
    assert client.called == ("search", {"q": "hello"})


def test_mcp_tool_error():
    res = McpTool(StubClient("boom", err=True), {"name": "t"}).run()
    assert res.ok is False and res.error == "boom"
