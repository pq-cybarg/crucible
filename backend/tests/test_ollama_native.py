"""Ollama-native /api/chat adapter — the path that honors resource limits (keep_alive/num_ctx) so
big local models stop freezing the machine. The OpenAI-compat endpoint ignores those."""
import json

from crucible.ollama_native import OllamaNativeModel, ollama_base, parse_ollama_stream


def test_ollama_base_from_any_url():
    for u in ("http://localhost:11434", "http://localhost:11434/v1",
              "http://localhost:11434/v1/chat/completions", "http://localhost:11434/api/chat"):
        assert ollama_base(u) == "http://localhost:11434"


def test_payload_carries_resource_limits():
    m = OllamaNativeModel("http://localhost:11434", served_model="qwen2.5:1.5b",
                          num_ctx=2048, keep_alive="0", max_output_tokens=128, num_gpu=10)
    p = m._payload([{"role": "user", "content": "hi"}], [], stream=True)
    assert p["model"] == "qwen2.5:1.5b" and p["stream"] is True
    assert p["keep_alive"] == "0"                               # unload after reply → frees RAM
    assert p["options"] == {"num_ctx": 2048, "num_predict": 128, "num_gpu": 10}


def test_payload_omits_unset_limits():
    m = OllamaNativeModel("http://localhost:11434", served_model="q")   # all defaults
    p = m._payload([{"role": "user", "content": "hi"}], [], stream=False)
    # keep_alive + num_ctx + num_gpu are only sent when set; num_predict defaults to the model's
    # output budget (parity with EndpointModel's max_tokens=1024) to avoid runaway generation.
    assert "keep_alive" not in p
    assert p["options"] == {"num_predict": 1024} and "num_ctx" not in p["options"]


def test_parse_native_stream_assembles_content():
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "Hel"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": "lo"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
    ]
    events = list(parse_ollama_stream(lines))
    assert events[0] == ("token", "Hel") and events[1] == ("token", "lo")
    assert events[-1] == ("final", {"role": "assistant", "content": "Hello"})


def test_parse_native_stream_normalizes_tool_calls():
    line = json.dumps({"message": {"role": "assistant", "content": "",
                                   "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}]},
                       "done": True})
    final = list(parse_ollama_stream([line]))[-1][1]
    assert final["tool_calls"][0]["function"]["name"] == "bash"
    assert final["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'   # dict → JSON string
    assert final["tool_calls"][0]["id"] == "call_0"


def test_call_recovers_from_tools_unsupported_400(monkeypatch):
    payloads = []

    class _Resp:
        def __init__(self, status): self.status_code = status
        text = '{"error":"model does not support tools"}'
        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError("should not raise after dropping tools")
        def json(self): return {"message": {"role": "assistant", "content": "ok"}}

    def fake_post(url, json, headers, timeout):
        payloads.append(json)
        assert url.endswith("/api/chat")                        # native endpoint, not /v1/...
        return _Resp(400 if "tools" in json else 200)

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)
    m = OllamaNativeModel("http://localhost:11434", served_model="heretic-20b:latest", keep_alive="0")
    out = m([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "bash"}}])
    assert "tools" in payloads[0] and "tools" not in payloads[1]
    assert out == {"role": "assistant", "content": "ok"} and m.supports_tools is False
