"""BYO-AI: driving the full agent tool-loop against any OpenAI-compatible upstream."""
import json

from crucible.agent import endpoint_model, extract_openai_message, parse_openai_stream


def _sse(*objs):
    return [f"data: {json.dumps(o)}" for o in objs] + ["data: [DONE]"]


def test_parse_stream_content_tokens():
    lines = _sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
    )
    events = list(parse_openai_stream(lines))
    assert events[0] == ("token", "Hel")
    assert events[1] == ("token", "lo")
    assert events[-1] == ("final", {"role": "assistant", "content": "Hello"})


def test_parse_stream_ignores_noise_and_done():
    lines = ["", ": keep-alive", "data: [DONE]"]
    assert list(parse_openai_stream(lines)) == [("final", {"role": "assistant", "content": ""})]


def test_parse_stream_assembles_fragmented_tool_calls():
    lines = _sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "read", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"path":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"a.txt"}'}}]}}]},
    )
    final = list(parse_openai_stream(lines))[-1]
    assert final[0] == "final"
    msg = final[1]
    assert msg["content"] == ""
    assert msg["tool_calls"] == [
        {"id": "c1", "type": "function",
         "function": {"name": "read", "arguments": '{"path":"a.txt"}'}}
    ]


def test_parse_stream_synthesizes_missing_tool_call_id():
    lines = _sse({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "ls", "arguments": "{}"}}]}}]})
    msg = list(parse_openai_stream(lines))[-1][1]
    assert msg["tool_calls"][0]["id"] == "call_0"


def test_extract_plain_message():
    body = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    assert extract_openai_message(body) == {"role": "assistant", "content": "hi"}


def test_extract_preserves_tool_calls():
    tc = [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
    body = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": tc}}]}
    out = extract_openai_message(body)
    assert out["content"] == ""           # None normalized to empty string
    assert out["tool_calls"] == tc


def test_extract_handles_empty_choices():
    assert extract_openai_message({}) == {"role": "assistant", "content": ""}


def test_endpoint_model_url_normalization(monkeypatch):
    seen = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json, headers, timeout):
        seen["url"] = url
        seen["headers"] = headers
        seen["payload"] = json
        return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)

    # bare host -> /v1/chat/completions appended
    m = endpoint_model("http://node:8081", token="secret", model_name="glm")
    out = m([{"role": "user", "content": "hey"}], [])
    assert seen["url"] == "http://node:8081/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["payload"]["model"] == "glm"
    assert "tools" not in seen["payload"]          # no tools -> key omitted
    assert out == {"role": "assistant", "content": "ok"}

    # already-/v1 endpoint -> only /chat/completions appended
    endpoint_model("http://node:8081/v1")([], [{"type": "function"}])
    assert seen["url"] == "http://node:8081/v1/chat/completions"
    assert seen["payload"]["tools"] == [{"type": "function"}]   # tools forwarded when present

    # full path passed through unchanged, no token -> no auth header
    endpoint_model("http://node:8081/v1/chat/completions")([], [])
    assert seen["url"] == "http://node:8081/v1/chat/completions"
    assert seen["headers"] == {}
