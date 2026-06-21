from crucible.cli import parse_chat_response


def test_parse_chat_response_plain():
    data = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    out = parse_chat_response(data)
    assert out["content"] == "hi" and out["tool_calls"] == []


def test_parse_chat_response_tool_calls():
    data = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "1", "function": {"name": "bash", "arguments": "{}"}}]}}]}
    out = parse_chat_response(data)
    assert out["tool_calls"][0]["function"]["name"] == "bash"
