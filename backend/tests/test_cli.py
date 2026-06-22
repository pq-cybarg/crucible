from crucible import cli
from crucible.cli import parse_chat_response


def test_parse_chat_response():
    assert parse_chat_response({"choices": [{"message": {"content": "hi"}}]})["content"] == "hi"
    tc = parse_chat_response({"choices": [{"message": {"tool_calls": [{"function": {"name": "bash"}}]}}]})
    assert tc["tool_calls"][0]["function"]["name"] == "bash"


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    assert cli.load_settings()["perm"] == "ask"  # default
    cli.save_settings({"endpoint": "http://x/v1", "control": "http://x", "perm": "allow"})
    assert cli.load_settings()["perm"] == "allow"


def test_sessions_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path))
    assert cli.list_sessions() == []
    cli.save_session("work", [{"role": "user", "content": "hi"}])
    assert cli.list_sessions() == ["work"]
    assert cli.load_session("work")[0]["content"] == "hi"
    assert cli.load_session("missing") == []
