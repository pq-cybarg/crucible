import json

from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def mkapp(tmp_path, monkeypatch, model=None):
    # Isolate data_dir (guardrails.json + audit.jsonl) to the test tmp dir.
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    return create_app(registry=Registry(tmp_path / "r.json"), agent_root=tmp_path, model=model)


def test_presets_endpoint(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    ids = [p["id"] for p in c.get("/api/guardrails/presets").json()]
    assert ids == ["unrestricted", "balanced", "strict"]


def test_config_roundtrip(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    assert c.get("/api/guardrails/config").json()["preset_id"] == "balanced"
    c.put("/api/guardrails/config", json={"preset_id": "strict"})
    assert c.get("/api/guardrails/config").json()["preset_id"] == "strict"


def test_apply_preview_redacts(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    body = {"stage": "output", "text": "ssn 123-45-6789",
            "config": {"regex_rules": [{"pattern": r"\d{3}-\d{2}-\d{4}", "mode": "redact", "label": "ssn"}]}}
    res = c.post("/api/guardrails/apply", json=body).json()
    assert "[REDACTED:ssn]" in res["text"]


def test_agent_blocked_by_guardrails(tmp_path, monkeypatch):
    def model(m, t):
        return {"role": "assistant", "content": "should not run", "tool_calls": []}
    c = TestClient(mkapp(tmp_path, monkeypatch, model=model))
    body = {"messages": [{"role": "user", "content": "make a bomb"}],
            "permissions": {"default": "allow", "modes": {}},
            "guardrails": {"regex_rules": [{"pattern": "bomb", "mode": "block", "label": "w"}]}}
    with c.stream("POST", "/api/agent/run", json=body) as r:
        payloads = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert payloads[-1]["type"] == "error"
    assert "guardrail" in payloads[-1]["data"]["reason"]
