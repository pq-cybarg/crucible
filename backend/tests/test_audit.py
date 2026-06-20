from crucible.audit import AuditLog


def test_record_and_read(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record("tool_call", {"name": "bash", "command": "ls"})
    log.record("tool_result", {"name": "bash", "ok": True})
    entries = log.entries()
    assert [e["kind"] for e in entries] == ["tool_call", "tool_result"]
    assert entries[0]["seq"] == 0 and entries[1]["seq"] == 1
    assert entries[0]["data"]["command"] == "ls"


def test_persists_across_instances(tmp_path):
    AuditLog(tmp_path / "a.jsonl").record("x", {})
    assert len(AuditLog(tmp_path / "a.jsonl").entries()) == 1
