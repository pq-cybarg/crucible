"""Memory endpoints + compaction integration + the recall_memory tool. Compaction crystallizes the
pre-compaction context into a versioned memory; the endpoints and the agent tool retrieve it."""
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry


class Summarizer:
    num_layers = 2
    def generate_chat(self, messages, n=256):
        return "SUMMARY: goal + key facts."


def mkapp(tmp_path, monkeypatch, adapter=None):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"),
                                 agent_root=tmp_path, abliteration_adapter=adapter))


def _long(n=20):
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"q{i} " + "x" * 40})
        msgs.append({"role": "assistant", "content": f"a{i} " + "y" * 40})
    return msgs


def test_compact_crystallizes_into_versioned_memory(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, Summarizer())
    r = c.post("/api/agent/compact", json={"messages": _long(20), "keep_recent": 4, "session_id": "s1"}).json()
    assert r["compacted"] is True and "memory" in r
    key = r["memory"]["key"]
    assert key == "m-0001" and r["memory"]["versioned"] is True

    # index shows the summary card (passthrough), NOT the bulk
    idx = c.get("/api/memory/index").json()
    assert idx["memories"][0]["key"] == key and "SUMMARY" in idx["memories"][0]["summary"]
    # opening it returns the full pre-compaction messages
    node = c.get(f"/api/memory/{key}").json()
    assert node["kind"] == "leaf" and len(node["messages"]) == len(_long(20))


def test_memory_consolidate_endpoint(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, Summarizer())
    c.post("/api/agent/compact", json={"messages": _long(6), "keep_recent": 2, "session_id": "s"})
    c.post("/api/agent/compact", json={"messages": _long(6), "keep_recent": 2, "session_id": "s"})
    dom = c.post("/api/memory/consolidate",
                 json={"keys": ["m-0001", "m-0002"], "summary": "domain", "label": "lab"}).json()
    assert dom["kind"] == "chunked"
    assert [m["key"] for m in c.get("/api/memory/index").json()["memories"]] == [dom["key"]]


def test_memory_recrystallize_explicit_subchunks(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, Summarizer())
    c.post("/api/agent/compact", json={"messages": _long(10), "keep_recent": 2})
    r = c.post("/api/memory/m-0001/recrystallize", json={"subchunks": [
        {"label": "one", "summary": "s1", "messages": [{"role": "user", "content": "a"}]},
        {"label": "two", "summary": "s2", "messages": [{"role": "user", "content": "b"}]},
    ]}).json()
    assert r["kind"] == "chunked" and len(r["children"]) == 2
    parent = c.get("/api/memory/m-0001").json()
    assert [ch["label"] for ch in parent["children"]] == ["one", "two"]


def test_memory_recrystallize_auto_split_uses_model(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, Summarizer())
    c.post("/api/agent/compact", json={"messages": _long(8), "keep_recent": 2})
    r = c.post("/api/memory/m-0001/recrystallize", json={"chunks": 3}).json()
    assert r["kind"] == "chunked" and len(r["children"]) >= 2   # auto-split + summarized


def test_memory_read_404(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, Summarizer())
    assert c.get("/api/memory/m-9999").status_code == 404


def test_recall_memory_tool(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, Summarizer())
    c.post("/api/agent/compact", json={"messages": _long(6), "keep_recent": 2, "session_id": "s"})
    from crucible.tools.memory import RecallMemory
    tool = RecallMemory()
    # no key -> the index (summaries)
    idx = tool.run()
    assert idx.ok and "m-0001" in idx.output and "SUMMARY" in idx.output
    # with key -> the full memory
    full = tool.run(key="m-0001")
    assert full.ok and "full context" in full.output
    assert tool.run(key="m-9999").ok is False
