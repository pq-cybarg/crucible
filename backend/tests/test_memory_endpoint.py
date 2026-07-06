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


def test_memory_search_endpoint_lexical(tmp_path, monkeypatch):
    for k in ("EMBED",):
        monkeypatch.delenv(f"CRUCIBLE_{k}_ENDPOINT", raising=False)   # force lexical
    c = mkapp(tmp_path, monkeypatch, Summarizer())
    from crucible.memory import MemoryStore
    from crucible.config import get_settings
    store = MemoryStore(get_settings().data_dir / "memory")
    store.crystallize([{"role": "user", "content": "x"}], "abliteration and refusal removal", label="a")
    store.crystallize([{"role": "user", "content": "y"}], "quantization for speed", label="b")
    res = c.get("/api/memory/search", params={"q": "refusal removal"}).json()
    assert res["method"] == "lexical" and res["matches"][0]["label"] == "a"


def test_memory_graph_link_priority_endpoints(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, Summarizer())
    from crucible.memory import MemoryStore
    from crucible.config import get_settings
    store = MemoryStore(get_settings().data_dir / "memory")
    store.crystallize([{"role": "user", "content": "x"}], "alpha", label="a")
    store.crystallize([{"role": "user", "content": "y"}], "beta", label="b")
    # link + priority via the API
    assert c.post("/api/memory/link", json={"src": "m-0001", "dst": "m-0002", "type": "relates"}).json()["to"] == "m-0002"
    assert c.post("/api/memory/m-0001/priority", json={"priority": 7}).json()["priority"] == 7
    # graph shows the link edge
    g = c.get("/api/memory/graph").json()
    assert g["n_nodes"] == 2 and any(e["kind"] == "link" for e in g["edges"])
    # priority sort surfaces the weighted memory; index reports available sorts
    idx = c.get("/api/memory/index", params={"sort": "priority"}).json()
    assert idx["memories"][0]["key"] == "m-0001" and "recency" in idx["sorts"]
    # guard: self-link -> 422
    assert c.post("/api/memory/link", json={"src": "m-0001", "dst": "m-0001"}).status_code == 422


def test_memory_graph_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    from crucible.tools.memory import CrystallizeMemory, LinkMemory, PrioritizeMemory
    CrystallizeMemory().run(summary="a", content="alpha", label="a")
    CrystallizeMemory().run(summary="b", content="beta", label="b")
    assert LinkMemory().run(src="m-0001", dst="m-0002", type="refines").ok
    assert PrioritizeMemory().run(key="m-0001", priority=5).ok
    assert LinkMemory().run(src="m-0001", dst="m-0001").ok is False   # self-link rejected


def test_recall_tool_query_search(tmp_path, monkeypatch):
    c = mkapp(tmp_path, monkeypatch, Summarizer())
    c.post("/api/agent/compact", json={"messages": _long(6), "keep_recent": 2, "session_id": "s"})
    from crucible.tools.memory import RecallMemory
    out = RecallMemory().run(query="SUMMARY goal facts")
    assert out.ok and ("m-0001" in out.output or "no memories" in out.output)


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


def test_memory_management_tools(tmp_path, monkeypatch):
    # all memory operations are available as agent tools, sharing the same store
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    from crucible.tools.memory import (ConsolidateMemory, CrystallizeMemory, RecallMemory,
                                       RecrystallizeMemory)
    # 1. crystallize a new memory (agent-authored)
    r = CrystallizeMemory().run(summary="the plan", content="do X then Y", label="plan", session="s")
    assert r.ok and "m-0001" in r.output
    r2 = CrystallizeMemory().run(summary="a fact", content="the sky is blue", session="s")
    assert r2.ok and "m-0002" in r2.output

    # 2. consolidate them into a domain node (auto by session, keys omitted)
    con = ConsolidateMemory().run(summary="everything about s", session="s")
    assert con.ok and "consolidated 2" in con.output
    # after consolidation the two originals are filed under one top-level node
    assert len(RecallMemory().run().output.splitlines()) == 2   # header + 1 top-level

    # 3. recrystallize a leaf that has several messages
    conv = [{"role": "user", "content": f"m{i}"} for i in range(6)]
    from crucible.memory import MemoryStore
    from crucible.config import get_settings
    MemoryStore(get_settings().data_dir / "memory").crystallize(conv, "six turns", label="six")
    # find its key via recall index
    idx = RecallMemory().run().output
    assert "m-0004" in idx    # domain=m-0003, this leaf=m-0004
    rec = RecrystallizeMemory().run(key="m-0004", summaries=["first half", "second half"], labels=["a", "b"])
    assert rec.ok and "2 subchunks" in rec.output

    # guards
    assert CrystallizeMemory().run(summary="", content="x").ok is False
    assert ConsolidateMemory().run(summary="s").ok is False       # no keys, no session
    assert RecrystallizeMemory().run(key="m-9999", summaries=["x"]).ok is False
