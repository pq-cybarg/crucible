"""Live agent sessions + loadable memory/context slots — the shared substrate for the web tabs & TUI."""
import pytest

from crucible.agent_sessions import AgentSessionStore


def store(tmp_path):
    return AgentSessionStore(tmp_path / "agent_sessions.json")


def test_create_list_and_persist(tmp_path):
    s = store(tmp_path)
    a = s.create("frontend work", cwd="/proj/web")
    b = s.create("backend work", cwd="/proj/api", model_id="qwen")
    assert a["id"] == "a-0001" and b["id"] == "a-0002"
    ids = {c["id"] for c in s.list()}
    assert ids == {"a-0001", "a-0002"}
    # a fresh store reads the same sessions back (persisted)
    assert {c["id"] for c in store(tmp_path).list()} == ids


def test_subagents_group_under_parent(tmp_path):
    s = store(tmp_path)
    parent = s.create("lead", cwd="/proj")["id"]
    child = s.create("researcher", cwd="/proj", parent_id=parent)["id"]
    # top-level list excludes the subagent
    assert [c["id"] for c in s.list(parent_id=None)] == [parent]
    # the parent's children are its subagents
    assert [c["id"] for c in s.children(parent)] == [child]
    with pytest.raises(KeyError):
        s.create("orphan", cwd="/x", parent_id="ghost")


def test_delete_cascades_to_subagents(tmp_path):
    s = store(tmp_path)
    p = s.create("lead", cwd="/proj")["id"]
    s.create("sub", cwd="/proj", parent_id=p)
    assert s.delete(p) is True
    assert s.list() == []                       # parent + child both gone
    assert s.delete("ghost") is False


def test_slots_load_unload_and_detach(tmp_path):
    s = store(tmp_path)
    a = s.create("work", cwd="/proj")["id"]
    s.attach_slot(a, "memory", "m-0007", label="abliteration notes")
    s.attach_slot(a, "memory", "m-0007")        # dup ignored (idempotent)
    doc = s.read(a)
    assert len(doc["slots"]) == 1 and doc["slots"][0]["enabled"] is True
    # slot OUT without removing
    s.set_slot_enabled(a, "memory", "m-0007", False)
    assert s.read(a)["slots"][0]["enabled"] is False
    # re-attaching re-enables
    s.attach_slot(a, "memory", "m-0007")
    assert s.read(a)["slots"][0]["enabled"] is True
    # detach removes entirely
    s.detach_slot(a, "memory", "m-0007")
    assert s.read(a)["slots"] == []
    with pytest.raises(KeyError):
        s.detach_slot(a, "memory", "m-0007")


def test_slot_guards(tmp_path):
    s = store(tmp_path)
    a = s.create("work", cwd="/proj")["id"]
    with pytest.raises(ValueError):
        s.attach_slot(a, "bogus", "x")
    with pytest.raises(ValueError):
        s.attach_slot(a, "context", a)          # cannot load itself


def test_assembled_context_injects_only_enabled_slots(tmp_path):
    s = store(tmp_path)
    other = s.create("other", cwd="/proj")["id"]
    s.update(other, messages=[{"role": "user", "content": "prior discussion"}])
    a = s.create("work", cwd="/proj")["id"]
    s.update(a, messages=[{"role": "user", "content": "current task"}])
    s.attach_slot(a, "memory", "m-1", label="notes")
    s.attach_slot(a, "context", other)

    ctx = s.assembled_context(a, memory_text=lambda k: f"BODY OF {k}")
    joined = "\n".join(m["content"] for m in ctx)
    assert "BODY OF m-1" in joined                       # memory loaded
    assert "prior discussion" in joined                  # context loaded
    assert ctx[-1]["content"] == "current task"          # own convo comes last

    # slot the memory OUT → it disappears from the live context, nothing else changes
    s.set_slot_enabled(a, "memory", "m-1", False)
    joined2 = "\n".join(m["content"] for m in s.assembled_context(a, memory_text=lambda k: f"BODY OF {k}"))
    assert "BODY OF m-1" not in joined2 and "prior discussion" in joined2
