import json

from crucible.agent import Agent
from crucible.audit import AuditLog
from crucible.permissions import PermissionPolicy
from crucible.tools.base import ToolRegistry
from crucible.tools.files import ReadFile, WriteFile


def make_registry(tmp_path):
    reg = ToolRegistry()
    reg.register(WriteFile(root=tmp_path)); reg.register(ReadFile(root=tmp_path))
    return reg


def scripted_model(responses):
    calls = iter(responses)
    def model(messages, tools):
        return next(calls)
    return model


def tool_call(id, name, args):
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_agent_executes_tool_then_finishes(tmp_path):
    model = scripted_model([
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("1", "write_file", {"path": "x.txt", "content": "hi"})]},
        {"role": "assistant", "content": "done writing", "tool_calls": []},
    ])
    agent = Agent(model=model, tools=make_registry(tmp_path),
                  permissions=PermissionPolicy(default="allow"),
                  audit=AuditLog(tmp_path / "audit.jsonl"))
    events = list(agent.run([{"role": "user", "content": "write hi to x.txt"}]))
    types = [e.type for e in events]
    assert "tool_call" in types and "tool_result" in types
    assert events[-1].type == "done"
    assert (tmp_path / "x.txt").read_text() == "hi"


def test_agent_blocks_denied_tool(tmp_path):
    model = scripted_model([
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("1", "write_file", {"path": "x.txt", "content": "hi"})]},
        {"role": "assistant", "content": "ok", "tool_calls": []},
    ])
    agent = Agent(model=model, tools=make_registry(tmp_path),
                  permissions=PermissionPolicy(modes={"write_file": "deny"}),
                  audit=AuditLog(tmp_path / "audit.jsonl"))
    events = list(agent.run([{"role": "user", "content": "write"}]))
    tr = [e for e in events if e.type == "tool_result"][0]
    assert tr.data["ok"] is False
    assert not (tmp_path / "x.txt").exists()


def test_agent_respects_max_iters(tmp_path):
    looping = {"role": "assistant", "content": None,
               "tool_calls": [tool_call("1", "read_file", {"path": "x.txt"})]}
    (tmp_path / "x.txt").write_text("hi")
    agent = Agent(model=lambda m, t: looping, tools=make_registry(tmp_path),
                  permissions=PermissionPolicy(default="allow"),
                  audit=AuditLog(tmp_path / "audit.jsonl"), max_iters=3)
    events = list(agent.run([{"role": "user", "content": "loop"}]))
    assert events[-1].type == "error"
    assert events[-1].data["reason"] == "max_iters exceeded"
    assert sum(1 for e in events if e.type == "tool_call") == 3


class _StreamModel:
    """A streaming Model: __call__ blocks; .stream() yields token deltas then final."""
    def __init__(self, chunks, final):
        self.chunks = chunks
        self.final = final
    def __call__(self, messages, tools):
        return self.final
    def stream(self, messages, tools):
        for c in self.chunks:
            yield ("token", c)
        yield ("final", self.final)


def test_agent_streams_token_deltas(tmp_path):
    model = _StreamModel(["Hel", "lo"], {"role": "assistant", "content": "Hello", "tool_calls": []})
    agent = Agent(model=model, tools=make_registry(tmp_path),
                  permissions=PermissionPolicy(default="allow"),
                  audit=AuditLog(tmp_path / "audit.jsonl"))
    events = list(agent.run([{"role": "user", "content": "hi"}]))
    deltas = [e.data["delta"] for e in events if e.type == "assistant_delta"]
    assert deltas == ["Hel", "lo"]
    final_assistant = [e for e in events if e.type == "assistant"][0]
    assert final_assistant.data == {"content": "Hello", "streamed": True}
    assert events[-1].type == "done"


def test_agent_stream_disabled_falls_back_to_blocking(tmp_path):
    model = _StreamModel(["x"], {"role": "assistant", "content": "Y", "tool_calls": []})
    agent = Agent(model=model, tools=make_registry(tmp_path),
                  permissions=PermissionPolicy(default="allow"),
                  audit=AuditLog(tmp_path / "audit.jsonl"), stream=False)
    events = list(agent.run([{"role": "user", "content": "hi"}]))
    assert not any(e.type == "assistant_delta" for e in events)
    assert [e for e in events if e.type == "assistant"][0].data["content"] == "Y"


def test_agent_emits_error_on_model_failure(tmp_path):
    def boom(messages, tools):
        raise RuntimeError("upstream down")
    agent = Agent(model=boom, tools=make_registry(tmp_path),
                  permissions=PermissionPolicy(default="allow"),
                  audit=AuditLog(tmp_path / "audit.jsonl"))
    events = list(agent.run([{"role": "user", "content": "hi"}]))
    assert events[-1].type == "error"
    assert "upstream down" in events[-1].data["reason"]
