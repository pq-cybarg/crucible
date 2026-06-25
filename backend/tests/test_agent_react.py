from crucible.agent_react import parse_react, react_preamble, react_run
from crucible.audit import AuditLog
from crucible.permissions import PermissionPolicy
from crucible.tools.base import ToolRegistry
from crucible.tools.files import ReadFile, WriteFile


def reg(tmp_path):
    r = ToolRegistry()
    r.register(WriteFile(root=tmp_path)); r.register(ReadFile(root=tmp_path))
    return r


def test_preamble_lists_tools(tmp_path):
    p = react_preamble(reg(tmp_path).schemas())
    assert "Action:" in p and "Final Answer:" in p
    assert "write_file" in p and "read_file" in p


def test_parse_action_with_json():
    s = parse_react('Thought: read it\nAction: read_file\nAction Input: {"path": "a.txt"}')
    assert s == {"kind": "action", "tool": "read_file", "input": {"path": "a.txt"}}


def test_parse_final_answer():
    s = parse_react("Thought: done\nFinal Answer: the file says hello")
    assert s["kind"] == "final" and s["text"] == "the file says hello"


def test_final_takes_precedence_when_first():
    s = parse_react("Final Answer: done now\nAction: read_file")
    assert s["kind"] == "final"


def test_bare_reply_is_final():
    assert parse_react("just a normal answer")["kind"] == "final"


def test_malformed_input_tolerated():
    s = parse_react("Action: read_file\nAction Input: a.txt")
    assert s["kind"] == "action" and s["tool"] == "read_file"
    assert s["input"] == {"input": "a.txt"}


def scripted(responses):
    it = iter(responses)
    def model(messages, tools):
        return {"role": "assistant", "content": next(it)}
    return model


def test_react_loop_executes_tool_then_finishes(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    model = scripted([
        'Thought: read it\nAction: read_file\nAction Input: {"path": "a.txt"}',
        "Final Answer: it says hello world",
    ])
    events = list(react_run(model, reg(tmp_path), [{"role": "user", "content": "read a.txt"}],
                            PermissionPolicy(default="allow"), AuditLog(tmp_path / "audit.jsonl")))
    types = [e.type for e in events]
    assert "tool_call" in types and "tool_result" in types
    assert events[-1].type == "done"
    tr = [e for e in events if e.type == "tool_result"][0]
    assert tr.data["ok"] is True and "hello world" in tr.data["output"]


def test_react_loop_respects_deny(tmp_path):
    model = scripted([
        'Action: write_file\nAction Input: {"path": "x.txt", "content": "hi"}',
        "Final Answer: blocked",
    ])
    events = list(react_run(model, reg(tmp_path), [{"role": "user", "content": "write"}],
                            PermissionPolicy(modes={"write_file": "deny"}),
                            AuditLog(tmp_path / "audit.jsonl")))
    tr = [e for e in events if e.type == "tool_result"][0]
    assert tr.data["ok"] is False
    assert not (tmp_path / "x.txt").exists()


def test_react_loop_unknown_tool(tmp_path):
    model = scripted(['Action: nonexistent\nAction Input: {}', "Final Answer: oops"])
    events = list(react_run(model, reg(tmp_path), [{"role": "user", "content": "go"}],
                            PermissionPolicy(default="allow"), AuditLog(tmp_path / "audit.jsonl")))
    tr = [e for e in events if e.type == "tool_result"][0]
    assert tr.data["ok"] is False and "no such tool" in tr.data["error"]
