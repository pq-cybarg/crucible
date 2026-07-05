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


def test_parse_sentinel_action_is_final():
    # "Action: none" means the model has no tool to run — treat the thought as the answer, not a tool
    from crucible.agent_react import parse_react
    s = parse_react("Thought: Hi there! How can I help?\nAction: none")
    assert s["kind"] == "final" and "Hi there" in s["text"]
    for sentinel in ("None", "null", "N/A", "finish", "nothing"):
        assert parse_react(f"Action: {sentinel}")["kind"] == "final"


def test_truncate_observation_caps_huge_output():
    from crucible.agent_react import truncate_observation
    small = "ok"
    assert truncate_observation(small) == small
    big = truncate_observation("x" * 100_000)
    assert len(big) < 7000 and "truncated" in big


def test_unknown_tool_in_ask_mode_does_not_prompt(tmp_path):
    # THE BUG: a hallucinated tool ('greet') under 'ask' used to fire an approval prompt and loop.
    # Now it returns a helpful observation and never asks the operator to approve a phantom tool.
    model = scripted(['Action: greet\nAction Input: {}', "Final Answer: hi"])
    prompted = []
    def approver(call_id, name, args):
        prompted.append(name)
        return True
    events = list(react_run(model, reg(tmp_path), [{"role": "user", "content": "hi"}],
                            PermissionPolicy(default="ask"),
                            AuditLog(tmp_path / "audit.jsonl"), approver=approver))
    assert prompted == []                                   # never prompted for the phantom tool
    assert not any(e.type == "permission_request" for e in events)
    tr = [e for e in events if e.type == "tool_result"][0]
    assert tr.data["ok"] is False and "Available tools" in tr.data["error"]


def test_sentinel_action_finishes_without_a_tool(tmp_path):
    # a plain greeting where the model emits "Action: none" should just finish, not loop on tools
    model = scripted(["Thought: greeting\nAction: none"])
    events = list(react_run(model, reg(tmp_path), [{"role": "user", "content": "hi"}],
                            PermissionPolicy(default="ask"), AuditLog(tmp_path / "audit.jsonl")))
    assert not any(e.type == "tool_call" for e in events)
    assert events[-1].type == "done"


def test_huge_tool_output_is_truncated_before_re_entering_the_conversation(tmp_path):
    # THE 400 BUG: a tool returning a massive output (e.g. `ls -R` over node_modules) overflowed the
    # model's context on the next turn. The observation fed back must be bounded.
    from crucible.agent_react import hybrid_run
    (tmp_path / "big.txt").write_text("y" * 100_000)
    turns, captured = iter([
        'Action: read_file\nAction Input: {"path": "big.txt"}',
        "Final Answer: read it",
    ]), {}

    def model(messages, tools):
        captured["last"] = messages          # capture the convo each call (2nd call has the observation)
        return {"role": "assistant", "content": next(turns)}

    list(hybrid_run(model, reg(tmp_path), [{"role": "user", "content": "read big.txt"}],
                    PermissionPolicy(default="allow"), AuditLog(tmp_path / "audit.jsonl")))
    # the observation that re-entered the conversation is bounded, not the full 100k chars
    obs = [m for m in captured["last"] if "truncated" in str(m.get("content", ""))]
    assert obs and all(len(str(m["content"])) < 8000 for m in obs)


def test_hybrid_preamble_mentions_both_paths(tmp_path):
    from crucible.agent_react import hybrid_preamble
    p = hybrid_preamble(reg(tmp_path).schemas())
    assert "function" in p.lower() and "Action:" in p and "Example:" in p


def test_hybrid_handles_native_tool_calls(tmp_path):
    import json as _json
    from crucible.agent_react import hybrid_run
    (tmp_path / "a.txt").write_text("hi there")
    # native model: returns OpenAI tool_calls, then a plain final answer
    steps = iter([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": _json.dumps({"path": "a.txt"})}}]},
        {"role": "assistant", "content": "the file says hi there"},
    ])
    model = lambda messages, tools: next(steps)
    events = list(hybrid_run(model, reg(tmp_path), [{"role": "user", "content": "read a.txt"}],
                             PermissionPolicy(default="allow"), AuditLog(tmp_path / "audit.jsonl")))
    types = [e.type for e in events]
    assert "tool_call" in types and "tool_result" in types
    tr = [e for e in events if e.type == "tool_result"][0]
    assert tr.data["ok"] and "hi there" in tr.data["output"]
    assert events[-1].type == "done"


def test_hybrid_handles_text_react_from_nonnative_model(tmp_path):
    from crucible.agent_react import hybrid_run
    (tmp_path / "a.txt").write_text("hello")
    # a model with NO function-calling: emits text ReAct, then a Final Answer
    steps = iter([
        {"role": "assistant", "content": 'Thought: read it\nAction: read_file\nAction Input: {"path": "a.txt"}'},
        {"role": "assistant", "content": "Final Answer: it says hello"},
    ])
    model = lambda messages, tools: next(steps)
    events = list(hybrid_run(model, reg(tmp_path), [{"role": "user", "content": "read a.txt"}],
                             PermissionPolicy(default="allow"), AuditLog(tmp_path / "audit.jsonl")))
    tr = [e for e in events if e.type == "tool_result"][0]
    assert tr.data["ok"] and "hello" in tr.data["output"]
    final = [e for e in events if e.type == "assistant"][-1]
    assert final.data["content"] == "it says hello"   # 'Final Answer:' stripped
    assert events[-1].type == "done"


def test_coerce_tool_name_snaps_near_misses():
    from crucible.agent_react import coerce_tool_name
    valid = ["list_dir", "read_file", "web_fetch"]
    assert coerce_tool_name("list_dir", valid) == "list_dir"      # exact
    assert coerce_tool_name("List_Dir", valid) == "list_dir"      # case
    assert coerce_tool_name("list_files", valid) == "list_dir"    # token overlap 'list'
    assert coerce_tool_name("read", valid) == "read_file"         # substring
    assert coerce_tool_name("xyzzy", valid) == "xyzzy"            # nothing close -> unchanged
    assert coerce_tool_name("anything", []) == "anything"         # no valid list


def _react_action_then_final():
    steps = iter([
        {"role": "assistant", "content": 'Action: read_file\nAction Input: {"path": "a.txt"}'},
        {"role": "assistant", "content": "Final Answer: done"},
    ])
    return lambda messages, tools: next(steps)


def test_ask_tool_emits_permission_request_and_approves(tmp_path):
    from crucible.agent_react import hybrid_run
    from crucible.permissions import PermissionPolicy
    (tmp_path / "a.txt").write_text("secret")
    events = list(hybrid_run(_react_action_then_final(), reg(tmp_path),
                             [{"role": "user", "content": "read a.txt"}],
                             PermissionPolicy(default="ask"), AuditLog(tmp_path / "audit.jsonl"),
                             approver=lambda cid, name, args: True))
    types = [e.type for e in events]
    assert "permission_request" in types
    tr = [e for e in events if e.type == "tool_result"][0]
    assert tr.data["ok"] and "secret" in tr.data["output"]


def test_ask_tool_denied_when_operator_rejects(tmp_path):
    from crucible.agent_react import hybrid_run
    from crucible.permissions import PermissionPolicy
    (tmp_path / "a.txt").write_text("secret")
    events = list(hybrid_run(_react_action_then_final(), reg(tmp_path),
                             [{"role": "user", "content": "read a.txt"}],
                             PermissionPolicy(default="ask"), AuditLog(tmp_path / "audit.jsonl"),
                             approver=lambda cid, name, args: False))
    tr = [e for e in events if e.type == "tool_result"][0]
    assert tr.data["ok"] is False and "rejected" in tr.data["error"]


def test_ask_tool_denied_without_approver(tmp_path):
    from crucible.agent_react import hybrid_run
    from crucible.permissions import PermissionPolicy
    events = list(hybrid_run(_react_action_then_final(), reg(tmp_path),
                             [{"role": "user", "content": "read a.txt"}],
                             PermissionPolicy(default="ask"), AuditLog(tmp_path / "audit.jsonl")))
    assert not any(e.type == "permission_request" for e in events)
    tr = [e for e in events if e.type == "tool_result"][0]
    assert "no approver" in tr.data["error"]
