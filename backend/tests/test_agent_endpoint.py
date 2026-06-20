import json

from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def tool_call(id, name, args):
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_agent_run_streams_events(tmp_path):
    responses = iter([
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("1", "write_file", {"path": "y.txt", "content": "hey"})]},
        {"role": "assistant", "content": "wrote it", "tool_calls": []},
    ])
    app = create_app(registry=Registry(tmp_path / "r.json"),
                     agent_root=tmp_path, model=lambda m, t: next(responses))
    c = TestClient(app)
    body = {"messages": [{"role": "user", "content": "write hey to y.txt"}],
            "permissions": {"default": "allow", "modes": {}}}
    with c.stream("POST", "/api/agent/run", json=body) as r:
        payloads = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert any(p["type"] == "tool_call" for p in payloads)
    assert payloads[-1]["type"] == "done"
    assert (tmp_path / "y.txt").read_text() == "hey"
