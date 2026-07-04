from crucible.orchestrate import collect_final, merge_results, run_subagent, run_swarm


class E:
    def __init__(self, type, data): self.type = type; self.data = data


def test_collect_final_from_stream():
    events = [E("assistant", {"content": "thinking"}), E("tool_call", {}),
              E("assistant", {"content": "final answer"}), E("done", {"content": "final answer"})]
    assert collect_final(events) == "final answer"


def test_collect_final_surfaces_error():
    assert "error" in collect_final([E("error", {"reason": "model down"})])


def test_run_subagent_captures():
    runner = lambda task: [E("assistant", {"content": f"did {task}"}), E("done", {"content": f"did {task}"})]
    assert run_subagent(runner, "X")["result"] == "did X"


def test_run_subagent_survives_crash():
    def boom(task): raise RuntimeError("kaboom")
    r = run_subagent(boom, "X")
    assert r["result"].startswith("[error")


def test_run_swarm_merges():
    runner = lambda task: [E("done", {"content": f"result:{task}"})]
    out = run_swarm(["a", "b", "  ", "c"], runner)   # blanks dropped
    assert out["n"] == 3 and out["succeeded"] == 3
    assert "### a" in out["combined"] and "result:c" in out["combined"]


def test_merge_counts_failures():
    m = merge_results([{"task": "a", "result": "ok"}, {"task": "b", "result": "[error: x]"}])
    assert m["n"] == 2 and m["succeeded"] == 1
