"""Fractal sub-agents: the spawn_agent tool + its depth/total budget. Pure orchestration —
run_child is injected, so the whole recursive tree is exercised without a live model."""
from crucible.orchestrate import SpawnBudget, SpawnAgent, make_spawn_tool


def test_budget_depth_and_total_caps():
    b = SpawnBudget(max_depth=2, max_total=3)
    ok, _ = b.can_spawn()
    assert ok
    child = b.take()               # depth 1
    assert child.depth == 1 and b.used == 1
    grand = child.take()           # depth 2
    assert grand.depth == 2
    ok, why = grand.can_spawn()    # at max_depth -> refuse
    assert not ok and "depth" in why


def test_budget_total_is_shared_across_the_tree():
    b = SpawnBudget(max_depth=9, max_total=2)
    c1 = b.take()
    c2 = b.take()
    assert b.used == 2
    ok, why = c1.can_spawn()       # total exhausted, even though c1 is shallow
    assert not ok and "budget exhausted" in why


def test_disabled_budget_refuses():
    ok, why = SpawnBudget(max_depth=0).can_spawn()
    assert not ok and "disabled" in why


def test_spawn_tool_empty_task_errors():
    t = make_spawn_tool(lambda task, budget: "x", SpawnBudget())
    r = t.run(task="   ")
    assert r.ok is False and "required" in (r.error or "")


def test_spawn_tool_refuses_when_budget_spent():
    spent = SpawnBudget(max_depth=1, max_total=1)
    spent.take()                   # exhaust the total
    t = make_spawn_tool(lambda task, budget: "should not run", spent)
    r = t.run(task="do it")
    assert r.ok is False and "exhausted" in (r.error or "")


def test_fractal_tree_runs_and_caps_hold():
    seen = []

    def run_child(task, budget):
        seen.append((task, budget.depth))
        tool = make_spawn_tool(run_child, budget)   # child carries its own (deeper) spawn tool
        a = tool.run(task=task + ".a")
        b = tool.run(task=task + ".b")
        return f"{task}->[{a.ok},{b.ok}]"

    root = SpawnBudget(max_depth=2, max_total=5)
    top = make_spawn_tool(run_child, root)
    out = top.run(task="root")
    assert out.ok is True
    assert root.used <= 5                    # fork-bomb guard held
    assert max(d for _, d in seen) <= 2      # depth cap held
    assert seen[0] == ("root", 1)            # root's child is depth 1


def test_spawn_tool_survives_a_dead_subagent():
    def boom(task, budget):
        raise RuntimeError("kaboom")
    t = make_spawn_tool(boom, SpawnBudget())
    r = t.run(task="x")
    assert r.ok is False and "sub-agent failed" in (r.error or "")


def test_description_reports_remaining_budget():
    t = make_spawn_tool(lambda task, budget: "x", SpawnBudget(max_depth=3, max_total=7))
    assert t.name == "spawn_agent"
    assert "depth 0/3" in t.description and "task" in t.parameters["required"]
