from __future__ import annotations
# Subagents / swarm. One agent loop is a single worker; a swarm delegates a task to several
# sub-agents (each its own agent loop with its own tools) and merges their results. Recursion
# — a sub-agent that itself spawns sub-agents — makes it fractal. This holds the pure
# orchestration: pull each sub-agent's final answer out of its event stream, merge a swarm's
# results, and the spawn_agent TOOL that lets a running agent branch the tree itself, bounded by
# a depth + total-count budget (a fork-bomb guard). The sub-agent runner reuses the hybrid loop.
from dataclasses import dataclass, field
from typing import Callable, Iterable

from crucible.tools.base import ToolResult


def collect_final(events: Iterable) -> str:
    """Extract a sub-agent's final answer from its AgentEvent stream (last assistant/done
    content; surfaces an error verbatim so the parent can see failures)."""
    final = ""
    for e in events:
        t = getattr(e, "type", None)
        data = getattr(e, "data", {}) or {}
        if t == "assistant":
            final = data.get("content", final) or final
        elif t == "done":
            final = data.get("content") or final
        elif t == "error":
            return f"[error: {data.get('reason', 'unknown')}]"
    return final


def run_subagent(runner: Callable[[str], Iterable], task: str) -> dict:
    """Run one sub-agent on a task via `runner` (task -> AgentEvent stream) and capture its
    final answer. `runner` is injected so this is testable without a live model."""
    try:
        result = collect_final(runner(task))
    except Exception as exc:      # a dead sub-agent must not kill the swarm
        result = f"[error: {exc}]"
    return {"task": task, "result": result}


def run_swarm(tasks: list[str], runner: Callable[[str], Iterable]) -> dict:
    """Run a sub-agent per task and merge. (Sequential here; a thread/async runner can drop in
    without changing callers.)"""
    results = [run_subagent(runner, t) for t in tasks if str(t).strip()]
    return merge_results(results)


def merge_results(results: list[dict]) -> dict:
    """Combine sub-agent results into one swarm result."""
    ok = [r for r in results if not str(r["result"]).startswith("[error")]
    combined = "\n\n".join(f"### {r['task']}\n{r['result']}" for r in results)
    return {"n": len(results), "succeeded": len(ok), "results": results, "combined": combined}


# --- fractal recursion: the spawn_agent tool + its fork-bomb budget --------------------------

@dataclass
class SpawnBudget:
    """Bounds a recursive agent tree so it can't runaway. `max_depth` caps how deep the tree
    goes (the root's own children are depth 1); `max_total` caps the TOTAL sub-agents across the
    whole tree (a fork-bomb guard) via a counter SHARED by every node in the tree."""
    max_depth: int = 2
    max_total: int = 8
    depth: int = 0
    _counter: list = field(default_factory=lambda: [0])

    def can_spawn(self) -> tuple[bool, str]:
        if self.max_depth <= 0 or self.max_total <= 0:
            return False, "spawning disabled"
        if self.depth >= self.max_depth:
            return False, f"max spawn depth {self.max_depth} reached — solve this level yourself"
        if self._counter[0] >= self.max_total:
            return False, f"spawn budget exhausted ({self.max_total} sub-agents used) — solve it yourself"
        return True, ""

    def take(self) -> "SpawnBudget":
        """Consume one spawn and return the child's budget (one level deeper, SAME shared total)."""
        self._counter[0] += 1
        return SpawnBudget(self.max_depth, self.max_total, self.depth + 1, self._counter)

    @property
    def used(self) -> int:
        return self._counter[0]


class SpawnAgent:
    """A tool the agent can CALL to delegate a focused subtask to a fresh sub-agent (its own tool
    loop, its own context). The sub-agent may itself have a spawn_agent tool (one level deeper),
    which is what makes the tree fractal — bounded by the SpawnBudget. `spawn(task)->str` is
    injected so the recursive wiring lives at the call site and this stays testable."""
    name = "spawn_agent"

    def __init__(self, spawn: Callable[[str], str], budget: "SpawnBudget"):
        self._spawn = spawn
        self._budget = budget
        remaining = max(0, budget.max_total - budget.used)
        self.description = (
            "Delegate a focused subtask to a fresh sub-agent (its own tool loop and clean "
            "context) and get back its final answer. Use it to decompose a big job, research a "
            "branch in isolation, or parallelize independent work. Sub-agents can spawn their own "
            f"(fractal), bounded — you are at depth {budget.depth}/{budget.max_depth}, "
            f"~{remaining} sub-agents left. If the budget is spent, just do the work yourself."
        )
        self.parameters = {
            "type": "object",
            "properties": {"task": {"type": "string",
                           "description": "the self-contained subtask for the sub-agent to complete"}},
            "required": ["task"],
        }

    def run(self, task: str = "", **_kw) -> ToolResult:
        if not str(task).strip():
            return ToolResult(ok=False, output="", error="task is required")
        ok, why = self._budget.can_spawn()
        if not ok:
            return ToolResult(ok=False, output="", error=why)
        try:
            return ToolResult(ok=True, output=self._spawn(str(task)))
        except Exception as exc:      # a dead sub-agent must not kill the parent
            return ToolResult(ok=False, output="", error=f"sub-agent failed: {exc}")


def make_spawn_tool(run_child: Callable[[str, "SpawnBudget"], str], budget: "SpawnBudget") -> SpawnAgent:
    """Build a spawn_agent tool bound to `budget`. On invocation it consumes one unit of the
    shared budget and calls run_child(task, child_budget); run_child is responsible for giving the
    child its OWN spawn tool (via make_spawn_tool(run_child, child_budget)) so recursion continues
    until the budget is spent. Injecting run_child keeps the whole tree testable without a model."""
    def spawn(task: str) -> str:
        child = budget.take()
        return run_child(task, child)
    return SpawnAgent(spawn, budget)
