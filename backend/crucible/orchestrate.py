from __future__ import annotations
# Subagents / swarm. One agent loop is a single worker; a swarm delegates a task to several
# sub-agents (each its own agent loop with its own tools) and merges their results. Recursion
# — a sub-agent that itself spawns sub-agents — makes it fractal. This holds the pure
# orchestration: pull each sub-agent's final answer out of its event stream, and merge a
# swarm's results. The sub-agent runner reuses the existing hybrid tool loop.
from typing import Callable, Iterable


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
