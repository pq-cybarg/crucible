from __future__ import annotations

from crucible.tools.base import ToolResult

# A simple shared todo list so the agent can plan multi-step work and track progress —
# the same "write a plan, check items off" skill a coding agent uses to stay on track.
STATUSES = ("pending", "in_progress", "done")


class TodoWrite:
    name = "todo_write"
    description = ("Record/replace the task plan for the current session. Pass the full list "
                   "each time. Each item: {task, status: pending|in_progress|done}.")
    parameters = {"type": "object", "properties": {
        "todos": {"type": "array", "items": {"type": "object", "properties": {
            "task": {"type": "string"}, "status": {"type": "string", "enum": list(STATUSES)}},
            "required": ["task"]}},
    }, "required": ["todos"]}

    def __init__(self, root=None):
        self.todos: list[dict] = []

    def run(self, todos: list) -> ToolResult:
        cleaned = []
        for t in todos:
            status = t.get("status", "pending")
            if status not in STATUSES:
                status = "pending"
            cleaned.append({"task": str(t.get("task", "")), "status": status})
        self.todos = cleaned
        done = sum(1 for t in cleaned if t["status"] == "done")
        lines = [f"[{'x' if t['status'] == 'done' else '~' if t['status'] == 'in_progress' else ' '}] {t['task']}"
                 for t in cleaned]
        return ToolResult(ok=True, output=f"{done}/{len(cleaned)} done\n" + "\n".join(lines))
