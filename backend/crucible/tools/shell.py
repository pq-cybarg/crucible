from __future__ import annotations
import subprocess
from pathlib import Path

from crucible.tools.base import ToolResult


# Cap combined output so a runaway command (e.g. `ls -R` over node_modules) can't return 100KB+
# that bloats the tool result and the conversation. The agent loop truncates further for context.
_MAX_OUTPUT = 30000


def _cap(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return (f"{text[: limit - 300]}\n… [output truncated: {len(text) - limit + 300} more chars. "
            f"Re-run scoped (e.g. add a path, `head`, or a filter) to see the rest.]")


class Bash:
    name = "bash"
    description = "Run a shell command in the working directory and return combined output."
    parameters = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

    def __init__(self, root: Path, timeout: float = 30):
        self.root = Path(root)
        self.timeout = timeout

    def run(self, command: str) -> ToolResult:
        try:
            p = subprocess.run(command, shell=True, cwd=self.root, capture_output=True,
                               text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, output="", error="timeout")
        out = _cap((p.stdout or "") + (p.stderr or ""))
        return ToolResult(ok=(p.returncode == 0), output=out,
                          error=None if p.returncode == 0 else f"exit {p.returncode}")
