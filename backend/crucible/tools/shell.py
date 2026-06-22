from __future__ import annotations
import subprocess
from pathlib import Path

from crucible.tools.base import ToolResult


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
        out = (p.stdout or "") + (p.stderr or "")
        return ToolResult(ok=(p.returncode == 0), output=out,
                          error=None if p.returncode == 0 else f"exit {p.returncode}")
