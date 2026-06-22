from __future__ import annotations
import re
from pathlib import Path

from crucible.tools.base import ToolResult


class Glob:
    name = "glob"
    description = "List files matching a glob pattern, relative to the working directory."
    parameters = {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}

    def __init__(self, root: Path):
        self.root = Path(root)

    def run(self, pattern: str) -> ToolResult:
        matches = sorted(
            str(p.relative_to(self.root)) for p in self.root.glob(pattern) if p.is_file()
        )
        return ToolResult(ok=True, output="\n".join(matches))


class Grep:
    name = "grep"
    description = "Search file contents for a regex; returns file:line:text matches."
    parameters = {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}

    def __init__(self, root: Path):
        self.root = Path(root)

    def run(self, pattern: str, path: str = ".") -> ToolResult:
        rx = re.compile(pattern)
        base = self.root / path
        files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        lines: list[str] = []
        for f in files:
            try:
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if rx.search(line):
                        lines.append(f"{f.relative_to(self.root)}:{i}:{line}")
            except (OSError, UnicodeDecodeError):
                continue
        return ToolResult(ok=True, output="\n".join(lines))
