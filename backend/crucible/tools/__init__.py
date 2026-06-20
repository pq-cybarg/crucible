from pathlib import Path

from crucible.tools.base import Tool, ToolRegistry, ToolResult, openai_schema  # noqa: F401
from crucible.tools.files import EditFile, ReadFile, WriteFile
from crucible.tools.search import Glob, Grep
from crucible.tools.shell import Bash


def default_registry(root: Path) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (ReadFile(root), WriteFile(root), EditFile(root),
                 Grep(root), Glob(root), Bash(root)):
        reg.register(tool)
    return reg
