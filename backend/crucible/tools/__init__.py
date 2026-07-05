from __future__ import annotations
from pathlib import Path

from crucible.tools.base import Tool, ToolRegistry, ToolResult, openai_schema  # noqa: F401
from crucible.tools.files import EditFile, ListDir, MultiEdit, ReadFile, WriteFile
from crucible.tools.search import Glob, Grep
from crucible.tools.shell import Bash
from crucible.tools.web import WebFetch, WebSearch
from crucible.tools.plan import TodoWrite
from crucible.tools.media import GenerateImage, Transcribe
from crucible.tools.memory import (ConsolidateMemory, CrystallizeMemory, RecallMemory,
                                   RecrystallizeMemory)


def default_registry(root: Path) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (ReadFile(root), WriteFile(root), EditFile(root), MultiEdit(root),
                 ListDir(root), Grep(root), Glob(root), Bash(root),
                 WebFetch(root), WebSearch(root), TodoWrite(root),
                 GenerateImage(root), Transcribe(root),
                 RecallMemory(root), CrystallizeMemory(root), RecrystallizeMemory(root),
                 ConsolidateMemory(root)):
        reg.register(tool)
    return reg
