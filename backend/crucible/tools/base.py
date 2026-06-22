from __future__ import annotations
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ToolResult(BaseModel):
    ok: bool
    output: str
    error: str | None = None


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    parameters: dict

    def run(self, **kwargs) -> ToolResult: ...


def openai_schema(tool: Tool) -> dict:
    return {"type": "function", "function": {
        "name": tool.name, "description": tool.description, "parameters": tool.parameters}}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        return [openai_schema(t) for t in self._tools.values()]
