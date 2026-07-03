from __future__ import annotations
from pathlib import Path

from crucible.tools.base import ToolResult


def _resolve(root: Path, path: str) -> Path:
    root = root.resolve()
    target = (root / path).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"path outside root: {path}")
    return target


class ReadFile:
    name = "read_file"
    description = "Read a UTF-8 text file relative to the working directory."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    def __init__(self, root: Path):
        self.root = Path(root)

    def run(self, path: str) -> ToolResult:
        try:
            return ToolResult(ok=True, output=_resolve(self.root, path).read_text())
        except ValueError as e:
            return ToolResult(ok=False, output="", error=str(e))
        except OSError as e:
            return ToolResult(ok=False, output="", error=f"cannot read: {e}")


class WriteFile:
    name = "write_file"
    description = "Write (overwrite) a UTF-8 text file relative to the working directory."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}

    def __init__(self, root: Path):
        self.root = Path(root)

    def run(self, path: str, content: str) -> ToolResult:
        try:
            t = _resolve(self.root, path)
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text(content)
            return ToolResult(ok=True, output=f"wrote {len(content)} bytes")
        except ValueError as e:
            return ToolResult(ok=False, output="", error=str(e))
        except OSError as e:
            return ToolResult(ok=False, output="", error=f"cannot write: {e}")


class EditFile:
    name = "edit_file"
    description = "Replace a unique substring in a file (fails if old text is missing or not unique)."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}

    def __init__(self, root: Path):
        self.root = Path(root)

    def run(self, path: str, old: str, new: str) -> ToolResult:
        try:
            t = _resolve(self.root, path)
            text = t.read_text()
            count = text.count(old)
            if count == 0:
                return ToolResult(ok=False, output="", error="old text not found")
            if count > 1:
                return ToolResult(ok=False, output="", error=f"old text not unique ({count} matches)")
            t.write_text(text.replace(old, new))
            return ToolResult(ok=True, output="edited 1 occurrence")
        except ValueError as e:
            return ToolResult(ok=False, output="", error=str(e))
        except OSError as e:
            return ToolResult(ok=False, output="", error=f"cannot edit: {e}")


class ListDir:
    name = "list_dir"
    description = "List the entries in a directory (relative to the working directory). Directories end with '/'."
    parameters = {"type": "object", "properties": {"path": {"type": "string", "description": "directory, default '.'"}}, "required": []}

    def __init__(self, root: Path):
        self.root = Path(root)

    def run(self, path: str = ".") -> ToolResult:
        try:
            d = _resolve(self.root, path)
            if not d.is_dir():
                return ToolResult(ok=False, output="", error="not a directory")
            entries = sorted(
                (e.name + ("/" if e.is_dir() else "") for e in d.iterdir()),
                key=lambda s: (not s.endswith("/"), s.lower()))
            return ToolResult(ok=True, output="\n".join(entries) or "(empty)")
        except ValueError as e:
            return ToolResult(ok=False, output="", error=str(e))
        except OSError as e:
            return ToolResult(ok=False, output="", error=f"cannot list: {e}")


class MultiEdit:
    name = "multi_edit"
    description = "Apply a sequence of unique-substring replacements to one file, atomically (all or nothing)."
    parameters = {"type": "object", "properties": {
        "path": {"type": "string"},
        "edits": {"type": "array", "items": {"type": "object", "properties": {
            "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["old", "new"]}},
    }, "required": ["path", "edits"]}

    def __init__(self, root: Path):
        self.root = Path(root)

    def run(self, path: str, edits: list) -> ToolResult:
        try:
            t = _resolve(self.root, path)
            text = t.read_text()
            for i, e in enumerate(edits):
                old, new = e.get("old", ""), e.get("new", "")
                count = text.count(old)
                if count == 0:
                    return ToolResult(ok=False, output="", error=f"edit {i}: old text not found")
                if count > 1:
                    return ToolResult(ok=False, output="", error=f"edit {i}: old text not unique ({count})")
                text = text.replace(old, new)
            t.write_text(text)
            return ToolResult(ok=True, output=f"applied {len(edits)} edits")
        except ValueError as e:
            return ToolResult(ok=False, output="", error=str(e))
        except OSError as e:
            return ToolResult(ok=False, output="", error=f"cannot edit: {e}")
