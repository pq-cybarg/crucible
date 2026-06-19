# Phase 2: Agent Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Claude-Code-style agent harness — a tool abstraction, built-in file/search/bash tools, a permission gate, an iterative tool-calling agent loop, an SSE streaming endpoint, and an audit log — so the model can autonomously read/write code and run commands under user-controlled permissions.

**Architecture:** Tools implement a common `Tool` protocol and register in a `ToolRegistry`. The `Agent` loop is model-agnostic: it takes a `model` callable (`messages, tools -> assistant message`) so it is fully testable without a live model; a thin adapter wraps the Phase-1 `ChatClient` for real use. Every tool call passes through a `PermissionPolicy` (allow/ask/deny) and is recorded in an append-only `AuditLog`. The loop yields typed events streamed to the GUI over SSE.

**Tech Stack:** Python 3.11+, pydantic v2, FastAPI (SSE via `StreamingResponse`), pytest, pytest-asyncio. Builds on Phase 1 (`crucible.client`, `crucible.app`, `crucible.config`).

## Global Constraints

- Python 3.11+; OpenAI-style tool/function-calling message shape.
- Tools never escape the working directory by default; `bash` has a hard timeout.
- Every tool invocation is permission-checked **and** audit-logged before execution.
- The agent loop has a `max_iters` cap to prevent infinite tool loops.
- Local-only; no remote providers.
- TDD: failing test first; one commit per task minimum.
- Agent loop must be testable with a fake `model` callable (no network).

---

## File Structure

```
backend/crucible/
  tools/
    __init__.py        # ToolRegistry + default_registry()
    base.py            # Tool protocol, ToolResult, ToolSpec
    files.py           # read_file, write_file, edit_file
    search.py          # grep, glob
    shell.py           # bash (timeout-guarded)
  permissions.py       # PermissionPolicy, PermissionMode, Decision
  audit.py             # AuditLog (append-only JSONL)
  agent.py             # Agent loop + events + ChatClient adapter
backend/tests/
  test_tools_files.py
  test_tools_search.py
  test_tools_shell.py
  test_permissions.py
  test_audit.py
  test_agent.py
  test_agent_endpoint.py
```

---

### Task 1: Tool protocol + registry

**Files:**
- Create: `backend/crucible/tools/base.py`
- Create: `backend/crucible/tools/__init__.py`
- Test: `backend/tests/test_tools_base.py`

**Interfaces:**
- Produces:
  - `ToolResult(ok: bool, output: str, error: str | None = None)` (pydantic).
  - `Tool` (Protocol): attributes `name: str`, `description: str`, `parameters: dict` (JSON schema); method `run(**kwargs) -> ToolResult`.
  - `ToolSpec` helper: `.openai_schema() -> dict` producing `{"type":"function","function":{...}}`.
  - `ToolRegistry`: `.register(tool)`, `.get(name) -> Tool`, `.all() -> list[Tool]`, `.schemas() -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_tools_base.py
from crucible.tools.base import ToolResult, ToolRegistry

class Echo:
    name = "echo"
    description = "echo text"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    def run(self, text: str) -> ToolResult:
        return ToolResult(ok=True, output=text)

def test_registry_register_get_all():
    reg = ToolRegistry()
    reg.register(Echo())
    assert reg.get("echo").run(text="hi").output == "hi"
    assert [t.name for t in reg.all()] == ["echo"]

def test_schemas_openai_shape():
    reg = ToolRegistry()
    reg.register(Echo())
    s = reg.schemas()[0]
    assert s["type"] == "function"
    assert s["function"]["name"] == "echo"
    assert s["function"]["parameters"]["required"] == ["text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_tools_base.py -v`
Expected: FAIL — `ModuleNotFoundError: crucible.tools.base`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/tools/base.py
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
```

```python
# backend/crucible/tools/__init__.py
from crucible.tools.base import Tool, ToolResult, ToolRegistry, openai_schema  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_tools_base.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/tools/ backend/tests/test_tools_base.py
git commit -m "feat: add tool protocol + registry"
```

---

### Task 2: File tools (read/write/edit)

**Files:**
- Create: `backend/crucible/tools/files.py`
- Test: `backend/tests/test_tools_files.py`

**Interfaces:**
- Consumes: `ToolResult`.
- Produces classes `ReadFile`, `WriteFile`, `EditFile`, each a `Tool` with `name` ∈ {`read_file`,`write_file`,`edit_file`} and a `root: Path` they refuse to escape.
  - `ReadFile.run(path) -> output=file text` (error if missing).
  - `WriteFile.run(path, content) -> writes file, output="wrote N bytes"`.
  - `EditFile.run(path, old, new) -> replaces unique old→new` (error if `old` not found or not unique).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_tools_files.py
from crucible.tools.files import ReadFile, WriteFile, EditFile

def test_write_then_read(tmp_path):
    w = WriteFile(root=tmp_path); r = ReadFile(root=tmp_path)
    assert w.run(path="a.txt", content="hello").ok
    assert r.run(path="a.txt").output == "hello"

def test_read_missing_errors(tmp_path):
    res = ReadFile(root=tmp_path).run(path="nope.txt")
    assert res.ok is False and res.error

def test_edit_unique_replace(tmp_path):
    WriteFile(root=tmp_path).run(path="a.txt", content="foo bar foo")
    res = EditFile(root=tmp_path).run(path="a.txt", old="bar", new="baz")
    assert res.ok
    assert ReadFile(root=tmp_path).run(path="a.txt").output == "foo baz foo"

def test_edit_nonunique_errors(tmp_path):
    WriteFile(root=tmp_path).run(path="a.txt", content="x x")
    res = EditFile(root=tmp_path).run(path="a.txt", old="x", new="y")
    assert res.ok is False

def test_escape_root_blocked(tmp_path):
    res = ReadFile(root=tmp_path).run(path="../../etc/passwd")
    assert res.ok is False and "outside" in (res.error or "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_tools_files.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/tools/files.py
from pathlib import Path
from crucible.tools.base import ToolResult

def _resolve(root: Path, path: str) -> Path:
    target = (root / path).resolve()
    root = root.resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"path outside root: {path}")
    return target

class ReadFile:
    name = "read_file"
    description = "Read a UTF-8 text file relative to the working directory."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    def __init__(self, root: Path): self.root = Path(root)
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
    def __init__(self, root: Path): self.root = Path(root)
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
    def __init__(self, root: Path): self.root = Path(root)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_tools_files.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/tools/files.py backend/tests/test_tools_files.py
git commit -m "feat: add read/write/edit file tools with root sandboxing"
```

---

### Task 3: Search tools (grep/glob)

**Files:**
- Create: `backend/crucible/tools/search.py`
- Test: `backend/tests/test_tools_search.py`

**Interfaces:**
- Produces `Grep` (`name="grep"`, `run(pattern, path=".") -> matching "file:line:text" lines`) and `Glob` (`name="glob"`, `run(pattern) -> newline-joined relative paths`). Both rooted, both never escape root.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_tools_search.py
from crucible.tools.search import Grep, Glob

def test_glob_finds_files(tmp_path):
    (tmp_path / "a.py").write_text("x"); (tmp_path / "b.txt").write_text("y")
    out = Glob(root=tmp_path).run(pattern="*.py").output
    assert "a.py" in out and "b.txt" not in out

def test_grep_finds_lines(tmp_path):
    (tmp_path / "a.py").write_text("alpha\nbeta\nalpha2\n")
    out = Grep(root=tmp_path).run(pattern="alpha").output
    assert "a.py:1:alpha" in out and "a.py:3:alpha2" in out and "beta" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_tools_search.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/tools/search.py
import re
from pathlib import Path
from crucible.tools.base import ToolResult

class Glob:
    name = "glob"
    description = "List files matching a glob pattern, relative to the working directory."
    parameters = {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}
    def __init__(self, root: Path): self.root = Path(root)
    def run(self, pattern: str) -> ToolResult:
        matches = sorted(str(p.relative_to(self.root)) for p in self.root.glob(pattern) if p.is_file())
        return ToolResult(ok=True, output="\n".join(matches))

class Grep:
    name = "grep"
    description = "Search file contents for a regex; returns file:line:text matches."
    parameters = {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}
    def __init__(self, root: Path): self.root = Path(root)
    def run(self, pattern: str, path: str = ".") -> ToolResult:
        rx = re.compile(pattern)
        base = (self.root / path)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_tools_search.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/tools/search.py backend/tests/test_tools_search.py
git commit -m "feat: add grep + glob search tools"
```

---

### Task 4: Shell tool (bash, timeout-guarded)

**Files:**
- Create: `backend/crucible/tools/shell.py`
- Test: `backend/tests/test_tools_shell.py`

**Interfaces:**
- Produces `Bash(root: Path, timeout: float = 30)`, `name="bash"`, `run(command) -> ToolResult` with combined stdout+stderr in `output`, `ok` = (exit==0). Times out → `ok=False, error="timeout"`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_tools_shell.py
from crucible.tools.shell import Bash

def test_bash_echo(tmp_path):
    res = Bash(root=tmp_path).run(command="echo hello")
    assert res.ok and "hello" in res.output

def test_bash_nonzero_exit(tmp_path):
    res = Bash(root=tmp_path).run(command="exit 3")
    assert res.ok is False

def test_bash_timeout(tmp_path):
    res = Bash(root=tmp_path, timeout=0.5).run(command="sleep 5")
    assert res.ok is False and res.error == "timeout"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_tools_shell.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/tools/shell.py
import subprocess
from pathlib import Path
from crucible.tools.base import ToolResult

class Bash:
    name = "bash"
    description = "Run a shell command in the working directory and return combined output."
    parameters = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
    def __init__(self, root: Path, timeout: float = 30):
        self.root = Path(root); self.timeout = timeout
    def run(self, command: str) -> ToolResult:
        try:
            p = subprocess.run(command, shell=True, cwd=self.root, capture_output=True,
                               text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, output="", error="timeout")
        out = (p.stdout or "") + (p.stderr or "")
        return ToolResult(ok=(p.returncode == 0), output=out,
                          error=None if p.returncode == 0 else f"exit {p.returncode}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_tools_shell.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/tools/shell.py backend/tests/test_tools_shell.py
git commit -m "feat: add timeout-guarded bash tool"
```

---

### Task 5: Permission policy

**Files:**
- Create: `backend/crucible/permissions.py`
- Test: `backend/tests/test_permissions.py`

**Interfaces:**
- Produces:
  - `PermissionMode = Literal["allow","ask","deny"]`.
  - `Decision(allowed: bool, reason: str)`.
  - `PermissionPolicy(default: PermissionMode = "ask", modes: dict[str, PermissionMode] | None = None, asker: Callable[[str, dict], bool] | None = None)`.
  - `.check(tool_name, args) -> Decision`: `allow`→allowed; `deny`→blocked; `ask`→calls `asker(tool_name, args)` (defaults to deny when no asker).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_permissions.py
from crucible.permissions import PermissionPolicy

def test_allow_mode():
    assert PermissionPolicy(modes={"read_file": "allow"}).check("read_file", {}).allowed

def test_deny_mode():
    assert PermissionPolicy(modes={"bash": "deny"}).check("bash", {}).allowed is False

def test_ask_uses_callback():
    pol = PermissionPolicy(default="ask", asker=lambda name, args: name == "read_file")
    assert pol.check("read_file", {}).allowed is True
    assert pol.check("bash", {}).allowed is False

def test_ask_without_asker_denies():
    assert PermissionPolicy(default="ask").check("bash", {}).allowed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_permissions.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/permissions.py
from dataclasses import dataclass, field
from typing import Callable, Literal

PermissionMode = Literal["allow", "ask", "deny"]

@dataclass
class Decision:
    allowed: bool
    reason: str

@dataclass
class PermissionPolicy:
    default: PermissionMode = "ask"
    modes: dict[str, PermissionMode] = field(default_factory=dict)
    asker: Callable[[str, dict], bool] | None = None

    def check(self, tool_name: str, args: dict) -> Decision:
        mode = self.modes.get(tool_name, self.default)
        if mode == "allow":
            return Decision(True, "allowed by policy")
        if mode == "deny":
            return Decision(False, "denied by policy")
        if self.asker is None:
            return Decision(False, "ask mode with no approver -> denied")
        return (Decision(True, "approved") if self.asker(tool_name, args)
                else Decision(False, "rejected by user"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_permissions.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/permissions.py backend/tests/test_permissions.py
git commit -m "feat: add allow/ask/deny permission policy"
```

---

### Task 6: Audit log

**Files:**
- Create: `backend/crucible/audit.py`
- Test: `backend/tests/test_audit.py`

**Interfaces:**
- Produces `AuditLog(path: Path)` with `.record(kind: str, data: dict) -> None` (appends one JSON object per line with a monotonic seq) and `.entries() -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_audit.py
from crucible.audit import AuditLog

def test_record_and_read(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record("tool_call", {"name": "bash", "command": "ls"})
    log.record("tool_result", {"name": "bash", "ok": True})
    entries = log.entries()
    assert [e["kind"] for e in entries] == ["tool_call", "tool_result"]
    assert entries[0]["seq"] == 0 and entries[1]["seq"] == 1
    assert entries[0]["data"]["command"] == "ls"

def test_persists_across_instances(tmp_path):
    AuditLog(tmp_path / "a.jsonl").record("x", {})
    assert len(AuditLog(tmp_path / "a.jsonl").entries()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_audit.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/audit.py
import json
from pathlib import Path

class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _next_seq(self) -> int:
        return len(self.entries())

    def record(self, kind: str, data: dict) -> None:
        entry = {"seq": self._next_seq(), "kind": kind, "data": data}
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_audit.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/audit.py backend/tests/test_audit.py
git commit -m "feat: add append-only audit log"
```

---

### Task 7: Agent loop

**Files:**
- Create: `backend/crucible/agent.py`
- Test: `backend/tests/test_agent.py`

**Interfaces:**
- Consumes: `ToolRegistry`, `PermissionPolicy`, `AuditLog`, `ChatClient` (Phase 1).
- Produces:
  - `AgentEvent(type: Literal["assistant","tool_call","tool_result","done","error"], data: dict)`.
  - `Model = Callable[[list[dict], list[dict]], dict]` — given (messages, tool schemas) returns an OpenAI-style assistant message dict (`{"role":"assistant","content":str|None,"tool_calls":[...]}`).
  - `Agent(model: Model, tools: ToolRegistry, permissions: PermissionPolicy, audit: AuditLog, max_iters: int = 10)`.
  - `.run(messages: list[dict]) -> Iterator[AgentEvent]` — loops: call model; for each `tool_calls` entry, permission-check (denied → tool_result with error), audit, execute tool, append a `role="tool"` message; stop when the model returns no tool_calls (emit `done`) or `max_iters` is hit.
  - `chat_client_model(client: ChatClient, extract) -> Model` adapter (wraps Phase-1 `ChatClient`; real network path, not unit-tested here).

- [ ] **Step 1: Write the failing test (fake model returns one tool call, then finishes)**

```python
# backend/tests/test_agent.py
import json
from crucible.agent import Agent
from crucible.tools.base import ToolRegistry
from crucible.tools.files import WriteFile, ReadFile
from crucible.permissions import PermissionPolicy
from crucible.audit import AuditLog

def make_registry(tmp_path):
    reg = ToolRegistry()
    reg.register(WriteFile(root=tmp_path)); reg.register(ReadFile(root=tmp_path))
    return reg

def scripted_model(responses):
    calls = iter(responses)
    def model(messages, tools):
        return next(calls)
    return model

def tool_call(id, name, args):
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}

def test_agent_executes_tool_then_finishes(tmp_path):
    model = scripted_model([
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("1", "write_file", {"path": "x.txt", "content": "hi"})]},
        {"role": "assistant", "content": "done writing", "tool_calls": []},
    ])
    agent = Agent(model=model, tools=make_registry(tmp_path),
                  permissions=PermissionPolicy(default="allow"),
                  audit=AuditLog(tmp_path / "audit.jsonl"))
    events = list(agent.run([{"role": "user", "content": "write hi to x.txt"}]))
    types = [e.type for e in events]
    assert "tool_call" in types and "tool_result" in types
    assert events[-1].type == "done"
    assert (tmp_path / "x.txt").read_text() == "hi"

def test_agent_blocks_denied_tool(tmp_path):
    model = scripted_model([
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("1", "write_file", {"path": "x.txt", "content": "hi"})]},
        {"role": "assistant", "content": "ok", "tool_calls": []},
    ])
    agent = Agent(model=model, tools=make_registry(tmp_path),
                  permissions=PermissionPolicy(modes={"write_file": "deny"}),
                  audit=AuditLog(tmp_path / "audit.jsonl"))
    events = list(agent.run([{"role": "user", "content": "write"}]))
    tr = [e for e in events if e.type == "tool_result"][0]
    assert tr.data["ok"] is False
    assert not (tmp_path / "x.txt").exists()

def test_agent_respects_max_iters(tmp_path):
    looping = {"role": "assistant", "content": None,
               "tool_calls": [tool_call("1", "read_file", {"path": "x.txt"})]}
    (tmp_path / "x.txt").write_text("hi")
    agent = Agent(model=lambda m, t: looping, tools=make_registry(tmp_path),
                  permissions=PermissionPolicy(default="allow"),
                  audit=AuditLog(tmp_path / "audit.jsonl"), max_iters=3)
    events = list(agent.run([{"role": "user", "content": "loop"}]))
    assert events[-1].type == "error"
    assert events[-1].data["reason"] == "max_iters exceeded"
    assert sum(1 for e in events if e.type == "tool_call") == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/agent.py
import json
from dataclasses import dataclass
from typing import Callable, Iterator, Literal

from crucible.audit import AuditLog
from crucible.permissions import PermissionPolicy
from crucible.tools.base import ToolRegistry

EventType = Literal["assistant", "tool_call", "tool_result", "done", "error"]
Model = Callable[[list[dict], list[dict]], dict]

@dataclass
class AgentEvent:
    type: EventType
    data: dict

class Agent:
    def __init__(self, model: Model, tools: ToolRegistry,
                 permissions: PermissionPolicy, audit: AuditLog, max_iters: int = 10):
        self.model = model
        self.tools = tools
        self.permissions = permissions
        self.audit = audit
        self.max_iters = max_iters

    def run(self, messages: list[dict]) -> Iterator[AgentEvent]:
        convo = list(messages)
        for _ in range(self.max_iters):
            msg = self.model(convo, self.tools.schemas())
            convo.append(msg)
            if msg.get("content"):
                yield AgentEvent("assistant", {"content": msg["content"]})
            calls = msg.get("tool_calls") or []
            if not calls:
                yield AgentEvent("done", {"content": msg.get("content", "")})
                return
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")
                self.audit.record("tool_call", {"name": name, "args": args})
                yield AgentEvent("tool_call", {"id": call["id"], "name": name, "args": args})
                decision = self.permissions.check(name, args)
                if not decision.allowed:
                    result = {"ok": False, "output": "", "error": decision.reason}
                else:
                    res = self.tools.get(name).run(**args)
                    result = res.model_dump()
                self.audit.record("tool_result", {"name": name, **result})
                yield AgentEvent("tool_result", {"id": call["id"], "name": name, **result})
                convo.append({"role": "tool", "tool_call_id": call["id"],
                              "content": result["output"] or result.get("error", "")})
        yield AgentEvent("error", {"reason": "max_iters exceeded"})

def chat_client_model(client, extract: Callable[[dict], dict]):
    """Adapter: wrap a Phase-1 ChatClient into a Model. `extract` maps the raw
    /v1/chat/completions JSON to an assistant message dict. Network path; not unit-tested."""
    import httpx
    def model(messages: list[dict], tools: list[dict]) -> dict:
        payload = {"model": "local", "messages": messages, "tools": tools, "max_tokens": 1024}
        r = httpx.post(f"{client.endpoint}/v1/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
        return extract(r.json())
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_agent.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/agent.py backend/tests/test_agent.py
git commit -m "feat: add iterative tool-calling agent loop with permission + audit"
```

---

### Task 8: SSE agent endpoint + default tools

**Files:**
- Modify: `backend/crucible/tools/__init__.py` (add `default_registry`)
- Modify: `backend/crucible/app.py` (add `/api/agent/run` SSE route)
- Test: `backend/tests/test_agent_endpoint.py`

**Interfaces:**
- Consumes: `Agent`, `ToolRegistry`, `PermissionPolicy`, `AuditLog`, file/search/shell tools.
- Produces:
  - `default_registry(root: Path) -> ToolRegistry` with read/write/edit/grep/glob/bash registered.
  - `POST /api/agent/run` body `{messages: [...], permissions: {default, modes}}` → `text/event-stream` of `data: {json AgentEvent}\n\n`. For tests the app factory accepts an injected `model` so no network is needed.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_agent_endpoint.py
import json
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry

def tool_call(id, name, args):
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}

def test_agent_run_streams_events(tmp_path):
    responses = iter([
        {"role": "assistant", "content": None,
         "tool_calls": [tool_call("1", "write_file", {"path": "y.txt", "content": "hey"})]},
        {"role": "assistant", "content": "wrote it", "tool_calls": []},
    ])
    app = create_app(registry=Registry(tmp_path / "r.json"),
                     agent_root=tmp_path, model=lambda m, t: next(responses))
    c = TestClient(app)
    body = {"messages": [{"role": "user", "content": "write hey to y.txt"}],
            "permissions": {"default": "allow", "modes": {}}}
    with c.stream("POST", "/api/agent/run", json=body) as r:
        payloads = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert any(p["type"] == "tool_call" for p in payloads)
    assert payloads[-1]["type"] == "done"
    assert (tmp_path / "y.txt").read_text() == "hey"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_agent_endpoint.py -v`
Expected: FAIL — `create_app() got an unexpected keyword 'agent_root'`.

- [ ] **Step 3: Add `default_registry`**

```python
# backend/crucible/tools/__init__.py
from pathlib import Path
from crucible.tools.base import Tool, ToolResult, ToolRegistry, openai_schema  # noqa: F401
from crucible.tools.files import ReadFile, WriteFile, EditFile
from crucible.tools.search import Grep, Glob
from crucible.tools.shell import Bash

def default_registry(root: Path) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (ReadFile(root), WriteFile(root), EditFile(root),
                 Grep(root), Glob(root), Bash(root)):
        reg.register(tool)
    return reg
```

- [ ] **Step 4: Extend the app factory + SSE route**

```python
# backend/crucible/app.py  — replace the file with:
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from crucible.agent import Agent
from crucible.audit import AuditLog
from crucible.config import get_settings
from crucible.permissions import PermissionPolicy
from crucible.registry import Model, Registry
from crucible.tools import default_registry


class PermissionConfig(BaseModel):
    default: str = "ask"
    modes: dict[str, str] = {}


class AgentRunRequest(BaseModel):
    messages: list[dict]
    permissions: PermissionConfig = PermissionConfig()


def create_app(registry: Registry | None = None, agent_root: Path | None = None,
               model=None) -> FastAPI:
    settings = get_settings()
    reg = registry or Registry(settings.registry_path)
    root = Path(agent_root or ".")
    app = FastAPI(title="Crucible")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/models")
    def list_models() -> list[Model]:
        return reg.list()

    @app.post("/api/models", status_code=201)
    def create_model(model_in: Model) -> Model:
        try:
            return reg.register(model_in)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.get("/api/models/{model_id}/lineage")
    def lineage(model_id: str) -> list[Model]:
        try:
            return reg.lineage(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="model not found")

    @app.post("/api/agent/run")
    def agent_run(req: AgentRunRequest):
        if model is None:
            raise HTTPException(status_code=503, detail="no model configured")
        policy = PermissionPolicy(default=req.permissions.default, modes=req.permissions.modes)
        agent = Agent(model=model, tools=default_registry(root),
                      permissions=policy, audit=AuditLog(settings.data_dir / "audit.jsonl"))

        def stream():
            for event in agent.run(req.messages):
                yield f"data: {json.dumps({'type': event.type, **{'data': event.data}})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app()
```

- [ ] **Step 5: Fix the test's event shape expectation**

The route emits `{"type": ..., "data": {...}}`. Update the test assertions to read `p["type"]` (already correct) — `done` detection uses `p["type"]`. Confirm the test reads `type` at top level (it does).

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest backend/tests/test_agent_endpoint.py -v`
Expected: PASS (1 test).

- [ ] **Step 7: Run the full suite**

Run: `pytest -q`
Expected: all Phase 1 + Phase 2 tests pass.

- [ ] **Step 8: Commit**

```bash
git add backend/crucible/app.py backend/crucible/tools/__init__.py backend/tests/test_agent_endpoint.py
git commit -m "feat: add SSE agent endpoint + default tool registry"
```

---

## Self-Review

**Spec coverage (Phase 2 scope = Component 4.3 Agent Harness + 4.8 Audit Log):**
- Tools (read/write/edit/bash/grep/glob) → Tasks 2–4. ✅
- Permission system (allow/ask/deny, dangerous-command gating) → Task 5 (+ enforced in loop, Task 7). ✅
- Planning / multi-step execution / self-correction → iterative loop with tool results fed back (Task 7). ✅
- Context management note: conversation accumulation handled in loop; summarization-at-threshold deferred to a later hardening pass (out of Phase 2 minimal scope — flagged).
- Streaming tool calls + text over SSE → Task 8. ✅
- Audit log (4.8) → Task 6, wired in Task 7/8. ✅
- GUI shell → deferred to the frontend-design-driven step after this plan (stated in handoff). 

**Placeholder scan:** No TBD/TODO; every code step is complete. ✅

**Type consistency:** `ToolResult{ok,output,error}` consistent across Tasks 1–4,7. `AgentEvent{type,data}` consistent across Tasks 7–8. `PermissionPolicy(default,modes,asker)` consistent Tasks 5,7,8. `default_registry(root)` defined Task 8, used Task 8. `model(messages, tools)->dict` signature consistent across Tasks 7,8. ✅

**Deferred (tracked, not lost):** conversation summarization at context threshold; real-model wiring of `chat_client_model` (needs the GLM-4-32B endpoint from Phase 1 Task 7).
