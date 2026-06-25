from __future__ import annotations
# ReAct tool-calling fallback. Many open-weight chat models (most small GGUF builds) don't
# support native OpenAI function-calling, so the normal tool-loop never fires. ReAct gets
# tools working with ANY chat model: we describe the tools and a strict text format in the
# prompt, let the model emit "Action: <tool> / Action Input: <json>" (or "Final Answer:"),
# parse it, run the tool, feed back "Observation:", and repeat. Pure parsing/prompt-building
# is unit-tested; the loop reuses the same AgentEvent protocol as the native path.
import json
import re
from typing import Iterator

from crucible.agent import AgentEvent
from crucible.audit import AuditLog
from crucible.permissions import PermissionPolicy
from crucible.tools.base import ToolRegistry

_ACTION = re.compile(r"Action\s*:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)
_INPUT = re.compile(r"Action\s*Input\s*:\s*(\{.*?\}|\S.*?)\s*(?:\n\n|\nObservation|\nThought|$)",
                    re.IGNORECASE | re.DOTALL)
_FINAL = re.compile(r"Final\s*Answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


def react_preamble(tool_schemas: list[dict]) -> str:
    """Build the system preamble describing the tools and the ReAct response format."""
    lines = ["You can use tools. To use one, reply EXACTLY in this format:",
             "Thought: <your reasoning>",
             "Action: <one tool name from the list>",
             'Action Input: <a JSON object of arguments>',
             "",
             "After you see the Observation, continue. When you are done, reply:",
             "Final Answer: <your answer to the user>",
             "",
             "Available tools:"]
    for t in tool_schemas:
        fn = t.get("function", t)
        name = fn.get("name", "?")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        arglist = ", ".join(params.keys())
        lines.append(f"- {name}({arglist}): {desc}")
    return "\n".join(lines)


def parse_react(text: str) -> dict:
    """Parse a model turn into a ReAct step. Returns one of:
    {"kind": "final", "text": ...} or {"kind": "action", "tool": ..., "input": {...}}.
    A 'Final Answer' takes precedence; a bare reply with no Action is treated as final."""
    fin = _FINAL.search(text)
    act = _ACTION.search(text)
    # If both appear, honor whichever comes first in the text.
    if fin and (not act or fin.start() < act.start()):
        return {"kind": "final", "text": fin.group(1).strip()}
    if act:
        tool = act.group(1).strip().strip("`\"'")
        raw = ""
        m = _INPUT.search(text, act.end() - 1)
        if m:
            raw = m.group(1).strip()
        args: dict = {}
        if raw:
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                # tolerate key=value or quoted bare string
                args = {"input": raw.strip().strip("`\"'")}
        if not isinstance(args, dict):
            args = {"input": args}
        return {"kind": "action", "tool": tool, "input": args}
    return {"kind": "final", "text": text.strip()}


def react_run(model, tools: ToolRegistry, messages: list[dict],
              permissions: PermissionPolicy, audit: AuditLog,
              max_iters: int = 8) -> Iterator[AgentEvent]:
    """Run a ReAct tool loop against a plain chat model (no native function-calling)."""
    convo = [{"role": "system", "content": react_preamble(tools.schemas())}, *messages]
    step_id = 0
    for _ in range(max_iters):
        try:
            msg = model(convo, [])               # plain chat, no tools param
        except Exception as exc:
            yield AgentEvent("error", {"reason": f"model call failed: {exc}"})
            return
        text = msg.get("content") or ""
        step = parse_react(text)
        if step["kind"] == "final":
            if step["text"]:
                yield AgentEvent("assistant", {"content": step["text"], "streamed": False})
            yield AgentEvent("done", {"content": step["text"]})
            return
        # action
        step_id += 1
        cid = f"react-{step_id}"
        name, args = step["tool"], step["input"]
        yield AgentEvent("tool_call", {"id": cid, "name": name, "args": args})
        if name not in {t.name for t in tools.all()}:
            observation = f"error: no such tool '{name}'"
            yield AgentEvent("tool_result", {"id": cid, "name": name, "ok": False,
                                             "output": "", "error": observation})
        else:
            decision = permissions.check(name, args)
            if not decision.allowed:
                observation = f"denied: {decision.reason}"
                yield AgentEvent("tool_result", {"id": cid, "name": name, "ok": False,
                                                 "output": "", "error": decision.reason})
            else:
                res = tools.get(name).run(**args).model_dump()
                observation = res.get("output") or res.get("error") or ""
                audit.record("tool_result", {"name": name, **res})
                yield AgentEvent("tool_result", {"id": cid, "name": name, **res})
        convo.append({"role": "assistant", "content": text})
        convo.append({"role": "user", "content": f"Observation: {observation}"})
    yield AgentEvent("error", {"reason": "max_iters exceeded"})
