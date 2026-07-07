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

# "Action: none" (and friends) means the model has NO tool to call — treat it as a final answer,
# not a phantom tool. Weak/uncensored models emit these constantly when a plain reply would do.
_SENTINEL_TOOLS = {"none", "null", "n/a", "na", "nil", "nothing", "no_tool", "no tool",
                   "notool", "finish", "done", "final", "final_answer", "answer", "stop", ""}

# Cap tool output before it's fed back into the model. A `bash: ls -R` in a big repo can return
# 100KB+ (whole node_modules), which overflows the model's context and the upstream returns a 400.
_OBSERVATION_LIMIT = 6000


def strip_scaffold(text: str) -> str:
    """Drop ReAct scaffolding lines (Action/Observation) and the 'Thought:' prefix, leaving the
    model's actual prose — used when a sentinel 'Action: none' should just be the final answer."""
    keep = []
    for line in (text or "").splitlines():
        low = line.strip().lower()
        if low.startswith(("action:", "action input:", "observation:")):
            continue
        s = line.strip()
        if low.startswith("thought:"):
            s = s[len("thought:"):].strip()
        keep.append(s)
    return "\n".join(k for k in keep if k).strip()


def truncate_observation(text, limit: int = _OBSERVATION_LIMIT) -> str:
    """Bound a tool observation so a huge output can't overflow the model's context (which upstreams
    reject with a 400). Keeps the head and tail with a clear elision marker."""
    t = "" if text is None else str(text)
    if len(t) <= limit:
        return t
    head, tail = t[: limit - 400], t[-300:]
    return f"{head}\n… [truncated {len(t) - limit + 700} chars of tool output] …\n{tail}"


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


def hybrid_preamble(tool_schemas: list[dict]) -> str:
    """A light preamble that keeps native function-calling for models that support it, but
    gives models that DON'T a text escape hatch — so tools work either way. Includes a
    one-shot example so weak/uncensored models follow the format reliably."""
    tools = []
    for t in tool_schemas:
        fn = t.get("function", t)
        params = fn.get("parameters", {}).get("properties", {})
        tools.append(f"- {fn.get('name', '?')}({', '.join(params.keys())}): {fn.get('description', '')}")
    return (
        "You have tools. If your runtime supports function/tool calls, just call them.\n"
        "If it does NOT, use this exact text format instead:\n"
        "Thought: <reasoning>\n"
        "Action: <tool name>\n"
        'Action Input: {"arg": "value"}\n'
        "then stop and wait for the Observation. When finished, reply:\n"
        "Final Answer: <answer>\n\n"
        "Example:\n"
        "Thought: I should read the file.\n"
        "Action: read_file\n"
        'Action Input: {"path": "notes.txt"}\n\n'
        "Available tools:\n" + "\n".join(tools)
    )


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
        tool = act.group(1).strip().strip("`\"'*")
        if tool.lower() in _SENTINEL_TOOLS:            # "Action: none" -> the model has no tool to run
            return {"kind": "final", "text": strip_scaffold(text) or text.strip()}
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


def _unwrap_args(args: dict) -> dict:
    """Local models often WRAP the real tool arguments under a single 'input'/'arguments'/'parameters'
    key — as a nested object or a JSON string — instead of passing them flat. Unwrap that so the tool
    gets {path, content} rather than {input: '{"path":...}'}. Only unwraps when the inner value is (or
    parses to) a dict, so a tool that legitimately takes a string 'input' is left untouched."""
    if isinstance(args, dict) and len(args) == 1:
        key = next(iter(args))
        if key in ("input", "arguments", "args", "parameters"):
            val = args[key]
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    return args
            if isinstance(val, dict):
                return val
    return args


def _execute(tools: ToolRegistry, audit: AuditLog, name: str, args: dict) -> dict:
    """Run one tool (permission already decided). Returns an OpenAI-style result dict."""
    if name not in {t.name for t in tools.all()}:
        return {"ok": False, "output": "", "error": f"no such tool '{name}'"}
    try:
        res = tools.get(name).run(**_unwrap_args(args)).model_dump()
    except TypeError as e:
        # a bad-shaped call shouldn't crash the loop — feed it back so the model can retry
        return {"ok": False, "output": "", "error": f"bad arguments for '{name}': {e}"}
    audit.record("tool_result", {"name": name, **res})
    return res


def _dispatch_tool(tools, permissions, audit, approver, call_id, name, args):
    """Authorize + run a tool, yielding a permission_request (when mode='ask' and an approver
    exists) then the tool_result. Returns the observation string (via generator return)."""
    valid = [t.name for t in tools.all()]
    name = coerce_tool_name(name, valid)              # snap hallucinated names to the nearest real one
    if name not in valid:
        # a phantom tool (e.g. the model invented 'greet'): don't prompt the operator to approve
        # something that can't run — feed a helpful observation so the model self-corrects.
        obs = (f"no such tool '{name}'. Available tools: {', '.join(valid)}. If you don't need a "
               "tool, reply with 'Final Answer: <your reply>'.")
        yield AgentEvent("tool_result", {"id": call_id, "name": name, "ok": False, "output": "", "error": obs})
        return obs
    mode = permissions.mode_for(name)
    if mode == "deny":
        result = {"ok": False, "output": "", "error": "denied by policy"}
    elif mode == "ask":
        if approver is None:
            result = {"ok": False, "output": "", "error": "ask mode with no approver -> denied"}
        else:
            yield AgentEvent("permission_request", {"id": call_id, "name": name, "args": args})
            if approver(call_id, name, args):        # blocks until the operator decides
                result = _execute(tools, audit, name, args)
            else:
                result = {"ok": False, "output": "", "error": "rejected by operator"}
    else:  # allow
        result = _execute(tools, audit, name, args)
    yield AgentEvent("tool_result", {"id": call_id, "name": name, **result})
    # truncate before it re-enters the conversation so a huge output can't overflow the context
    return truncate_observation(result.get("output") or result.get("error", ""))


def hybrid_run(model, tools: ToolRegistry, messages: list[dict],
               permissions: PermissionPolicy, audit: AuditLog,
               max_iters: int = 10, approver=None) -> Iterator[AgentEvent]:
    """Universal tool loop: accepts BOTH native OpenAI tool_calls AND text ReAct actions in
    the same loop, so tools work whether or not the model was designed for them. Native
    models keep function-calling; models without it use the text format from the preamble.
    This is the default forge loop — no toggle needed to give a 'heretic' model tools.
    `approver(call_id, name, args)->bool` handles 'ask' tools (blocks until the operator decides)."""
    convo: list[dict] = [{"role": "system", "content": hybrid_preamble(tools.schemas())}, *messages]
    streamer = getattr(model, "stream", None)
    use_stream = callable(streamer)
    rid = 0
    for _ in range(max_iters):
        streamed = False
        try:
            if use_stream:
                msg: dict = {"role": "assistant", "content": ""}
                for kind, payload in streamer(convo, tools.schemas()):
                    if kind == "token":
                        streamed = True
                        yield AgentEvent("assistant_delta", {"delta": payload})
                    else:
                        msg = payload
            else:
                msg = model(convo, tools.schemas())      # offer native tools every turn
        except Exception as exc:
            yield AgentEvent("error", {"reason": f"model call failed: {exc}"})
            return
        convo.append(msg)
        text = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        if calls:                                        # --- native tool-call path ---
            if text or streamed:
                yield AgentEvent("assistant", {"content": text, "streamed": streamed})
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"].get("arguments") or "{}")
                yield AgentEvent("tool_call", {"id": call["id"], "name": name, "args": args})
                obs = yield from _dispatch_tool(tools, permissions, audit, approver,
                                                call["id"], name, args)
                convo.append({"role": "tool", "tool_call_id": call["id"], "content": obs})
            continue
        step = parse_react(text)
        if step["kind"] == "action":                     # --- text ReAct path ---
            if streamed:                                 # finalize the streamed scaffolding turn
                yield AgentEvent("assistant", {"content": text, "streamed": True})
            rid += 1
            cid = f"react-{rid}"
            name, args = step["tool"], step["input"]
            yield AgentEvent("tool_call", {"id": cid, "name": name, "args": args})
            obs = yield from _dispatch_tool(tools, permissions, audit, approver, cid, name, args)
            convo.append({"role": "user", "content": f"Observation: {obs}"})
            continue
        if streamed or text:                             # --- final (clean text replaces raw) ---
            yield AgentEvent("assistant", {"content": step["text"], "streamed": streamed})
        yield AgentEvent("done", {"content": step.get("text", "")})
        return
    yield AgentEvent("error", {"reason": "max_iters exceeded"})


def coerce_tool_name(name: str, valid: list[str]) -> str:
    """Snap a hallucinated tool name to the closest valid one (weak models often paraphrase,
    e.g. 'list_files' -> 'list_dir'). Exact/case-insensitive first, then token overlap, then
    substring; falls back to the original if nothing is close."""
    if not valid:
        return name
    if name in valid:
        return name
    low = name.lower()
    by_lower = {v.lower(): v for v in valid}
    if low in by_lower:
        return by_lower[low]
    ntok = set(re.split(r"[_\s]+", low))
    best, best_score = name, 0.0
    for v in valid:
        vtok = set(re.split(r"[_\s]+", v.lower()))
        overlap = len(ntok & vtok)
        score = overlap / max(1, len(ntok | vtok))
        if low in v.lower() or v.lower() in low:
            score = max(score, 0.5)
        if score > best_score:
            best, best_score = v, score
    return best if best_score >= 0.3 else name


def react_to_openai_tool_call(step: dict, idx: int = 0) -> dict:
    """Convert a parsed ReAct action into an OpenAI tool_call — so a model that can only do
    text ReAct can still answer a client (OpenCode) with NATIVE function-calling structure."""
    return {"id": f"call_{idx}", "type": "function",
            "function": {"name": step["tool"], "arguments": json.dumps(step["input"])}}


def react_run(model, tools: ToolRegistry, messages: list[dict],
              permissions: PermissionPolicy, audit: AuditLog,
              max_iters: int = 8, approver=None) -> Iterator[AgentEvent]:
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
        step_id += 1
        cid = f"react-{step_id}"
        name, args = step["tool"], step["input"]
        yield AgentEvent("tool_call", {"id": cid, "name": name, "args": args})
        observation = yield from _dispatch_tool(tools, permissions, audit, approver, cid, name, args)
        convo.append({"role": "assistant", "content": text})
        convo.append({"role": "user", "content": f"Observation: {observation}"})
    yield AgentEvent("error", {"reason": "max_iters exceeded"})
