"""crucible — a local agentic coding harness CLI (Claude-Code-equivalent, term: crucible).

Talks to any OpenAI-compatible endpoint (local llama-server, the Crucible torch server,
or a REMOTE Windows node). Runs the agent tool-loop with allow/ask/deny permissions.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import httpx

from crucible.agent import Agent
from crucible.audit import AuditLog
from crucible.permissions import PermissionPolicy
from crucible.tools import default_registry

BANNER = "crucible — local agentic coding harness"
HELP = """commands:
  /help              show this help
  /endpoint <url>    point at a chat endpoint (e.g. a remote Windows node)
  /recipe <layers> <rank>   set the served runtime recipe (control API)
  /recipe clear      clear the served recipe
  /diagnose <id>     run censorship diagnosis on model <id>
  /perm <mode>       tool permission: allow | ask | deny
  /tools             list available tools
  /clear             reset the conversation
  /exit              quit
(any other text is sent to the agent)"""


def parse_chat_response(data: dict) -> dict:
    msg = data["choices"][0]["message"]
    return {"role": "assistant", "content": msg.get("content"), "tool_calls": msg.get("tool_calls") or []}


def make_model(chat_url: str):
    def model(messages, tools):
        payload = {"model": "crucible", "messages": messages, "max_tokens": 1024}
        if tools:
            payload["tools"] = tools
        r = httpx.post(chat_url, json=payload, timeout=600)
        r.raise_for_status()
        return parse_chat_response(r.json())
    return model


def _ask(name: str, args: dict) -> bool:
    ans = input(f"  ↳ allow tool '{name}' {json.dumps(args)[:80]}? [y/N] ").strip().lower()
    return ans in ("y", "yes")


def _print_event(ev) -> str:
    if ev.type == "assistant" and ev.data.get("content"):
        print(f"\n{ev.data['content']}")
        return ev.data["content"]
    if ev.type == "tool_call":
        print(f"  · {ev.data['name']}({json.dumps(ev.data['args'])[:80]})")
    elif ev.type == "tool_result":
        ok = ev.data.get("ok")
        out = (ev.data.get("output") or ev.data.get("error") or "")[:200]
        print(f"    {'✓' if ok else '✗'} {out}")
    elif ev.type == "error":
        print(f"  ! {ev.data.get('reason')}")
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crucible", description=BANNER)
    ap.add_argument("--endpoint", default=os.environ.get("CRUCIBLE_ENDPOINT", "http://127.0.0.1:8400/v1"))
    ap.add_argument("--control", default=os.environ.get("CRUCIBLE_CONTROL", "http://127.0.0.1:8400"))
    ap.add_argument("--perm", default="ask", choices=["allow", "ask", "deny"])
    ap.add_argument("prompt", nargs="*")
    a = ap.parse_args(argv)

    state = {"chat": a.endpoint.rstrip("/") + "/chat/completions", "perm": a.perm, "convo": []}
    audit = AuditLog(Path.home() / ".crucible" / "cli-audit.jsonl")

    def run_turn(text: str) -> None:
        state["convo"].append({"role": "user", "content": text})
        agent = Agent(make_model(state["chat"]), default_registry(Path.cwd()),
                      PermissionPolicy(default=state["perm"], asker=_ask), audit)
        final = ""
        for ev in agent.run(state["convo"]):
            got = _print_event(ev)
            if got:
                final = got
        if final:
            state["convo"].append({"role": "assistant", "content": final})

    def command(line: str) -> bool:
        parts = line.split()
        cmd = parts[0]
        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            print(HELP)
        elif cmd == "/endpoint" and len(parts) > 1:
            state["chat"] = parts[1].rstrip("/") + "/chat/completions"
            print(f"  endpoint -> {parts[1]}")
        elif cmd == "/perm" and len(parts) > 1:
            state["perm"] = parts[1]
            print(f"  permission -> {parts[1]}")
        elif cmd == "/tools":
            print("  " + ", ".join(t.name for t in default_registry(Path.cwd()).all()))
        elif cmd == "/clear":
            state["convo"] = []
            print("  conversation reset")
        elif cmd == "/recipe" and len(parts) > 1 and parts[1] == "clear":
            httpx.delete(f"{a.control}/api/inference/recipe")
            print("  recipe cleared")
        elif cmd == "/recipe" and len(parts) >= 3:
            layers = [int(x) for x in parts[1].split(",")]
            rank = int(parts[2])
            r = httpx.post(f"{a.control}/api/inference/recipe",
                           json={"base_id": "served", "layers": layers, "rank": rank, "coefficient": 1.0})
            print(f"  recipe -> {r.json().get('active')}")
        elif cmd == "/diagnose" and len(parts) > 1:
            r = httpx.post(f"{a.control}/api/abliteration/diagnose", json={"base_id": parts[1]}, timeout=300)
            d = r.json()
            print(f"  best_layer {d.get('best_layer')} surgical {d.get('surgical')}")
        else:
            print("  unknown command — /help")
        return True

    if a.prompt:
        run_turn(" ".join(a.prompt))
        return 0
    print(BANNER)
    print(f"endpoint: {a.endpoint}  ·  /help for commands")
    while True:
        try:
            line = input("\ncrucible> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not command(line):
                break
        else:
            run_turn(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
