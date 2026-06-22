from __future__ import annotations
"""crucible — a local agentic coding harness CLI (Claude-Code-equivalent, term: crucible).

Talks to any OpenAI-compatible endpoint (local llama-server, the Crucible torch server,
or a REMOTE Windows node). Agent tool-loop with allow/ask/deny permissions, a settings
file, and persistent sessions.
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
from crucible.mcp import load_mcp

BANNER = "crucible — local agentic coding harness"
DEFAULTS = {"endpoint": "http://127.0.0.1:8400/v1", "control": "http://127.0.0.1:8400", "perm": "ask", "token": ""}
HELP = """commands:
  /help                 this help
  /endpoint <url>       point at a chat endpoint (e.g. a remote Windows node)
  /perm <allow|ask|deny>  tool permission
  /tools                list tools
  /save <name>          save this session
  /load <name>          load a session
  /sessions             list saved sessions
  /config               show settings   ·   /config save  persists current settings
  /recipe <layers> <rank> | /recipe clear   set/clear the served runtime recipe
  /diagnose <id>        run censorship diagnosis
  /clear                reset conversation   ·   /exit  quit
(any other text → the agent)"""


def home() -> Path:
    h = Path(os.environ.get("CRUCIBLE_HOME", Path.home() / ".crucible"))
    h.mkdir(parents=True, exist_ok=True)
    return h


def load_settings() -> dict:
    f = home() / "settings.json"
    out = dict(DEFAULTS)
    if f.exists():
        try:
            out.update(json.loads(f.read_text()))
        except (OSError, ValueError):
            pass
    return out


def save_settings(s: dict) -> None:
    (home() / "settings.json").write_text(json.dumps(s, indent=2))


def session_path(name: str) -> Path:
    d = home() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.json"


def save_session(name: str, convo: list[dict]) -> None:
    session_path(name).write_text(json.dumps(convo, indent=2))


def load_session(name: str) -> list[dict]:
    f = session_path(name)
    return json.loads(f.read_text()) if f.exists() else []


def list_sessions() -> list[str]:
    d = home() / "sessions"
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def parse_chat_response(data: dict) -> dict:
    msg = data["choices"][0]["message"]
    return {"role": "assistant", "content": msg.get("content"), "tool_calls": msg.get("tool_calls") or []}


def make_model(chat_url: str, token: str = ""):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    def model(messages, tools):
        payload = {"model": "crucible", "messages": messages, "max_tokens": 1024}
        if tools:
            payload["tools"] = tools
        r = httpx.post(chat_url, json=payload, headers=headers, timeout=600)
        r.raise_for_status()
        return parse_chat_response(r.json())
    return model


def _ask(name: str, args: dict) -> bool:
    return input(f"  ↳ allow '{name}' {json.dumps(args)[:80]}? [y/N] ").strip().lower() in ("y", "yes")


def _print_event(ev) -> str:
    if ev.type == "assistant" and ev.data.get("content"):
        print(f"\n{ev.data['content']}")
        return ev.data["content"]
    if ev.type == "tool_call":
        print(f"  · {ev.data['name']}({json.dumps(ev.data['args'])[:80]})")
    elif ev.type == "tool_result":
        print(f"    {'✓' if ev.data.get('ok') else '✗'} {(ev.data.get('output') or ev.data.get('error') or '')[:200]}")
    elif ev.type == "error":
        print(f"  ! {ev.data.get('reason')}")
    return ""


def main(argv=None) -> int:
    cfg = load_settings()
    ap = argparse.ArgumentParser(prog="crucible", description=BANNER)
    ap.add_argument("--endpoint", default=cfg["endpoint"])
    ap.add_argument("--control", default=cfg["control"])
    ap.add_argument("--perm", default=cfg["perm"], choices=["allow", "ask", "deny"])
    ap.add_argument("--token", default=cfg.get("token", ""))
    ap.add_argument("--session", default=None)
    ap.add_argument("prompt", nargs="*")
    a = ap.parse_args(argv)

    st = {"chat": a.endpoint.rstrip("/") + "/chat/completions", "endpoint": a.endpoint,
          "control": a.control, "perm": a.perm,
          "convo": load_session(a.session) if a.session else [], "session": a.session, "token": a.token}
    audit = AuditLog(home() / "cli-audit.jsonl")
    registry = default_registry(Path.cwd())
    mcp_clients, mcp_tools = load_mcp(cfg.get("mcp", {}))
    for _t in mcp_tools:
        registry.register(_t)

    def run_turn(text: str) -> None:
        st["convo"].append({"role": "user", "content": text})
        agent = Agent(make_model(st["chat"], st["token"]), registry,
                      PermissionPolicy(default=st["perm"], asker=_ask), audit)
        final = ""
        for ev in agent.run(st["convo"]):
            got = _print_event(ev)
            if got:
                final = got
        if final:
            st["convo"].append({"role": "assistant", "content": final})
        if st["session"]:
            save_session(st["session"], st["convo"])

    def command(line: str) -> bool:
        parts = line.split()
        cmd = parts[0]
        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            print(HELP)
        elif cmd == "/endpoint" and len(parts) > 1:
            st["chat"] = parts[1].rstrip("/") + "/chat/completions"
            st["endpoint"] = parts[1]
            print(f"  endpoint → {parts[1]}")
        elif cmd == "/perm" and len(parts) > 1:
            st["perm"] = parts[1]
            print(f"  permission → {parts[1]}")
        elif cmd == "/tools":
            print("  " + ", ".join(t.name for t in registry.all()))
        elif cmd == "/mcp":
            print(f"  {len(mcp_clients)} MCP server(s), {len(mcp_tools)} tool(s): " + ", ".join(t.name for t in mcp_tools))
        elif cmd == "/save" and len(parts) > 1:
            st["session"] = parts[1]
            save_session(parts[1], st["convo"])
            print(f"  saved session '{parts[1]}' ({len(st['convo'])} msgs)")
        elif cmd == "/load" and len(parts) > 1:
            st["session"] = parts[1]
            st["convo"] = load_session(parts[1])
            print(f"  loaded '{parts[1]}' ({len(st['convo'])} msgs)")
        elif cmd == "/sessions":
            print("  " + (", ".join(list_sessions()) or "(none)"))
        elif cmd == "/config" and len(parts) > 1 and parts[1] == "save":
            save_settings({"endpoint": st["endpoint"], "control": st["control"], "perm": st["perm"]})
            print(f"  settings saved → {home() / 'settings.json'}")
        elif cmd == "/config":
            print(f"  endpoint={st['endpoint']} control={st['control']} perm={st['perm']} session={st['session']}")
        elif cmd == "/clear":
            st["convo"] = []
            print("  conversation reset")
        elif cmd == "/recipe" and len(parts) > 1 and parts[1] == "clear":
            httpx.delete(f"{st['control']}/api/inference/recipe")
            print("  recipe cleared")
        elif cmd == "/recipe" and len(parts) >= 3:
            httpx.post(f"{st['control']}/api/inference/recipe",
                       json={"base_id": "served", "layers": [int(x) for x in parts[1].split(",")],
                             "rank": int(parts[2]), "coefficient": 1.0})
            print("  recipe set")
        elif cmd == "/diagnose" and len(parts) > 1:
            d = httpx.post(f"{st['control']}/api/abliteration/diagnose", json={"base_id": parts[1]}, timeout=300).json()
            print(f"  best_layer {d.get('best_layer')} surgical {d.get('surgical')}")
        else:
            print("  unknown command — /help")
        return True

    if a.prompt:
        run_turn(" ".join(a.prompt))
        return 0
    print(BANNER)
    print(f"endpoint: {st['endpoint']} · perm: {st['perm']}{' · session: ' + st['session'] if st['session'] else ''} · /help")
    while True:
        try:
            line = input(f"\ncrucible[{st['perm']}]> ").strip()
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
