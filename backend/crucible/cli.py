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
  /sessions             list saved sessions (in this project)
  /where                show working dir, project root, session store + memory dir
  /config               show settings   ·   /config save  persists current settings
  /recipe <layers> <rank> | /recipe clear   set/clear the served runtime recipe
  /diagnose <id>        run censorship diagnosis
  /clear                reset conversation   ·   /exit  quit
(any other text → the agent)"""


# A project is the nearest ancestor of the CWD carrying one of these markers — so sessions/settings
# live INSIDE the project folder and travel with it. Unlike a global store keyed by absolute path
# (Claude-style), moving or renaming the folder never orphans your sessions.
PROJECT_MARKERS = (".crucible", ".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod")


def project_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for d in (cur, *cur.parents):
        if any((d / m).exists() for m in PROJECT_MARKERS):
            return d
    return cur


def store_dir() -> Path:
    """Where CLI state (sessions, settings, audit) lives. `CRUCIBLE_HOME` forces a specific dir
    (e.g. a shared/global one); otherwise it's <project>/.crucible — inside the folder, so the whole
    working set moves with the project. Created on demand."""
    env = os.environ.get("CRUCIBLE_HOME")
    d = Path(env).expanduser() if env else project_root() / ".crucible"
    d.mkdir(parents=True, exist_ok=True)
    return d


# kept as an alias so existing call sites (audit path etc.) read naturally
home = store_dir


def load_settings() -> dict:
    f = store_dir() / "settings.json"
    out = dict(DEFAULTS)
    if f.exists():
        try:
            out.update(json.loads(f.read_text()))
        except (OSError, ValueError):
            pass
    return out


def save_settings(s: dict) -> None:
    (store_dir() / "settings.json").write_text(json.dumps(s, indent=2))


def session_path(name: str) -> Path:
    d = store_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.json"


def most_recent_session() -> str | None:
    """The newest session in this project — what --continue resumes."""
    d = store_dir() / "sessions"
    files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if d.exists() else []
    return files[0].stem if files else None


def save_session(name: str, convo: list[dict]) -> None:
    session_path(name).write_text(json.dumps(convo, indent=2))


def load_session(name: str) -> list[dict]:
    f = session_path(name)
    return json.loads(f.read_text()) if f.exists() else []


def list_sessions() -> list[str]:
    d = store_dir() / "sessions"
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def browse_sessions() -> list[dict]:
    """Rich, newest-first listing of every session in THIS project — for browsing/picking, not just
    names: message count, when it was last touched, and a preview of the first user turn."""
    import time
    d = store_dir() / "sessions"
    out = []
    for p in (d.glob("*.json") if d.exists() else []):
        try:
            convo = json.loads(p.read_text())
        except (OSError, ValueError):
            convo = []
        first = next((m.get("content", "") for m in convo if m.get("role") == "user"), "")
        out.append({"name": p.stem, "msgs": len(convo), "mtime": p.stat().st_mtime,
                    "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                    "preview": " ".join(first.split())[:60]})
    return sorted(out, key=lambda s: s["mtime"], reverse=True)


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
    ap.add_argument("--session", default=None, help="name a session to save/load in <project>/.crucible/sessions")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true",
                    help="resume the most recent session in this project")
    ap.add_argument("--resume", nargs="?", const="__recent__", default=None,
                    help="resume a named session (or the most recent if no name given)")
    ap.add_argument("prompt", nargs="*")
    a = ap.parse_args(argv)

    # Resolve the session to resume: --resume <name> | --resume/-c (most recent) | --session <name>.
    sess = a.session
    if a.resume is not None:
        sess = most_recent_session() if a.resume == "__recent__" else a.resume
    elif a.cont:
        sess = most_recent_session()
    if (a.cont or a.resume is not None) and sess is None:
        print("  (no previous session in this project — starting fresh as 'default')")
        sess = "default"
    convo = load_session(sess) if (sess and session_path(sess).exists()) else []

    st = {"chat": a.endpoint.rstrip("/") + "/chat/completions", "endpoint": a.endpoint,
          "control": a.control, "perm": a.perm,
          "convo": convo, "session": sess, "token": a.token}
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
            rows = browse_sessions()
            if not rows:
                print("  (no sessions in this project yet — /save <name> or run with -c to start one)")
            for s in rows:
                mark = "→" if s["name"] == st["session"] else " "
                print(f"  {mark} {s['name']:20} {s['msgs']:>4} msgs · {s['when']} · {s['preview']}")
        elif cmd == "/config" and len(parts) > 1 and parts[1] == "save":
            save_settings({"endpoint": st["endpoint"], "control": st["control"], "perm": st["perm"]})
            print(f"  settings saved → {home() / 'settings.json'}")
        elif cmd == "/config":
            print(f"  endpoint={st['endpoint']} control={st['control']} perm={st['perm']} session={st['session']}")
        elif cmd == "/where":
            root = project_root()
            print(f"  cwd:           {Path.cwd()}")
            print(f"  project root:  {root}" + ("  (found via marker)" if root != Path.cwd() else "  (no marker — using cwd)"))
            print(f"  session store: {store_dir() / 'sessions'}   ← inside the project, moves with it")
            cur = st["session"]
            print(f"  this session:  {cur or '(unsaved)'}" + (f"  → {session_path(cur)}" if cur else ""))
            try:
                cfgd = httpx.get(f"{st['control']}/api/config", timeout=3).json()
                print(f"  memory dir:    {cfgd.get('memory_dir')}   (server-side; set CRUCIBLE_DATA_DIR to relocate)")
            except (httpx.HTTPError, ValueError):
                print("  memory dir:    (backend offline — start the Crucible API to see it)")
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
    print(f"project: {project_root()} · sessions: {store_dir() / 'sessions'} (/where for details)")
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
