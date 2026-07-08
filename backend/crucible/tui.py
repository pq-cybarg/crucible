from __future__ import annotations
# Fullscreen TUI for the agent workbench — a thin client over the SAME backend as the web UI, so its
# tabs, subagents, and loadable memory/context slots are the exact same objects. Left: agent tabs
# (subagents indented). Centre: the active tab's slots, its live assembled context, and a composer to
# RUN the agent in its working directory. Right: a browser of everything loadable (memories + other
# contexts). Everything routes through /api/agent-sessions on the control server.
import json
from typing import Any, Optional

import httpx

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, RichLog, Static

# Slash commands available in the composer (name -> help). Shown by /help and the autocomplete line.
COMMANDS: dict[str, str] = {
    "/help": "list commands",
    "/models": "pick the model for this tab (browse & select)",
    "/new": "/new [title] — open a new agent tab in the current dir",
    "/sub": "/sub [title] — open a subagent under this tab",
    "/close": "close the active tab",
    "/clear": "clear this tab's conversation",
    "/where": "show working dir, project, session + memory locations",
    "/slots": "list the slots loaded into this tab",
    "/load": "/load <memory-key|session-id> — load a slot",
}


class ModelPicker(ModalScreen):
    """A browse-and-select model list (like OpenCode's model picker). Returns the chosen model id."""
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    ModelPicker { align: center middle; }
    #picker { width: 70; height: 22; border: round $accent; background: $panel; padding: 1; }
    #picker Label { color: $accent; text-style: bold; }
    """

    def __init__(self, models: list[dict]):
        super().__init__()
        self.models = models

    def compose(self) -> ComposeResult:
        items = []
        for m in self.models:
            dot = "[green]●[/green]" if m["online"] else ("[yellow]○[/yellow]" if m["servable"] else "[dim]·[/dim]")
            it = ListItem(Label(f"{dot} {m['name']}  [dim]{m['id']} · {m['kind']}[/dim]"))
            it.model_id = m["id"]  # type: ignore[attr-defined]
            items.append(it)
        with Vertical(id="picker"):
            yield Label("SELECT MODEL  (Enter = choose · Esc = cancel)")
            yield ListView(*items)   # construct with items — can't append before mount

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(getattr(event.item, "model_id", None))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ApprovalScreen(ModalScreen):
    """Prompt to approve/deny a pending 'ask' tool call from a running tab."""
    BINDINGS = [Binding("a", "ok", "Approve"), Binding("d", "no", "Deny"), Binding("escape", "no", "Deny")]
    CSS = """
    ApprovalScreen { align: center middle; }
    #ask { width: 66; height: auto; border: round $warning; background: $panel; padding: 1 2; }
    #ask Label { color: $warning; text-style: bold; }
    """

    def __init__(self, name: str, args: dict):
        super().__init__()
        self.tool_name = name
        self.args = args

    def compose(self) -> ComposeResult:
        with Vertical(id="ask"):
            yield Label(f"approve tool: {self.tool_name}?")
            yield Static(str(self.args)[:200])
            yield Static("[b]a[/b] approve   ·   [b]d[/b] deny", markup=True)

    def action_ok(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class Client:
    """Blocking HTTP client for the control server (used from worker threads)."""

    def __init__(self, control: str):
        self.base = control.rstrip("/")

    def _j(self, r: httpx.Response) -> Any:
        r.raise_for_status()
        return r.json()

    def sessions(self) -> list[dict]:
        return self._j(httpx.get(f"{self.base}/api/agent-sessions", timeout=10)).get("sessions", [])

    def get(self, sid: str) -> dict:
        return self._j(httpx.get(f"{self.base}/api/agent-sessions/{sid}", timeout=10))

    def context(self, sid: str) -> list[dict]:
        return self._j(httpx.get(f"{self.base}/api/agent-sessions/{sid}/context", timeout=10)).get("messages", [])

    def create(self, title: str, cwd: str, parent_id: Optional[str] = None) -> dict:
        return self._j(httpx.post(f"{self.base}/api/agent-sessions",
                                  json={"title": title, "cwd": cwd, "parent_id": parent_id}, timeout=10))

    def delete(self, sid: str) -> None:
        httpx.delete(f"{self.base}/api/agent-sessions/{sid}", timeout=10)

    def memories(self) -> list[dict]:
        return self._j(httpx.get(f"{self.base}/api/memory/index", timeout=10)).get("memories", [])

    def attach(self, sid: str, kind: str, ref: str, label: str = "") -> None:
        httpx.post(f"{self.base}/api/agent-sessions/{sid}/slots",
                   json={"kind": kind, "ref": ref, "label": label}, timeout=10)

    def toggle(self, sid: str, kind: str, ref: str, enabled: bool) -> None:
        httpx.patch(f"{self.base}/api/agent-sessions/{sid}/slots",
                    json={"kind": kind, "ref": ref, "enabled": enabled}, timeout=10)

    def detach(self, sid: str, kind: str, ref: str) -> None:
        httpx.request("DELETE", f"{self.base}/api/agent-sessions/{sid}/slots",
                      params={"kind": kind, "ref": ref}, timeout=10)

    def models(self) -> list[dict]:
        """Registered models merged with live status (online/servable) — for the /models picker."""
        rows = self._j(httpx.get(f"{self.base}/api/models", timeout=10))
        try:   # status probes every endpoint and can be slow with many models — keep the picker snappy
            status = {s["id"]: s for s in self._j(httpx.get(f"{self.base}/api/models/status", timeout=4))}
        except httpx.HTTPError:
            status = {}
        return [{"id": m["id"], "name": m.get("name", m["id"]), "kind": m.get("kind", ""),
                 "online": status.get(m["id"], {}).get("online", False),
                 "servable": status.get(m["id"], {}).get("servable", False)} for m in rows]

    def set_model(self, sid: str, model_id: str) -> None:
        httpx.patch(f"{self.base}/api/agent-sessions/{sid}", json={"model_id": model_id}, timeout=10)

    def approve(self, run_id: str, call_id: str, approved: bool) -> None:
        httpx.post(f"{self.base}/api/agent/approve",
                   json={"run_id": run_id, "call_id": call_id, "approved": approved}, timeout=10)

    def update_messages(self, sid: str, messages: list) -> None:
        httpx.patch(f"{self.base}/api/agent-sessions/{sid}", json={"messages": messages}, timeout=10)


class FaceWidget(Static):
    """The avatar face box: a low-res, low-fps pixel-art face in the sidebar that blinks and shifts
    expression in real time — so you see the companion react while you code, decoupled from replies."""

    def __init__(self, avatar, cols: int = 18):
        super().__init__(id="face")
        self.avatar = avatar
        self.cols = cols
        self.expression = "neutral"
        self._t = 0
        self._blink = False

    def on_mount(self) -> None:
        self.redraw()
        self.set_interval(0.4, self._tick)     # low frame rate on purpose

    def _tick(self) -> None:
        self._t += 1
        self._blink = (self._t % 12 == 0)      # a quick blink roughly every ~5s
        self.redraw()

    def redraw(self) -> None:
        from rich.text import Text
        from crucible.avatar import render_tui
        try:
            ov = {"eyes": "closed"} if self._blink else None
            lines = render_tui(self.avatar, self.expression, overrides=ov, cols=self.cols)
            self.update(Text.from_ansi("\n".join(lines)))
        except Exception as e:
            self.update(f"[face error: {e}]")

    def set_expression(self, expression: str) -> None:
        self.expression = expression
        self.redraw()


class CrucibleTUI(App):
    CSS = """
    #cols { height: 1fr; }
    #left, #right { width: 32; border: round $panel; }
    #centre { width: 1fr; border: round $panel; }
    .heading { color: $accent; text-style: bold; padding: 0 1; }
    #slots { height: 8; border-bottom: solid $panel; }
    #context { height: 1fr; }
    #face { height: auto; padding: 0 1; }
    #suggest { height: auto; color: $text-muted; padding: 0 1; }
    Input { dock: bottom; }
    ListView { height: 1fr; }
    """
    BINDINGS = [
        Binding("n", "new_agent", "New agent"),
        Binding("s", "new_subagent", "Subagent"),
        Binding("x", "close_tab", "Close tab"),
        Binding("t", "toggle_slot", "Load/unload slot"),
        Binding("l", "load_selected", "Load from browser"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, control: str = "http://127.0.0.1:8400", cwd: str = "."):
        super().__init__()
        self.client = Client(control)
        self.control = control
        self.cwd = cwd
        self.active: Optional[str] = None
        self._sessions: list[dict] = []
        self._active_doc: dict = {}
        self._browse: list[dict] = []       # loadable items: {kind, ref, label}
        self._bootstrapped = False          # auto-open a tab for cwd on first successful load
        try:
            from crucible.avatar_gen import ensure_default_avatar
            from crucible.config import get_settings
            self._avatar = ensure_default_avatar(str(get_settings().data_dir))
        except Exception:
            self._avatar = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="cols"):
            with Vertical(id="left"):
                yield Label("AGENT TABS", classes="heading")
                yield ListView(id="tabs")
            with Vertical(id="centre"):
                yield Label("SLOTS  (t: load/unload · type below to run)", classes="heading")
                yield ListView(id="slots")
                yield Label("LIVE CONTEXT", classes="heading")
                yield RichLog(id="context", wrap=True, markup=True)
                yield Static("", id="suggest")
                yield Input(placeholder="message the agent — or /help for commands…", id="composer")
            with Vertical(id="right"):
                if self._avatar is not None:
                    yield Label("COMPANION", classes="heading")
                    yield FaceWidget(self._avatar)
                yield Label("BROWSE · LOAD (l)", classes="heading")
                yield ListView(id="browser")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "crucible — agent workbench"
        self.sub_title = self.control
        self.refresh_all()

    # --- data loading (threaded) ------------------------------------------------------------------
    @work(thread=True, exclusive=True)
    def refresh_all(self) -> None:
        import os
        try:
            sessions = self.client.sessions()
            # first launch in a real project → open a tab bound to this directory, ready to code
            if not self._bootstrapped:
                self._bootstrapped = True
                if not any(s["cwd"] == self.cwd for s in sessions):
                    card = self.client.create(os.path.basename(self.cwd.rstrip("/")) or "agent", self.cwd)
                    self.active = card["id"]
                    sessions = self.client.sessions()
            memories = self.client.memories()
        except httpx.HTTPError as e:
            self.call_from_thread(self._toast, f"backend offline: {e}")
            return
        self.call_from_thread(self._render_tabs, sessions)
        if self.active is None and sessions:
            self.active = sessions[0]["id"]
        if self.active:
            try:
                doc = self.client.get(self.active)
                ctx = self.client.context(self.active)
            except httpx.HTTPError:
                doc, ctx = {}, []
            self.call_from_thread(self._render_active, doc, ctx, sessions, memories)

    def _toast(self, msg: str) -> None:
        self.query_one("#context", RichLog).write(f"[red]{msg}[/red]")

    def _render_tabs(self, sessions: list[dict]) -> None:
        self._sessions = sessions
        lv = self.query_one("#tabs", ListView)
        lv.clear()
        top = [s for s in sessions if not s["parent_id"]]
        for t in top:
            lv.append(self._tab_item(t, 0))
            for k in [s for s in sessions if s["parent_id"] == t["id"]]:
                lv.append(self._tab_item(k, 1))

    def _tab_item(self, s: dict, depth: int) -> ListItem:
        mark = "» " if s["id"] == self.active else ("  " if depth == 0 else "  ↳ ")
        run = "●" if s["status"] == "running" else " "
        item = ListItem(Label(f"{mark}{run} {s['title']}  [dim]{s['n_loaded']}/{s['n_slots']}··{s['cwd']}[/dim]"))
        item.crucible_id = s["id"]  # type: ignore[attr-defined]
        return item

    def _render_active(self, doc: dict, ctx: list[dict], sessions: list[dict], memories: list[dict]) -> None:
        self._active_doc = doc
        # slots
        slots = self.query_one("#slots", ListView)
        slots.clear()
        for sl in doc.get("slots", []):
            box = "[green]■[/green]" if sl["enabled"] else "[dim]□[/dim]"
            it = ListItem(Label(f"{box} [b]{sl['kind']}[/b] {sl['ref']}  [dim]{sl['label']}[/dim]"))
            it.crucible_slot = (sl["kind"], sl["ref"], sl["enabled"])  # type: ignore[attr-defined]
            slots.append(it)
        # live context
        log = self.query_one("#context", RichLog)
        log.clear()
        for m in ctx:
            log.write(f"[cyan]{m.get('role')}[/cyan] {m.get('content','')[:600]}")
        # browser: memories + other sessions
        self._browse = [{"kind": "memory", "ref": m["key"], "label": m["label"]} for m in memories]
        self._browse += [{"kind": "context", "ref": s["id"], "label": s["title"]}
                         for s in sessions if s["id"] != self.active]
        loaded = {(sl["kind"], sl["ref"]) for sl in doc.get("slots", [])}
        br = self.query_one("#browser", ListView)
        br.clear()
        for b in self._browse:
            tag = "[green]loaded[/green]" if (b["kind"], b["ref"]) in loaded else "load"
            it = ListItem(Label(f"[dim]{tag}[/dim] [b]{b['kind']}[/b] {b['ref']}  {b['label']}"))
            it.crucible_browse = b  # type: ignore[attr-defined]
            br.append(it)

    # --- interactions -----------------------------------------------------------------------------
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if getattr(item, "crucible_id", None):     # a tab was chosen → make it active
            self.active = item.crucible_id          # type: ignore[attr-defined]
            self.refresh_all()

    def action_refresh(self) -> None:
        self.refresh_all()

    @work(thread=True)
    def action_new_agent(self, parent: Optional[str] = None, title: Optional[str] = None) -> None:
        import os
        cwd = os.getcwd()
        name = title or (("sub of " + self.active) if parent else os.path.basename(cwd.rstrip("/")) or "agent")
        try:
            card = self.client.create(name, cwd, parent_id=parent)
        except httpx.HTTPError as e:
            self.call_from_thread(self._toast, f"create failed: {e}"); return
        self.active = card["id"]
        self.refresh_all()

    def action_new_subagent(self) -> None:
        if self.active:
            self.action_new_agent(self.active)

    @work(thread=True)
    def action_close_tab(self) -> None:
        if not self.active:
            return
        try:
            self.client.delete(self.active)
        except httpx.HTTPError:
            pass
        self.active = None
        self.refresh_all()

    @work(thread=True)
    def action_toggle_slot(self) -> None:
        item = self.query_one("#slots", ListView).highlighted_child
        got = getattr(item, "crucible_slot", None)
        if not (self.active and got):
            return
        kind, ref, enabled = got
        try:
            self.client.toggle(self.active, kind, ref, not enabled)
        except httpx.HTTPError:
            pass
        self.refresh_all()

    @work(thread=True)
    def action_load_selected(self) -> None:
        item = self.query_one("#browser", ListView).highlighted_child
        b = getattr(item, "crucible_browse", None)
        if not (self.active and b):
            return
        try:
            self.client.attach(self.active, b["kind"], b["ref"], b["label"])
        except httpx.HTTPError:
            pass
        self.refresh_all()

    def on_input_changed(self, event: Input.Changed) -> None:
        # slash-command autocomplete: show matching commands as you type "/"
        sug = self.query_one("#suggest", Static)
        val = event.value
        if val.startswith("/"):
            head = val.split(" ", 1)[0]
            hits = [f"[b]{c}[/b] — {h}" for c, h in COMMANDS.items() if c.startswith(head)]
            sug.update("  ".join(hits[:6]) if hits else "[dim]no such command — /help[/dim]")
        else:
            sug.update("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        val = event.value.strip()
        event.input.value = ""
        self.query_one("#suggest", Static).update("")
        if not val:
            return
        if val.startswith("/"):
            self._command(val)
            return
        if not self.active:
            self._toast("no active tab — /new to open one")
            return
        self.query_one("#context", RichLog).write(f"[cyan]user[/cyan] {val}")
        self._run(self.active, val)

    def _command(self, line: str) -> None:
        parts = line.split()
        cmd, args = parts[0], parts[1:]
        log = self.query_one("#context", RichLog)
        if cmd == "/help":
            log.write("[b]commands[/b]: " + "  ".join(f"[cyan]{c}[/cyan]" for c in COMMANDS))
            for c, h in COMMANDS.items():
                log.write(f"  [cyan]{c}[/cyan] — {h}")
        elif cmd == "/models":
            self._open_models()
        elif cmd == "/new":
            self.action_new_agent(title=" ".join(args) or None)
        elif cmd == "/sub":
            if self.active:
                self.action_new_agent(self.active, title=" ".join(args) or None)
        elif cmd == "/close":
            self.action_close_tab()
        elif cmd == "/clear":
            self._clear_convo()
        elif cmd == "/where":
            import os
            log.write(f"[b]cwd[/b] {os.getcwd()}")
            log.write(f"[b]control[/b] {self.control}")
            if self.active:
                d = self._active_doc
                log.write(f"[b]tab[/b] {self.active} · dir {d.get('cwd')} · model {d.get('model_id') or '(server default)'}")
        elif cmd == "/slots":
            for sl in self._active_doc.get("slots", []):
                log.write(f"  {'■' if sl['enabled'] else '□'} {sl['kind']} {sl['ref']} {sl['label']}")
        elif cmd == "/load" and args:
            ref = args[0]
            kind = "context" if ref.startswith("a-") else "memory"
            self._load_ref(kind, ref)
        else:
            log.write(f"[dim]unknown command '{cmd}' — /help[/dim]")

    @work(thread=True)
    def _open_models(self) -> None:
        try:
            models = self.client.models()
        except httpx.HTTPError as e:
            self.call_from_thread(self._toast, f"models unavailable: {e}"); return
        self.call_from_thread(self._show_picker, models)

    def _show_picker(self, models: list[dict]) -> None:
        def picked(model_id: Optional[str]) -> None:
            if model_id and self.active:
                self._set_model(model_id)
        self.push_screen(ModelPicker(models), picked)

    @work(thread=True)
    def _set_model(self, model_id: str) -> None:
        if self.active:
            try:
                self.client.set_model(self.active, model_id)
            except httpx.HTTPError:
                pass
        self.refresh_all()

    @work(thread=True)
    def _load_ref(self, kind: str, ref: str) -> None:
        if self.active:
            try:
                self.client.attach(self.active, kind, ref)
            except httpx.HTTPError as e:
                self.call_from_thread(self._toast, f"load failed: {e}")
        self.refresh_all()

    @work(thread=True)
    def _clear_convo(self) -> None:
        if self.active:
            try:
                self.client.update_messages(self.active, [])
            except httpx.HTTPError:
                pass
        self.refresh_all()

    @work(thread=True)
    def _run(self, sid: str, message: str) -> None:
        import time
        log = self.query_one("#context", RichLog)
        run_id = f"{sid}-{int(time.monotonic() * 1000)}"
        acc = ""
        try:
            with httpx.stream("POST", f"{self.control}/api/agent-sessions/{sid}/run",
                              json={"message": message, "run_id": run_id}, timeout=600) as r:
                if r.status_code >= 400:
                    self.call_from_thread(log.write, f"[red]run error {r.status_code}[/red]"); return
                for line in r.iter_lines():
                    line = (line or "").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:])
                    except ValueError:
                        continue
                    t = ev["type"]
                    if t == "assistant_delta":
                        acc += str(ev["data"].get("delta", ""))
                    elif t in ("assistant", "done") and ev["data"].get("content"):
                        acc = str(ev["data"]["content"])
                    elif t == "tool_call":
                        self.call_from_thread(log.write, f"[yellow]→ {ev['data'].get('name')}[/yellow] {json.dumps(ev['data'].get('args'))[:80]}")
                    elif t == "tool_result":
                        self.call_from_thread(log.write, f"  {'✓' if ev['data'].get('ok') else '✗'} {str(ev['data'].get('output') or ev['data'].get('error') or '')[:120]}")
                    elif t == "permission_request":
                        self.call_from_thread(self._ask_approval, run_id, str(ev["data"]["id"]),
                                              str(ev["data"]["name"]), ev["data"].get("args", {}))
                    elif t == "error":
                        self.call_from_thread(log.write, f"[red]{ev['data'].get('reason')}[/red]")
        except httpx.HTTPError as e:
            self.call_from_thread(log.write, f"[red]{e}[/red]")
        if acc:
            self.call_from_thread(log.write, f"[green]assistant[/green] {acc}")
        self.refresh_all()

    def _ask_approval(self, run_id: str, call_id: str, name: str, args: dict) -> None:
        def decided(ok: Optional[bool]) -> None:
            self._send_approval(run_id, call_id, bool(ok))
        self.push_screen(ApprovalScreen(name, args), decided)

    @work(thread=True)
    def _send_approval(self, run_id: str, call_id: str, approved: bool) -> None:
        try:
            self.client.approve(run_id, call_id, approved)
        except httpx.HTTPError:
            pass


def run_tui(control: str = "http://127.0.0.1:8400", cwd: str = ".") -> None:
    CrucibleTUI(control, cwd=cwd).run()
