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
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, RichLog, Static


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


class CrucibleTUI(App):
    CSS = """
    #cols { height: 1fr; }
    #left, #right { width: 32; border: round $panel; }
    #centre { width: 1fr; border: round $panel; }
    .heading { color: $accent; text-style: bold; padding: 0 1; }
    #slots { height: 8; border-bottom: solid $panel; }
    #context { height: 1fr; }
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
                yield Input(placeholder="message the active agent — Enter runs it in its directory…", id="composer")
            with Vertical(id="right"):
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
    def action_new_agent(self, parent: Optional[str] = None) -> None:
        # a tab in the current directory by default; title from the id
        import os
        try:
            card = self.client.create("agent", os.getcwd(), parent_id=parent)
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

    def on_input_submitted(self, event: Input.Submitted) -> None:
        msg = event.value.strip()
        if not (self.active and msg):
            return
        event.input.value = ""
        self.query_one("#context", RichLog).write(f"[cyan]user[/cyan] {msg}")
        self._run(self.active, msg)

    @work(thread=True)
    def _run(self, sid: str, message: str) -> None:
        log = self.query_one("#context", RichLog)
        acc = ""
        try:
            with httpx.stream("POST", f"{self.control}/api/agent-sessions/{sid}/run",
                              json={"message": message}, timeout=600) as r:
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
                    if ev["type"] == "assistant_delta":
                        acc += str(ev["data"].get("delta", ""))
                    elif ev["type"] in ("assistant", "done") and ev["data"].get("content"):
                        acc = str(ev["data"]["content"])
                    elif ev["type"] == "error":
                        self.call_from_thread(log.write, f"[red]{ev['data'].get('reason')}[/red]")
        except httpx.HTTPError as e:
            self.call_from_thread(log.write, f"[red]{e}[/red]")
        if acc:
            self.call_from_thread(log.write, f"[green]assistant[/green] {acc}")
        self.refresh_all()


def run_tui(control: str = "http://127.0.0.1:8400", cwd: str = ".") -> None:
    CrucibleTUI(control, cwd=cwd).run()
