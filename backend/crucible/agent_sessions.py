from __future__ import annotations
# Live AGENT SESSIONS — the shared substrate behind the web tabs AND the TUI. Each session is an agent
# bound to a working DIRECTORY, optionally a child of another (a SUBAGENT), with its own conversation
# and a set of SLOTS: crystallized memories and other contexts that can be LOADED or UNLOADED into its
# live context at will. One server-side store so every surface (web, TUI, CLI) sees the same tabs,
# the same slotting, and the same browseable set. Persisted as JSON; pure + unit-tested.
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Slot:
    """A loadable piece of context. kind='memory' → a crystallized memory (ref=its key); kind='context'
    → another agent session's conversation (ref=its id). `enabled` is the in/out toggle — a slot can be
    attached but temporarily unloaded without losing it."""
    kind: str
    ref: str
    label: str = ""
    enabled: bool = True


@dataclass
class AgentSession:
    id: str
    title: str
    cwd: str
    model_id: Optional[str] = None
    parent_id: Optional[str] = None          # set → this is a subagent of another session
    status: str = "idle"                     # idle | running
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    messages: list[dict] = field(default_factory=list)
    slots: list[Slot] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AgentSession":
        slots = [Slot(**s) for s in (d.get("slots") or [])]
        return cls(id=d["id"], title=d.get("title", ""), cwd=d.get("cwd", ""),
                   model_id=d.get("model_id"), parent_id=d.get("parent_id"),
                   status=d.get("status", "idle"), created=d.get("created", _now()),
                   updated=d.get("updated", _now()), messages=list(d.get("messages") or []), slots=slots)

    def card(self) -> dict:
        """A cheap summary for tabs/browsers — no message bodies."""
        return {"id": self.id, "title": self.title, "cwd": self.cwd, "model_id": self.model_id,
                "parent_id": self.parent_id, "status": self.status, "created": self.created,
                "updated": self.updated, "n_messages": len(self.messages),
                "n_slots": len(self.slots), "n_loaded": sum(1 for s in self.slots if s.enabled)}


class AgentSessionStore:
    """CRUD + slotting for live agent sessions, persisted to one JSON file. Ids are stable, readable
    (a-0001…). Deleting a session also removes its subagents (its children)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._seq = 0
        self._sessions: dict[str, AgentSession] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
            self._seq = int(data.get("seq", 0))
            self._sessions = {k: AgentSession.from_dict(v) for k, v in (data.get("sessions") or {}).items()}
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            self._seq, self._sessions = 0, {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"seq": self._seq, "sessions": {k: v.to_dict() for k, v in self._sessions.items()}}, indent=2))

    def _next_id(self) -> str:
        self._seq += 1
        return f"a-{self._seq:04d}"

    # --- CRUD -------------------------------------------------------------------------------------
    def create(self, title: str, cwd: str, model_id: Optional[str] = None,
               parent_id: Optional[str] = None) -> dict:
        if parent_id is not None and parent_id not in self._sessions:
            raise KeyError(f"parent '{parent_id}' not found")
        sid = self._next_id()
        s = AgentSession(id=sid, title=title or sid, cwd=cwd or ".", model_id=model_id, parent_id=parent_id)
        self._sessions[sid] = s
        self._save()
        return s.card()

    def get(self, sid: str) -> AgentSession:
        s = self._sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        return s

    def read(self, sid: str) -> dict:
        return self.get(sid).to_dict()

    def list(self, parent_id: str | None = "__all__") -> list[dict]:
        """Cards for the tab bar / browser, newest activity first. parent_id='__all__' (default) →
        every session; None → only top-level (no parent); an id → that session's subagents."""
        items = list(self._sessions.values())
        if parent_id != "__all__":
            items = [s for s in items if s.parent_id == parent_id]
        return [s.card() for s in sorted(items, key=lambda s: s.updated, reverse=True)]

    def children(self, sid: str) -> list[dict]:
        return self.list(parent_id=sid)

    def update(self, sid: str, **fields) -> dict:
        s = self.get(sid)
        for k in ("title", "cwd", "model_id", "status", "messages"):
            if k in fields and fields[k] is not None:
                setattr(s, k, fields[k])
        s.updated = _now()
        self._save()
        return s.card()

    def delete(self, sid: str) -> bool:
        if sid not in self._sessions:
            return False
        # cascade to subagents
        for child in [c for c in self._sessions.values() if c.parent_id == sid]:
            self._sessions.pop(child.id, None)
        self._sessions.pop(sid, None)
        self._save()
        return True

    # --- slotting: load / unload memories & contexts ----------------------------------------------
    def attach_slot(self, sid: str, kind: str, ref: str, label: str = "") -> dict:
        if kind not in ("memory", "context"):
            raise ValueError("slot kind must be 'memory' or 'context'")
        s = self.get(sid)
        if kind == "context" and ref == sid:
            raise ValueError("a session cannot load itself as context")
        for existing in s.slots:
            if existing.kind == kind and existing.ref == ref:
                existing.enabled = True                 # re-attaching a detached slot re-enables it
                s.updated = _now(); self._save()
                return s.to_dict()
        s.slots.append(Slot(kind=kind, ref=ref, label=label, enabled=True))
        s.updated = _now(); self._save()
        return s.to_dict()

    def set_slot_enabled(self, sid: str, kind: str, ref: str, enabled: bool) -> dict:
        """Slot IN or OUT without removing — the loadable/unloadable toggle."""
        s = self.get(sid)
        for slot in s.slots:
            if slot.kind == kind and slot.ref == ref:
                slot.enabled = enabled
                s.updated = _now(); self._save()
                return s.to_dict()
        raise KeyError(f"no {kind} slot '{ref}'")

    def detach_slot(self, sid: str, kind: str, ref: str) -> dict:
        s = self.get(sid)
        before = len(s.slots)
        s.slots = [sl for sl in s.slots if not (sl.kind == kind and sl.ref == ref)]
        if len(s.slots) == before:
            raise KeyError(f"no {kind} slot '{ref}'")
        s.updated = _now(); self._save()
        return s.to_dict()

    # --- assembled live context -------------------------------------------------------------------
    def assembled_context(self, sid: str,
                          memory_text: Callable[[str], str] | None = None) -> list[dict]:
        """The session's LIVE context: enabled memory slots + enabled context slots injected ahead of
        the conversation. Memories are resolved to text via `memory_text(key)`; context slots pull the
        other session's messages. Disabled (slotted-out) slots contribute nothing. This is exactly what
        a run would send — so a UI can preview it."""
        s = self.get(sid)
        prefix: list[dict] = []
        for slot in s.slots:
            if not slot.enabled:
                continue
            if slot.kind == "memory" and memory_text is not None:
                try:
                    body = memory_text(slot.ref)
                except Exception:
                    body = ""
                if body:
                    prefix.append({"role": "system",
                                   "content": f"[loaded memory {slot.ref}{' — ' + slot.label if slot.label else ''}]\n{body}"})
            elif slot.kind == "context":
                other = self._sessions.get(slot.ref)
                if other is not None and other.messages:
                    convo = "\n".join(f"{m.get('role')}: {m.get('content','')}" for m in other.messages)
                    prefix.append({"role": "system",
                                   "content": f"[loaded context {slot.ref}{' — ' + slot.label if slot.label else ''}]\n{convo}"})
        return prefix + list(s.messages)
