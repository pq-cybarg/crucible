from __future__ import annotations
# Agent hierarchy profiles. A fractal agent tree works better when each LAYER uses the right model
# and each worker is PAIRED with a lighter "communicator" that relays between layers — compressing a
# deep child's result before it climbs back up, so a parent never has to process raw deep-leaf text.
# A PROFILE names a per-layer configuration: for each depth, which model does the work (usually the
# stronger one) and which lighter/simpler model carries messages up and down. Deeper than the defined
# layers reuses the last layer. Multiple named profiles are supported; the store persists them.
# Pure config + a pure `relay` step (the summarizing solver is injected) — both unit-tested.
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class Layer:
    worker: Optional[str] = None         # model id doing the work at this depth (None = default model)
    communicator: Optional[str] = None   # lighter model relaying up/down (None = no relay / passthrough)


@dataclass
class HierarchyProfile:
    name: str
    layers: list[Layer] = field(default_factory=list)

    def at(self, depth: int) -> Layer:
        """The layer config for a given tree depth; deeper than defined reuses the last layer."""
        if not self.layers:
            return Layer()
        return self.layers[min(max(0, depth), len(self.layers) - 1)]

    def to_dict(self) -> dict:
        return {"name": self.name, "layers": [{"worker": l.worker, "communicator": l.communicator} for l in self.layers]}

    @classmethod
    def from_dict(cls, d: dict) -> "HierarchyProfile":
        return cls(name=str(d.get("name", "")),
                   layers=[Layer(worker=l.get("worker"), communicator=l.get("communicator"))
                           for l in (d.get("layers") or [])])


RELAY_INSTRUCTION = (
    "You are a communicator between layers of an agent tree. Compress the sub-agent result below to "
    "its essential findings and decisions for the parent — keep facts and conclusions, drop chatter. "
    "Be terse.\n\nSUB-AGENT RESULT:\n")


def relay(result: str, communicator: Optional[Callable[[str], str]]) -> str:
    """The communicator step: a lighter model compresses a child's output before it goes UP a layer,
    so ancestors read a tight summary instead of raw deep-leaf text. No communicator -> passthrough."""
    if communicator is None or not (result or "").strip():
        return result
    try:
        out = communicator(RELAY_INSTRUCTION + result)
        return out.strip() or result
    except Exception:
        return result           # a flaky relay must never lose the child's answer


DEFAULT_PROFILES = [
    HierarchyProfile("flat", [Layer()]),   # every layer uses the default model, no relay
    HierarchyProfile("worker+communicator", [Layer(worker=None, communicator=None)]),  # filled in by the operator
]


class ProfileStore:
    """Persists named hierarchy profiles to a JSON file (like the recipe/preset stores)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            self._write({p.name: p.to_dict() for p in DEFAULT_PROFILES})

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    def list(self) -> list[dict]:
        return list(self._read().values())

    def get(self, name: str) -> HierarchyProfile:
        d = self._read().get(name)
        if d is None:
            raise KeyError(name)
        return HierarchyProfile.from_dict(d)

    def save(self, profile: HierarchyProfile) -> dict:
        data = self._read()
        data[profile.name] = profile.to_dict()
        self._write(data)
        return profile.to_dict()

    def delete(self, name: str) -> bool:
        data = self._read()
        if name in data:
            del data[name]
            self._write(data)
            return True
        return False
