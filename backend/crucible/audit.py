from __future__ import annotations
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
