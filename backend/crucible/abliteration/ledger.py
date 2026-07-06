from __future__ import annotations
# Git-like edit ledger for live in-memory model editing. Each commit stores the PRE-EDIT values of
# only the tensors it changed (a delta) — so history is tiny even for a 1.5TB model, and revert is
# exact (restore the originals, no lossy inverse). A composed/multimodal model is several PARTS
# (vision/audio encoder, connector, language model, moderation head), each versioned separately: so
# every commit is also tagged with the part(s) it touched, giving each part its own LINEAGE and a
# per-part revert — undo the vision-encoder edit without disturbing the language-model one.
from crucible.abliteration.composition import part_of


class EditLedger:
    def __init__(self):
        self.commits: list[dict] = []
        self.deltas: dict[str, dict] = {}   # commit_id -> {tensor_name: original_ndarray}
        self.branch_name = "main"

    def record(self, op: str, params: dict, summary: str, metrics: dict, deltas: dict) -> dict:
        cid = f"c{len(self.commits) + 1}"
        parent = self.commits[-1]["id"] if self.commits else None
        parts = sorted({part_of(name) for name in deltas}) if deltas else []
        commit = {"id": cid, "parent": parent, "branch": self.branch_name,
                  "op": op, "params": params, "summary": summary, "metrics": metrics,
                  "tensors": sorted(deltas.keys()), "parts": parts}
        self.commits.append(commit)
        self.deltas[cid] = deltas
        return commit

    def log(self, part: str | None = None) -> list[dict]:
        """All commits, or only those that touched `part`."""
        if part is None:
            return self.commits
        return [c for c in self.commits if part in c.get("parts", [])]

    def lineage(self) -> list[dict]:
        """Per-part version chains: for each part any commit touched, its ordered commits + count +
        latest — so each subsystem's edit history is independently visible and revertable."""
        by_part: dict[str, list[dict]] = {}
        for c in self.commits:
            for p in c.get("parts", []):
                by_part.setdefault(p, []).append(c)
        return [{"part": p, "n_versions": len(cs), "latest": cs[-1]["id"],
                 "commits": [{"id": c["id"], "op": c["op"], "summary": c["summary"]} for c in cs]}
                for p, cs in sorted(by_part.items())]

    def latest_for_part(self, part: str) -> dict | None:
        """The most recent commit that touched `part` (None if the part was never edited)."""
        for c in reversed(self.commits):
            if part in c.get("parts", []):
                return c
        return None

    def deltas_for_part(self, commit_id: str, part: str) -> dict:
        """The subset of a commit's pre-edit deltas belonging to `part` — for a per-part revert."""
        return {name: W for name, W in self.get_deltas(commit_id).items() if part_of(name) == part}

    def get_deltas(self, commit_id: str) -> dict:
        if commit_id not in self.deltas:
            raise KeyError(commit_id)
        return self.deltas[commit_id]

    def set_branch(self, name: str) -> str:
        self.branch_name = name
        return name
