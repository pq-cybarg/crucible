# Git-like edit ledger for live in-memory model editing. Each commit stores the
# PRE-EDIT values of only the tensors it changed (a delta) — so history is tiny even
# for a 1.5TB model, and revert is exact (restore the originals, no lossy inverse).


class EditLedger:
    def __init__(self):
        self.commits: list[dict] = []
        self.deltas: dict[str, dict] = {}   # commit_id -> {tensor_name: original_ndarray}
        self.branch_name = "main"

    def record(self, op: str, params: dict, summary: str, metrics: dict, deltas: dict) -> dict:
        cid = f"c{len(self.commits) + 1}"
        parent = self.commits[-1]["id"] if self.commits else None
        commit = {"id": cid, "parent": parent, "branch": self.branch_name,
                  "op": op, "params": params, "summary": summary, "metrics": metrics,
                  "tensors": sorted(deltas.keys())}
        self.commits.append(commit)
        self.deltas[cid] = deltas
        return commit

    def log(self) -> list[dict]:
        return self.commits

    def get_deltas(self, commit_id: str) -> dict:
        if commit_id not in self.deltas:
            raise KeyError(commit_id)
        return self.deltas[commit_id]

    def set_branch(self, name: str) -> str:
        self.branch_name = name
        return name
