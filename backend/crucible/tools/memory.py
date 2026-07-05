from __future__ import annotations
# Agent-facing memory retrieval. Compaction crystallizes old context into a versioned memory tree;
# this tool lets the model RECALL it with summary-gated progressive disclosure: call with no args to
# scan the index (key + label + summary of each top-level memory), then open exactly the one you need
# — a leaf returns its full messages, a chunked memory returns its sub-memory summaries to drill in.
# Read summaries first, open bodies only when the summary says it's relevant.
from crucible.tools.base import ToolResult


def _format_node(node: dict) -> str:
    head = f"{node['key']} [{node.get('kind', 'leaf')}] {node.get('label', '')}\nsummary: {node.get('summary', '')}"
    if node.get("kind") == "chunked":
        kids = "\n".join(f"  {c['key']} [{c.get('size', 0)}] {c.get('label', '')}: {c.get('summary', '')}"
                         for c in node.get("children", []))
        return head + "\nsub-memories (open one with key=...):\n" + kids
    msgs = node.get("messages", [])
    body = "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in msgs)
    return head + f"\n--- full context ({len(msgs)} messages) ---\n" + body


class RecallMemory:
    name = "recall_memory"
    description = (
        "Retrieve crystallized memory from earlier (compacted) context. Call with NO arguments to get "
        "the index — each memory's key, label, and summary — then decide what's relevant. Call with "
        "key=<m-XXXX> to open one: a leaf returns its full messages; a chunked memory returns its "
        "sub-memory summaries so you can drill down. Read summaries first; only open the full content "
        "you actually need — this is how you recover old context without reloading the whole thread.")
    parameters = {"type": "object", "properties": {
        "key": {"type": "string", "description": "memory key to open (e.g. m-0002); omit for the index"},
        "session": {"type": "string", "description": "optional session filter when listing the index"}},
        "required": []}

    def __init__(self, root=None):
        from crucible.config import get_settings
        self._root = get_settings().data_dir / "memory"

    def run(self, key: str = "", session: str = "") -> ToolResult:
        from crucible.memory import MemoryStore
        try:
            store = MemoryStore(self._root)
            if (key or "").strip():
                return ToolResult(ok=True, output=_format_node(store.read(key.strip())))
            idx = store.index(session or None)
            if not idx:
                return ToolResult(ok=True, output="(no crystallized memories yet)")
            lines = [f"{c['key']} [{c['kind']}, {c['size']}] {c['label']}: {c['summary']}" for c in idx]
            return ToolResult(ok=True, output="Crystallized memories — open one with key=...:\n" + "\n".join(lines))
        except KeyError:
            return ToolResult(ok=False, output="", error=f"no memory '{key}'")
        except Exception as e:                      # a memory-store hiccup must not kill the agent turn
            return ToolResult(ok=False, output="", error=f"recall failed: {e}")
