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
        "the index — each memory's key, label, and summary. Call with query=<text> to RELEVANCE-SEARCH "
        "the memories (ranked by how well they match). Call with key=<m-XXXX> to open one: a leaf returns "
        "its full messages; a chunked memory returns its sub-memory summaries to drill down. Read "
        "summaries first; only open the full content you need — recover old context without reloading it.")
    parameters = {"type": "object", "properties": {
        "key": {"type": "string", "description": "memory key to open (e.g. m-0002); omit for index/search"},
        "query": {"type": "string", "description": "relevance-search the memories by this text"},
        "session": {"type": "string", "description": "optional session filter"}},
        "required": []}

    def __init__(self, root=None):
        from crucible.config import get_settings
        self._root = get_settings().data_dir / "memory"

    def run(self, key: str = "", query: str = "", session: str = "") -> ToolResult:
        from crucible.memory import MemoryStore
        try:
            store = MemoryStore(self._root)
            if (key or "").strip():
                return ToolResult(ok=True, output=_format_node(store.read(key.strip())))
            if (query or "").strip():
                res = store.search(query.strip(), embedder=None, session=session or None)   # lexical from a tool
                if not res["matches"]:
                    return ToolResult(ok=True, output=f"(no memories match '{query}')")
                lines = [f"{m['key']} [{m['kind']}, score {m['score']}] {m['label']}: {m['summary']}"
                         for m in res["matches"]]
                return ToolResult(ok=True, output=f"Memories matching '{query}' ({res['method']}):\n" + "\n".join(lines))
            idx = store.index(session or None)
            if not idx:
                return ToolResult(ok=True, output="(no crystallized memories yet)")
            lines = [f"{c['key']} [{c['kind']}, {c['size']}] {c['label']}: {c['summary']}" for c in idx]
            return ToolResult(ok=True, output="Crystallized memories — open one with key=...:\n" + "\n".join(lines))
        except KeyError:
            return ToolResult(ok=False, output="", error=f"no memory '{key}'")
        except Exception as e:                      # a memory-store hiccup must not kill the agent turn
            return ToolResult(ok=False, output="", error=f"recall failed: {e}")


def _store():
    from crucible.config import get_settings
    from crucible.memory import MemoryStore
    return MemoryStore(get_settings().data_dir / "memory")


class CrystallizeMemory:
    name = "crystallize_memory"
    description = (
        "Save a durable memory you want to keep across compactions — a fact, a decision, a plan. Give "
        "a one-line SUMMARY (the passthrough others will scan) and the full CONTENT. Optionally a short "
        "label and a session tag. Returns the new memory key.")
    parameters = {"type": "object", "properties": {
        "summary": {"type": "string", "description": "one-line summary — the passthrough"},
        "content": {"type": "string", "description": "the full material to remember"},
        "label": {"type": "string"}, "session": {"type": "string"}}, "required": ["summary", "content"]}

    def __init__(self, root=None):
        pass

    def run(self, summary: str = "", content: str = "", label: str = "", session: str = "") -> ToolResult:
        if not summary.strip() or not content.strip():
            return ToolResult(ok=False, output="", error="summary and content are required")
        try:
            node = _store().crystallize([{"role": "memory", "content": content}], summary,
                                        label=label, session=session)
            return ToolResult(ok=True, output=f"saved {node['key']} ({node['label']})")
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"crystallize failed: {e}")


class RecrystallizeMemory:
    name = "recrystallize_memory"
    description = (
        "Reorganize a leaf memory into finer labelled subchunks so future recalls drill down instead of "
        "re-reading it all. YOU are the summarizer: provide one summary per subchunk (and optional "
        "labels); the memory's messages are split into that many contiguous parts. Returns the child keys.")
    parameters = {"type": "object", "properties": {
        "key": {"type": "string"},
        "summaries": {"type": "array", "items": {"type": "string"},
                      "description": "one summary per subchunk (order = chronological split)"},
        "labels": {"type": "array", "items": {"type": "string"}}},
        "required": ["key", "summaries"]}

    def __init__(self, root=None):
        pass

    def run(self, key: str = "", summaries=None, labels=None) -> ToolResult:
        summaries = summaries or []
        labels = labels or []
        if not key.strip() or not summaries:
            return ToolResult(ok=False, output="", error="key and at least one summary are required")
        try:
            store = _store()
            node = store.read(key.strip())
            msgs = node.get("messages")
            if node.get("kind") != "leaf" or not msgs:
                return ToolResult(ok=False, output="", error="can only re-crystallize a leaf memory with messages")
            k = max(1, min(len(summaries), len(msgs)))
            size = (len(msgs) + k - 1) // k
            subchunks = []
            for i in range(k):
                grp = msgs[i * size:(i + 1) * size]
                if not grp:
                    break
                subchunks.append({"label": labels[i] if i < len(labels) else "",
                                  "summary": summaries[i], "messages": grp})
            res = store.recrystallize(key.strip(), subchunks)
            return ToolResult(ok=True, output=f"{key} -> {len(res['children'])} subchunks: {', '.join(res['children'])}")
        except KeyError:
            return ToolResult(ok=False, output="", error=f"no memory '{key}'")
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"recrystallize failed: {e}")


class ConsolidateMemory:
    name = "consolidate_memory"
    description = (
        "File a SET of memories under a new parent to organize/prune. Siblings bubble to their shared "
        "parent; cross-tree or several top-level memories form a new top-level DOMAIN node. Give the keys "
        "(2+) and a summary for the new parent. OR pass a session and omit keys to auto-consolidate all "
        "of that session's top-level memories. Returns the new parent key.")
    parameters = {"type": "object", "properties": {
        "keys": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"}, "label": {"type": "string"},
        "session": {"type": "string", "description": "auto-consolidate this session's top-level memories when keys omitted"}},
        "required": ["summary"]}

    def __init__(self, root=None):
        pass

    def run(self, summary: str = "", keys=None, label: str = "", session: str = "") -> ToolResult:
        if not summary.strip():
            return ToolResult(ok=False, output="", error="summary is required")
        try:
            store = _store()
            keys = list(keys or [])
            if not keys and session:                # auto-consolidate a session's top-level memories
                keys = [c["key"] for c in store.index(session)]
            if len(keys) < 2:
                return ToolResult(ok=False, output="", error="need at least two memories to consolidate")
            card = store.consolidate(keys, summary, label, session)
            return ToolResult(ok=True, output=f"consolidated {len(keys)} into {card['key']} ({card['label']})")
        except (ValueError, KeyError) as e:
            return ToolResult(ok=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"consolidate failed: {e}")
