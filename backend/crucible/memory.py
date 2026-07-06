from __future__ import annotations
# Crystallized memory. Unlike assistants that DISCARD old turns on compaction, Crucible keeps every
# compacted context as a durable, git-VERSIONED "crystallized memory": a node with a short LABEL, a
# SUMMARY (the cheap passthrough a model reads first to decide whether to open the full thing), and
# either the full messages (a leaf) or organized sub-memories (chunked). Memories can be RE-
# crystallized — reorganized into better-labelled/summarized subchunks — so an agent drills down
# through summaries instead of ever re-reading the whole context. Every mutation is a git commit, so
# the history of the thread AND of how it was reorganized is inspectable and recoverable.
#
# This lives under the user's private data dir (~/.crucible/memory) — a SEPARATE local git repo,
# never the public project repo. The tree logic is pure and unit-tested; git is best-effort
# persistence (the JSON files are the source of truth; commits add the version history on top).
import json
import os
import subprocess
from pathlib import Path
from typing import Optional


def derive_label(summary: str, n_words: int = 6) -> str:
    """A short human/AI-readable label from a summary (first few meaningful words). The AI can
    replace it with a better one via re-crystallization; this is just a sensible default."""
    words = [w for w in (summary or "").replace("\n", " ").split(" ") if w.strip()]
    label = " ".join(words[:n_words]).strip(" .,:;—-")
    return label or "memory"


class MemoryStore:
    """Git-versioned store of crystallized memories. Keys are globally unique (m-0001, …). A node is
    a leaf (holds messages) or chunked (holds child memory keys). index() is the summary passthrough;
    read() opens one node; recrystallize() reorganizes a node into labelled/summarized subchunks."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.mem_dir = self.root / "memories"
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.versioned = self._init_git()

    # --- git (best-effort) -----------------------------------------------------------------
    def _git(self, *args: str, check: bool = True) -> Optional[subprocess.CompletedProcess]:
        try:
            return subprocess.run(["git", "-C", str(self.root), *args],
                                  capture_output=True, text=True, check=check)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _init_git(self) -> bool:
        fresh = not (self.root / ".git").exists()
        if fresh and self._git("init", "-q") is None:
            return False
        self._set_identity()
        return True

    def _set_identity(self) -> None:
        """Bind an EXPLICIT LOCAL git identity so commits are never blocked by a machine ghid-guard
        and never silently inherit a global (real) identity — an opsec safeguard. Defaults to the
        sanctioned pq-cybarg pseudonym for this project (override via CRUCIBLE_GIT_NAME /
        CRUCIBLE_GIT_EMAIL). Any identity already bound (e.g. by `ghid`) is left untouched."""
        got = self._git("config", "--local", "user.email", check=False)
        if got is not None and got.returncode == 0 and got.stdout.strip():
            return
        name = os.environ.get("CRUCIBLE_GIT_NAME", "pq-cybarg")
        email = os.environ.get("CRUCIBLE_GIT_EMAIL", "resistant@tuta.com")
        self._git("config", "--local", "user.name", name, check=False)
        self._git("config", "--local", "user.email", email, check=False)

    def _commit(self, rel_paths: list[str], subject: str) -> Optional[str]:
        if not self.versioned:
            return None
        self._git("add", *rel_paths, check=False)
        r = self._git("commit", "-q", "-m", subject, check=False)   # uses the bound local identity
        if r is None or r.returncode != 0:
            return None
        ref = self._git("rev-parse", "--short", "HEAD", check=False)
        return ref.stdout.strip() if ref and ref.returncode == 0 else None

    # --- node io ---------------------------------------------------------------------------
    def _path(self, key: str) -> Path:
        return self.mem_dir / f"{key}.json"

    def _next_key(self) -> str:
        return f"m-{len(list(self.mem_dir.glob('m-*.json'))) + 1:04d}"

    def _write(self, node: dict) -> str:
        p = self._path(node["key"])
        p.write_text(json.dumps(node, indent=2, ensure_ascii=False))
        return str(p.relative_to(self.root))

    def _load(self, key: str) -> dict:
        p = self._path(key)
        if not p.exists():
            raise KeyError(key)
        return json.loads(p.read_text())

    # --- public API ------------------------------------------------------------------------
    def crystallize(self, messages: list[dict], summary: str, label: str = "",
                    session: str = "", stats: Optional[dict] = None,
                    parent: Optional[str] = None) -> dict:
        """Store a context as a crystallized LEAF memory (label + summary + full messages) and commit
        it. Returns the node (with its git ref). This is what compaction calls to keep the old turns."""
        key = self._next_key()
        node = {"key": key, "kind": "leaf", "label": label or derive_label(summary),
                "summary": summary or "", "session": session, "parent": parent,
                "n_messages": len(messages), "stats": stats or {}, "messages": list(messages)}
        rel = self._write(node)
        node["ref"] = self._commit([rel], f"crystallize {key}: {node['label']} ({len(messages)} msgs)")
        return node

    def index(self, session: Optional[str] = None, sort: str = "recency") -> list[dict]:
        """The SUMMARY PASSTHROUGH: every TOP-LEVEL memory as a cheap card — for a model to scan
        before deciding which to open. Optionally filter by session and ORDER by a configurable key
        (recency / priority / size / degree / label) so recall is prioritized, not just insertion-order."""
        from crucible.sorting import sort_items
        out = []
        for p in sorted(self.mem_dir.glob("m-*.json")):
            n = json.loads(p.read_text())
            if n.get("parent") is not None:
                continue
            if session is not None and n.get("session") != session:
                continue
            out.append(self._card(n))
        return sort_items(out, sort)

    def _card(self, n: dict) -> dict:
        size = len(n.get("children", [])) if n.get("kind") == "chunked" else n.get("n_messages", 0)
        return {"key": n["key"], "label": n.get("label", ""), "summary": n.get("summary", ""),
                "kind": n.get("kind", "leaf"), "session": n.get("session", ""),
                "size": size, "ref": n.get("ref"),
                "priority": int(n.get("priority", 0)), "degree": len(n.get("links", []) or [])}

    def read(self, key: str) -> dict:
        """Open one memory. A LEAF returns its full messages; a CHUNKED node returns its children's
        summary cards (drill down further) — so you never load more context than you asked for."""
        n = self._load(key)
        card = self._card(n)
        if n.get("kind") == "chunked":
            card["children"] = [self._card(self._load(c)) for c in n.get("children", [])]
        else:
            card["messages"] = n.get("messages", [])
            card["stats"] = n.get("stats", {})
        return card

    def recrystallize(self, key: str, subchunks: list[dict]) -> dict:
        """Reorganize a leaf memory into better-labelled/summarized SUBCHUNKS (same design, one level
        down). subchunks: [{label, summary, messages}]. Each becomes a child leaf; the parent becomes
        a chunked node holding their keys (its bulky messages move into the children). Versioned."""
        n = self._load(key)
        if not subchunks:
            raise ValueError("recrystallize needs at least one subchunk")
        child_keys, rels = [], []
        for sc in subchunks:
            ck = self._next_key()
            child = {"key": ck, "kind": "leaf", "label": sc.get("label") or derive_label(sc.get("summary", "")),
                     "summary": sc.get("summary", ""), "session": n.get("session", ""), "parent": key,
                     "n_messages": len(sc.get("messages", [])), "stats": {}, "messages": list(sc.get("messages", []))}
            rels.append(self._write(child))
            child_keys.append(ck)
        n["kind"] = "chunked"
        n["children"] = child_keys
        n.pop("messages", None)              # bulk moves into the children; parent keeps label+summary
        n["n_messages"] = sum(len(sc.get("messages", [])) for sc in subchunks)
        rels.append(self._write(n))
        ref = self._commit(rels, f"recrystallize {key} -> {len(child_keys)} subchunks")
        n["ref"] = ref
        return {"key": key, "children": child_keys, "ref": ref, "kind": "chunked"}

    def _ancestors(self, key: str) -> list[str]:
        """[key, parent, grandparent, …] up to a top-level root (cycle-guarded)."""
        chain, cur, seen = [], key, set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            try:
                cur = self._load(cur).get("parent")
            except KeyError:
                break
        return chain

    def _lca(self, keys: list[str]) -> Optional[str]:
        """Highest shared parent (lowest common ancestor) of the keys, excluding the keys
        themselves — None when they span different trees / are all top-level (=> consolidate at top)."""
        chains = [self._ancestors(k) for k in keys]
        common = set(chains[0])
        for c in chains[1:]:
            common &= set(c)
        common -= set(keys)
        for k in chains[0]:          # deepest-first order preserved from the first chain
            if k in common:
                return k
        return None

    def consolidate(self, keys: list[str], summary: str, label: str = "",
                    session: str = "") -> dict:
        """The inverse of recrystallize: file a SET of memories under a NEW chunked parent (label +
        summary). Placement follows the highest shared parent — siblings consolidate UNDER their
        shared parent (no top-level clutter); memories from different trees (or several top-level
        memories) consolidate into a NEW TOP-LEVEL domain node (top-level pruning). Versioned."""
        keys = list(dict.fromkeys(keys))
        if len(keys) < 2:
            raise ValueError("consolidate needs at least two memories")
        nodes = {k: self._load(k) for k in keys}          # KeyError if any missing
        for k in keys:                                     # can't consolidate a node with its own ancestor
            if (set(self._ancestors(k)) - {k}) & set(keys):
                raise ValueError("cannot consolidate a memory together with its own ancestor")
        target_parent = self._lca(keys)
        new_key = self._next_key()
        new_node = {"key": new_key, "kind": "chunked", "label": label or derive_label(summary),
                    "summary": summary or "", "session": session or nodes[keys[0]].get("session", ""),
                    "parent": target_parent, "children": list(keys),
                    "n_messages": sum(n.get("n_messages", 0) for n in nodes.values()), "stats": {}}
        rels = []
        for k, n in nodes.items():                         # reparent each selected memory under the new node
            old_parent = n.get("parent")
            n["parent"] = new_key
            rels.append(self._write(n))
            if old_parent is not None and old_parent != target_parent:
                op = self._load(old_parent)
                op["children"] = [c for c in op.get("children", []) if c != k]
                rels.append(self._write(op))
        rels.append(self._write(new_node))
        if target_parent is not None:                      # splice the new node into its shared parent
            tp = self._load(target_parent)
            ch = [c for c in tp.get("children", []) if c not in keys]
            ch.append(new_key)
            tp["children"] = ch
            rels.append(self._write(tp))
        new_node["ref"] = self._commit(rels, f"consolidate {len(keys)} -> {new_key}: {new_node['label']}")
        return self._card(new_node)

    def tree(self, session: Optional[str] = None) -> list[dict]:
        """The full nested tree of summary cards (top-level memories with recursive children) — for
        a browser UI. Reads only labels/summaries/keys, not message bodies."""
        def node(key: str) -> dict:
            n = self._load(key)
            card = self._card(n)
            if n.get("kind") == "chunked":
                card["children"] = [node(c) for c in n.get("children", [])]
            return card
        return [node(c["key"]) for c in self.index(session)]

    def search(self, query: str, embedder=None, k: int = 5, session: Optional[str] = None,
               sort: str = "relevance") -> dict:
        """Relevance search over ALL memories (leaves + chunked, at any depth) by their label +
        summary. With an embedder it's semantic (cosine); without one it's lexical (BM25) — the
        method is reported honestly. Results are ordered by `sort` (default relevance; or priority /
        recency / … to blend ranking with agent priority). Returns {method, matches:[card + score]}."""
        from crucible.rag import rank
        from crucible.sorting import sort_items
        cards = []
        for p in sorted(self.mem_dir.glob("m-*.json")):
            n = json.loads(p.read_text())
            if session is not None and n.get("session") != session:
                continue
            cards.append(self._card(n))
        if not cards:
            return {"method": "semantic" if embedder else "lexical", "matches": []}
        docs = [f"{c['label']} {c['summary']}" for c in cards]
        r = rank(query, docs, k=k, embedder=embedder)
        matches = [{**cards[res["index"]], "score": res["score"]} for res in r["results"]]
        if sort != "relevance":
            matches = sort_items(matches, sort)
        return {"method": r["method"], "matches": matches}

    # --- DAG layer: priority + typed cross-links (a memory graph, not just a tree) ----------------
    # The parent/children hierarchy is one acyclic view; on top of it, arbitrary DIRECTED TYPED LINKS
    # (relates/refines/contradicts/…) turn memory into a graph. Links may point anywhere — even
    # forming cycles — so it's "conditionally semicyclic"; every traversal here is visited-guarded so
    # it always terminates. Priority lets an agent weight what matters for cheap prioritized recall.

    def set_priority(self, key: str, priority: int) -> dict:
        """Weight a memory (higher = recalled first when sorting by priority)."""
        n = self._load(key)
        n["priority"] = int(priority)
        rel = self._write(n)
        self._commit([rel], f"priority {key} = {priority}")
        return self._card(n)

    def link(self, src: str, dst: str, type: str = "relates") -> dict:
        """Add a directed typed edge src -> dst (semicyclic allowed). Both must exist; duplicates of
        the same (dst,type) are ignored. This is what makes memory a DAG/graph rather than a tree."""
        if src == dst:
            raise ValueError("a memory cannot link to itself")
        self._load(dst)                       # KeyError if the target is missing
        n = self._load(src)
        links = n.setdefault("links", [])
        if not any(e.get("to") == dst and e.get("type") == type for e in links):
            links.append({"to": dst, "type": type})
            rel = self._write(n)
            self._commit([rel], f"link {src} -{type}-> {dst}")
        return {"from": src, "to": dst, "type": type}

    def unlink(self, src: str, dst: str) -> dict:
        """Remove all directed edges src -> dst."""
        n = self._load(src)
        before = len(n.get("links", []) or [])
        n["links"] = [e for e in n.get("links", []) or [] if e.get("to") != dst]
        if len(n["links"]) != before:
            rel = self._write(n)
            self._commit([rel], f"unlink {src} -> {dst}")
        return {"from": src, "to": dst, "removed": before - len(n["links"])}

    def graph(self, session: Optional[str] = None) -> dict:
        """The full memory GRAPH: every node as a card, plus edges — 'parent' edges (the tree) and
        the directed typed cross-'link' edges (the semicyclic layer). For a graph view + traversal."""
        nodes, edges = [], []
        for p in sorted(self.mem_dir.glob("m-*.json")):
            n = json.loads(p.read_text())
            if session is not None and n.get("session") != session:
                continue
            nodes.append({**self._card(n), "parent": n.get("parent")})
            for c in n.get("children", []) or []:
                edges.append({"from": n["key"], "to": c, "type": "child", "kind": "parent"})
            for e in n.get("links", []) or []:
                edges.append({"from": n["key"], "to": e["to"], "type": e.get("type", "relates"), "kind": "link"})
        return {"nodes": nodes, "edges": edges, "n_nodes": len(nodes), "n_edges": len(edges)}
