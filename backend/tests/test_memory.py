"""Crystallized memory: git-versioned store with summary-passthrough retrieval and recursive
re-crystallization. Uses a real tmp git repo, so the version history is exercised for real."""
import subprocess

import pytest

from crucible.memory import MemoryStore, derive_label


def _convo(n, tag="t"):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"{tag}{i}"} for i in range(n)]


def test_derive_label():
    assert derive_label("Set up the abliteration pipeline for qwen and verify refusal drop") \
        == "Set up the abliteration pipeline for"
    assert derive_label("") == "memory"


def test_crystallize_creates_leaf_and_commits(tmp_path):
    s = MemoryStore(tmp_path / "mem")
    node = s.crystallize(_convo(6), "summary of the first stretch", session="sess1")
    assert node["key"] == "m-0001" and node["kind"] == "leaf" and node["n_messages"] == 6
    assert node["label"] == "summary of the first stretch"
    # it was committed to git (real version history)
    assert s.versioned is True and node["ref"]
    log = subprocess.run(["git", "-C", str(tmp_path / "mem"), "log", "--oneline"],
                         capture_output=True, text=True).stdout
    assert "crystallize m-0001" in log


def test_index_is_summary_passthrough(tmp_path):
    s = MemoryStore(tmp_path / "mem")
    s.crystallize(_convo(4), "first summary", session="a")
    s.crystallize(_convo(8), "second summary", session="a")
    idx = s.index()
    assert [c["key"] for c in idx] == ["m-0001", "m-0002"]
    # cards carry label + summary + size, but NOT the message bodies (cheap to scan)
    assert idx[0]["summary"] == "first summary" and idx[1]["size"] == 8
    assert "messages" not in idx[0]


def test_index_filters_by_session(tmp_path):
    s = MemoryStore(tmp_path / "mem")
    s.crystallize(_convo(2), "s a", session="a")
    s.crystallize(_convo(2), "s b", session="b")
    assert [c["key"] for c in s.index(session="a")] == ["m-0001"]
    assert [c["key"] for c in s.index(session="b")] == ["m-0002"]


def test_read_leaf_returns_messages(tmp_path):
    s = MemoryStore(tmp_path / "mem")
    s.crystallize(_convo(5, "x"), "sum")
    r = s.read("m-0001")
    assert r["kind"] == "leaf" and len(r["messages"]) == 5 and r["messages"][0]["content"] == "x0"


def test_recrystallize_reorganizes_into_subchunks(tmp_path):
    s = MemoryStore(tmp_path / "mem")
    s.crystallize(_convo(10), "the whole thing")
    res = s.recrystallize("m-0001", [
        {"label": "setup", "summary": "environment + model load", "messages": _convo(4, "a")},
        {"label": "editing", "summary": "abliteration + verify", "messages": _convo(6, "b")},
    ])
    assert res["kind"] == "chunked" and len(res["children"]) == 2

    # parent is now chunked: reading it returns child SUMMARY CARDS, not the bulk messages
    parent = s.read("m-0001")
    assert parent["kind"] == "chunked" and "messages" not in parent
    assert [c["label"] for c in parent["children"]] == ["setup", "editing"]
    assert all("messages" not in c for c in parent["children"])   # drill-down is summary-first

    # drilling into a child leaf returns its messages
    child = s.read(res["children"][0])
    assert child["kind"] == "leaf" and len(child["messages"]) == 4

    # parent dropped from the top-level index count of messages -> now counts children
    assert s.index()[0]["size"] == 2 and s.index()[0]["kind"] == "chunked"


def test_recursive_recrystallize_builds_a_tree(tmp_path):
    s = MemoryStore(tmp_path / "mem")
    s.crystallize(_convo(12), "root")
    s.recrystallize("m-0001", [{"label": "half1", "summary": "s1", "messages": _convo(6)},
                               {"label": "half2", "summary": "s2", "messages": _convo(6)}])
    # further crystallize a child (m-0002) into finer subchunks
    s.recrystallize("m-0002", [{"label": "q1", "summary": "q1s", "messages": _convo(3)},
                               {"label": "q2", "summary": "q2s", "messages": _convo(3)}])
    tree = s.tree()
    assert len(tree) == 1 and tree[0]["key"] == "m-0001"
    kids = tree[0]["children"]
    assert kids[0]["key"] == "m-0002" and [g["label"] for g in kids[0]["children"]] == ["q1", "q2"]


def test_recrystallize_empty_raises(tmp_path):
    s = MemoryStore(tmp_path / "mem")
    s.crystallize(_convo(2), "x")
    with pytest.raises(ValueError):
        s.recrystallize("m-0001", [])


def test_consolidate_top_level_into_domain_node(tmp_path):
    # two separate top-level memories -> a NEW top-level domain parent (top-level pruning)
    s = MemoryStore(tmp_path / "mem")
    s.crystallize(_convo(4), "abliteration work", session="s")
    s.crystallize(_convo(4), "benchmarking work", session="s")
    dom = s.consolidate(["m-0001", "m-0002"], "all the model-lab work", label="model lab")
    assert dom["kind"] == "chunked" and dom["label"] == "model lab"
    # the two originals are no longer top-level; the domain node is the only top-level entry
    top = s.index()
    assert [c["key"] for c in top] == [dom["key"]] and top[0]["size"] == 2
    # reading the domain node drills into the two originals (summary-first)
    kids = s.read(dom["key"])["children"]
    assert {c["summary"] for c in kids} == {"abliteration work", "benchmarking work"}


def test_consolidate_siblings_bubbles_to_shared_parent(tmp_path):
    # within a tree: consolidating siblings files them UNDER their shared parent, not at top level
    s = MemoryStore(tmp_path / "mem")
    s.crystallize(_convo(12), "root")
    s.recrystallize("m-0001", [{"label": "a", "summary": "sa", "messages": _convo(3)},
                               {"label": "b", "summary": "sb", "messages": _convo(3)},
                               {"label": "c", "summary": "sc", "messages": _convo(3)}])
    # children are m-0002 (a), m-0003 (b), m-0004 (c) under root m-0001
    grp = s.consolidate(["m-0002", "m-0003"], "a+b grouped")
    # the group node's parent is the shared parent (root), NOT top level
    root = s.read("m-0001")
    assert root["kind"] == "chunked"
    child_keys = [c["key"] for c in root["children"]]
    assert grp["key"] in child_keys and "m-0002" not in child_keys and "m-0004" in child_keys
    # top level is still just the root (no clutter)
    assert [c["key"] for c in s.index()] == ["m-0001"]
    # the group holds the two consolidated siblings
    assert {c["key"] for c in s.read(grp["key"])["children"]} == {"m-0002", "m-0003"}


def test_consolidate_needs_two_and_rejects_ancestor(tmp_path):
    s = MemoryStore(tmp_path / "mem")
    s.crystallize(_convo(6), "root")
    s.recrystallize("m-0001", [{"label": "x", "summary": "sx", "messages": _convo(6)}])
    with pytest.raises(ValueError):
        s.consolidate(["m-0001"], "only one")                 # < 2
    with pytest.raises(ValueError):
        s.consolidate(["m-0001", "m-0002"], "parent+child")   # m-0001 is an ancestor of m-0002


def test_persists_across_instances(tmp_path):
    root = tmp_path / "mem"
    MemoryStore(root).crystallize(_convo(3), "durable summary", session="s")
    # a fresh store over the same dir sees the memory (source of truth is on disk, versioned in git)
    again = MemoryStore(root)
    assert again.read("m-0001")["summary"] == "durable summary"
    assert again.index()[0]["key"] == "m-0001"


def test_search_lexical_ranks_by_relevance(tmp_path):
    s = MemoryStore(tmp_path / "mem")
    s.crystallize(_convo(2), "abliteration removed the refusal direction", label="uncensor")
    s.crystallize(_convo(2), "quantized the weights to run faster locally", label="quantize")
    res = s.search("refusal direction", embedder=None)
    assert res["method"] == "lexical"
    assert res["matches"][0]["key"] == "m-0001" and res["matches"][0]["score"] > 0
    assert s.search("xylophone", embedder=None)["matches"] == []     # no keyword -> nothing


def test_search_semantic_with_embedder(tmp_path):
    s = MemoryStore(tmp_path / "mem")
    s.crystallize(_convo(2), "image safety gate in the vision encoder", label="vision")
    s.crystallize(_convo(2), "text refusal in the language model", label="text")

    def embed(texts):
        out = []
        for t in texts:
            out.append([1 if "vision" in t.lower() or "image" in t.lower() else 0,
                        1 if "text" in t.lower() or "language" in t.lower() else 0])
        return out

    res = s.search("picture", embedder=embed)   # 'picture' won't match lexically, but embed maps it? no
    # the query embeds to [0,0] here (no keyword), so semantic returns nothing positive — honest
    assert res["method"] == "semantic"
    # a query the embedder DOES place in vision-space ranks the vision memory first
    res2 = s.search("vision image", embedder=embed)
    assert res2["matches"][0]["key"] == "m-0001"
