"""Configurable ordering for memory cards + context turns — one reusable sorter."""
from crucible.sorting import SORTS, sort_items


ITEMS = [
    {"key": "m-0001", "label": "beta", "size": 3, "priority": 1, "links": [{"to": "m-0002"}]},
    {"key": "m-0002", "label": "alpha", "size": 9, "priority": 5, "links": []},
    {"key": "m-0003", "label": "gamma", "size": 1, "priority": 0, "links": [{"to": "x"}, {"to": "y"}]},
]


def test_recency_newest_first():
    assert [i["key"] for i in sort_items(ITEMS, "recency")] == ["m-0003", "m-0002", "m-0001"]


def test_oldest_ascends():
    assert [i["key"] for i in sort_items(ITEMS, "oldest")] == ["m-0001", "m-0002", "m-0003"]


def test_priority_then_recency():
    assert [i["key"] for i in sort_items(ITEMS, "priority")][0] == "m-0002"   # priority 5


def test_size_biggest_first():
    assert [i["size"] for i in sort_items(ITEMS, "size")] == [9, 3, 1]


def test_degree_by_link_count():
    assert sort_items(ITEMS, "degree")[0]["key"] == "m-0003"   # 2 links


def test_relevance_uses_score():
    scored = [{"key": "a", "score": 0.2}, {"key": "b", "score": 0.9}, {"key": "c", "score": 0.5}]
    assert [i["key"] for i in sort_items(scored, "relevance")] == ["b", "c", "a"]


def test_label_ascending():
    assert [i["label"] for i in sort_items(ITEMS, "label")] == ["alpha", "beta", "gamma"]


def test_direction_override_and_unknown_key():
    assert [i["size"] for i in sort_items(ITEMS, "size", descending=False)] == [1, 3, 9]
    assert sort_items(ITEMS, "nonsense") == ITEMS       # untouched
    assert "priority" in SORTS and "recency" in SORTS
