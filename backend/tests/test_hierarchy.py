"""Agent hierarchy profiles: per-layer worker + lighter communicator model pairs, multi-layer,
multi-profile, persisted. Plus the communicator `relay` step (injected summarizer)."""
from crucible.hierarchy import HierarchyProfile, Layer, ProfileStore, relay


def test_layer_at_depth_reuses_last():
    p = HierarchyProfile("p", [Layer("big", "small"), Layer("mid", "tiny")])
    assert p.at(0).worker == "big" and p.at(0).communicator == "small"
    assert p.at(1).worker == "mid"
    assert p.at(5).worker == "mid"          # deeper than defined -> last layer
    assert HierarchyProfile("e", []).at(3) == Layer()   # empty -> default


def test_roundtrip_dict():
    p = HierarchyProfile("x", [Layer("a", "b")])
    assert HierarchyProfile.from_dict(p.to_dict()).layers[0].communicator == "b"


def test_relay_compresses_via_communicator():
    calls = []
    def comm(prompt: str) -> str:
        calls.append(prompt)
        return "TIGHT SUMMARY"
    assert relay("a very long child result " * 50, comm) == "TIGHT SUMMARY"
    assert "communicator" in calls[0].lower()
    # no communicator -> passthrough; empty result -> unchanged
    assert relay("keep me", None) == "keep me"
    assert relay("", comm) == ""


def test_relay_survives_a_flaky_communicator():
    def boom(_): raise RuntimeError("down")
    assert relay("child answer", boom) == "child answer"    # never loses the answer


def test_profile_store_crud(tmp_path):
    store = ProfileStore(tmp_path / "profiles.json")
    names = {p["name"] for p in store.list()}
    assert "flat" in names                                  # seeded defaults
    store.save(HierarchyProfile("research", [Layer("opus", "haiku"), Layer("sonnet", "haiku")]))
    got = store.get("research")
    assert got.at(0).worker == "opus" and got.at(1).communicator == "haiku"
    assert store.delete("research") is True and store.delete("nope") is False


def test_profile_store_multiprofile_persists(tmp_path):
    path = tmp_path / "p.json"
    ProfileStore(path).save(HierarchyProfile("a", [Layer("m1", "c1")]))
    ProfileStore(path).save(HierarchyProfile("b", [Layer("m2", "c2")]))
    fresh = ProfileStore(path)
    names = {p["name"] for p in fresh.list()}
    assert {"a", "b"} <= names and fresh.get("b").at(0).worker == "m2"
