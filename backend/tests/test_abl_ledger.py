import numpy as np
import pytest

from crucible.abliteration.ledger import EditLedger


def test_record_log_and_parents():
    led = EditLedger()
    c1 = led.record("inplace", {"layers": [1]}, "edit 1", {"refusal": 0.9}, {"w": np.zeros(3)})
    c2 = led.record("inplace", {"layers": [2]}, "edit 2", {"refusal": 0.0}, {"w": np.ones(3)})
    assert c1["id"] == "c1" and c1["parent"] is None
    assert c2["parent"] == "c1"
    assert [c["id"] for c in led.log()] == ["c1", "c2"]
    assert led.commits[0]["tensors"] == ["w"]


def test_deltas_for_revert():
    led = EditLedger()
    original = np.array([1.0, 2.0, 3.0])
    led.record("inplace", {}, "e", {}, {"w": original})
    assert np.allclose(led.get_deltas("c1")["w"], original)
    with pytest.raises(KeyError):
        led.get_deltas("nope")


def test_branch():
    led = EditLedger()
    led.set_branch("experiment")
    c = led.record("inplace", {}, "e", {}, {})
    assert c["branch"] == "experiment"


def _led_with_parts():
    led = EditLedger()
    # edit the language model, then the vision encoder, then the language model again
    led.record("inplace", {}, "text edit 1", {}, {"model.layers.0.mlp.down_proj.weight": np.zeros(2)})
    led.record("inplace", {}, "vision edit", {}, {"vision_tower.blocks.0.attn.o_proj.weight": np.ones(2)})
    led.record("inplace", {}, "text edit 2", {}, {"model.layers.5.mlp.down_proj.weight": np.zeros(2)})
    return led


def test_commit_tags_the_part():
    led = _led_with_parts()
    assert led.commits[0]["parts"] == ["language_model"]
    assert led.commits[1]["parts"] == ["vision_encoder"]


def test_log_filters_by_part():
    led = _led_with_parts()
    assert [c["id"] for c in led.log("language_model")] == ["c1", "c3"]
    assert [c["id"] for c in led.log("vision_encoder")] == ["c2"]
    assert len(led.log()) == 3


def test_lineage_is_per_part():
    lin = {p["part"]: p for p in _led_with_parts().lineage()}
    assert lin["language_model"]["n_versions"] == 2 and lin["language_model"]["latest"] == "c3"
    assert lin["vision_encoder"]["n_versions"] == 1 and lin["vision_encoder"]["latest"] == "c2"


def test_latest_and_deltas_for_part():
    led = _led_with_parts()
    assert led.latest_for_part("language_model")["id"] == "c3"
    assert led.latest_for_part("audio_encoder") is None
    # per-part deltas: c3 only touched a language_model tensor
    d = led.deltas_for_part("c3", "language_model")
    assert list(d) == ["model.layers.5.mlp.down_proj.weight"]
    assert led.deltas_for_part("c3", "vision_encoder") == {}   # nothing of that part in c3
