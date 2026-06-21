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
