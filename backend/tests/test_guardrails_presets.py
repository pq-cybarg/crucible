import pytest

from crucible.guardrails.presets import PRESETS, get_preset


def test_three_presets_ordered_by_intensity():
    ids = [p.id for p in PRESETS]
    assert ids == ["unrestricted", "balanced", "strict"]
    assert get_preset("unrestricted").intensity == 0
    assert get_preset("strict").intensity == 100


def test_unrestricted_prompt_is_empty_or_minimal():
    assert len(get_preset("unrestricted").system_prompt) <= len(get_preset("strict").system_prompt)


def test_missing_preset_raises():
    with pytest.raises(KeyError):
        get_preset("nope")
