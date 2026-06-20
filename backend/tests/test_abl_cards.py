from crucible.abliteration.cards import build_model_card, reproducibility_hash


def test_hash_is_stable_and_order_independent():
    assert reproducibility_hash({"a": 1, "b": 2}) == reproducibility_hash({"b": 2, "a": 1})
    assert len(reproducibility_hash({"a": 1})) == 16


def test_card_has_expected_fields():
    card = build_model_card("glm-4-32b", "glm-4-32b-abl", "abliteration", 20, 1.0,
                            {"harmful": 32, "harmless": 32}, 5120)
    assert card["base_id"] == "glm-4-32b" and card["layer"] == 20
    assert card["eval_delta"] is None and len(card["repro_hash"]) == 16
