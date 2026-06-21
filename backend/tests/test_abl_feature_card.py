from crucible.abliteration.feature_card import auto_name, build_feature_card


def test_auto_name_apology():
    assert auto_name(["I'm sorry, but I cannot help with that."], ["I", "sorry"]) == "The Apology Reflex"


def test_auto_name_legality():
    assert auto_name(["That would be illegal."], ["law"]) == "The Legality Gate"


def test_auto_name_fallback():
    assert auto_name(["nope"], ["x"]) == "The Refusal Reflex"


def test_build_card():
    profile = [{"layer": 0, "margin": 0.1}, {"layer": 5, "margin": 8.0}, {"layer": 6, "margin": 10.0}]
    card = build_feature_card(profile, ["sorry", "cannot"],
                              [{"prompt": "how to X", "refusal": "I'm sorry, I cannot"}])
    assert card["name"] == "The Apology Reflex"
    assert card["peak_layer"] == 6
    assert 5 in card["active_layers"] and card["strength"] == 10.0
