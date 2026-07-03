from crucible.routing import AUTO_ALIASES, choose_model, routing_explain


def test_honors_available_requested_model():
    assert choose_model("b", ["a", "b", "c"]) == "b"


def test_auto_alias_picks_nearest():
    for alias in ("", "auto", "crucible", "nearest"):
        assert alias in AUTO_ALIASES
        assert choose_model(alias, ["a", "b"]) == "a"


def test_requested_unavailable_falls_to_preferences():
    up = {"a", "c"}
    chosen = choose_model("b", ["a", "b", "c"], preferences=["c", "a"],
                          is_available=lambda i: i in up)
    assert chosen == "c"                       # b down -> first available preference


def test_no_preferences_falls_to_nearest_available():
    up = {"c"}
    chosen = choose_model("b", ["a", "b", "c"], is_available=lambda i: i in up)
    assert chosen == "c"                       # only c answers


def test_returns_none_when_nothing_available():
    assert choose_model("b", ["a", "b"], is_available=lambda i: False) is None


def test_unknown_requested_ignored():
    assert choose_model("ghost", ["a", "b"]) == "a"


def test_explain_messages():
    assert "requested model 'b'" in routing_explain("b", "b", None)
    assert "preference" in routing_explain("b", "c", ["c"])
    assert "nearest" in routing_explain("b", "a", None)
    assert routing_explain("b", None, None) == "no model available"
