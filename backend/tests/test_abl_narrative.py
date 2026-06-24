from crucible.abliteration.narrative import plain_diagnosis, translate

JARGON = ("residual", "rank-1", "rank 1", "orthogonal", "projection", "logit",
          "o_proj", "down_proj", "matrix", "tensor", "softmax", "eigen")

SURGICAL_DIAG = {
    "best_layer": 21,
    "layer_profile": [{"layer": 21, "margin": 4.5}, {"layer": 10, "margin": 1.0}],
    "mean_removed_fraction": 0.02,
    "surgical": True,
    "heaviest_component": "o_proj",
    "collateral_risk": "low",
}

ENTANGLED_DIAG = {
    "best_layer": 8,
    "layer_profile": [{"layer": 8, "margin": 1.2}],
    "mean_removed_fraction": 0.18,
    "surgical": False,
    "heaviest_component": "down_proj",
    "collateral_risk": "elevated",
}


def _no_jargon(narr: dict) -> None:
    blob = " ".join(str(narr[k]) for k in ("headline", "locate", "evidence", "target", "repair", "risk"))
    blob += " " + " ".join(narr["steps"])
    low = blob.lower()
    for term in JARGON:
        assert term not in low, f"jargon leaked: {term!r}"


def test_surgical_narrative_is_plain_and_actionable():
    n = plain_diagnosis(SURGICAL_DIAG)
    _no_jargon(n)
    assert "21" in n["headline"]
    assert "cleanly" in n["headline"]
    assert n["confidence"] == "high"
    assert len(n["steps"]) >= 3
    assert "low collateral" in n["risk"]


def test_entangled_narrative_warns():
    n = plain_diagnosis(ENTANGLED_DIAG)
    _no_jargon(n)
    assert "carefully" in n["headline"]
    assert "ntangled" in n["risk"]            # "Entangled"
    assert n["confidence"] in ("low", "moderate")


def test_causal_evidence_is_woven_in():
    n = plain_diagnosis(SURGICAL_DIAG, causal={"peak_layer": 21, "peak_restoration": 0.85})
    _no_jargon(n)
    assert "cause" in n["evidence"].lower()
    assert "85%" in n["evidence"]


def test_weak_causal_says_widen_search():
    n = plain_diagnosis(SURGICAL_DIAG, causal={"peak_layer": 3, "peak_restoration": 0.05})
    assert "widen" in n["evidence"].lower()


def test_multipath_warning_and_extra_step():
    n = plain_diagnosis(SURGICAL_DIAG, multidir={"n_directions": 3, "sticky_fraction": 0.4})
    _no_jargon(n)
    assert "secondary path" in n["risk"]
    assert any("secondary" in s for s in n["steps"])


def test_translate_without_translator_flags_untranslated():
    n = plain_diagnosis(SURGICAL_DIAG)
    es = translate(n, "es")
    assert es["language"] == "es" and es["translated"] is False
    assert es["headline"] == n["headline"]    # unchanged English


def test_translate_with_translator_applies_to_all_prose():
    n = plain_diagnosis(SURGICAL_DIAG)
    es = translate(n, "es", translator=lambda text, lang: f"[{lang}] {text}")
    assert es["translated"] is True
    assert es["headline"].startswith("[es] ")
    assert all(s.startswith("[es] ") for s in es["steps"])


def test_translate_english_is_noop():
    n = plain_diagnosis(SURGICAL_DIAG)
    assert translate(n, "en")["headline"] == n["headline"]
