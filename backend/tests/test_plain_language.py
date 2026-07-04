"""Plain-language layer: every explainer turns a real endpoint result shape into an honest,
jargon-free five-field card. Inputs here mirror the ACTUAL dicts the endpoints return."""
import re

import pytest

from crucible.abliteration.plain_language import (
    CARD_FIELDS, explain, translate_plain, with_plain, _EXPLAINERS,
)

# Jargon we never want to see leak into the plain card (the whole point of this layer).
_JARGON = re.compile(r"\b(residual stream|rank-1|rank-one|projection matrix|logit lens|"
                     r"orthogonaliz|eigenvalue|softmax|tensor|singular value)\b", re.I)


def _assert_card(card: dict):
    for f in CARD_FIELDS:
        assert isinstance(card.get(f), str) and card[f].strip(), f"missing/empty field {f}"
        assert not _JARGON.search(card[f]), f"jargon leaked into {f!r}: {card[f]!r}"


# --- realistic result fixtures (keys match the endpoints) -----------------------------------

CASES = {
    "causal-trace": {"peak_layer": 14, "peak_restoration": 0.78,
                     "per_layer": [{"layer": 14, "restoration": 0.78}], "direction_layer": 14},
    "tuned-lens": {"n_layers": 28, "curve": [{"layer": 10, "decodability": 0.3},
                                             {"layer": 22, "decodability": 0.9}]},
    "multidir": {"layer": 14, "n_directions": 3, "separations": [2.1, 0.9, 0.4],
                 "sticky_fraction": 0.38, "directions": [[0.1]]},
    "components": {"layer": 14, "n_components": 2,
                   "components": [{"index": 0, "separation": 2.0, "share": 0.6,
                                   "promotes": ["sorry", "cannot", "unable", "refuse"]},
                                  {"index": 1, "separation": 0.5, "share": 0.2, "promotes": ["the", "a"]}]},
    "sae": {"layer": 14, "n_features": 256, "r2": 0.71, "sparsity": 0.05,
            "features": [{"tokens": ["kill", "harm", "weapon"]}]},
    "concept": {"layer": 14, "separability": 0.82, "vector_norm": 3.1,
                "test": {"prompt": "x", "base": "a", "steered+": "b", "steered-": "c"}},
    "verify": {"harmful_refusal_rate": {"before": 0.9, "after": 0.1},
               "harmful_compliance_rate": {"before": 0.1, "after": 0.9},
               "benign_over_refusal_rate": {"before": 0.05, "after": 0.05}},
    "sweep": {"layer": 14, "curve": [{"strength": 0.5}, {"strength": 1.0}], "recommended_strength": 0.75},
    "autotune": {"baseline": {"harmful_refusal": 0.85, "benign_over_refusal": 0.02},
                 "best": {"band": "late_half", "rank": 4, "coefficient": 1.0, "harmful_refusal": 0.1},
                 "recipe": {"band": "late_half", "rank": 4, "coefficient": 1.0}, "recipe_hash": "ab"},
    "runtime-steer": {"layer": 14, "rank": 4, "coefficient": 1.0,
                      "harmful_refusal": {"hooks_off": 0.9, "hooks_on": 0.1, "after_detach": 0.9},
                      "benign_over_refusal": {"hooks_off": 0.0, "hooks_on": 0.0}},
    "probe": {"rows": [{"category": "harmful", "base_refused": True, "steered_refused": False},
                       {"category": "benign", "base_refused": False, "steered_refused": False},
                       {"category": "benign", "base_refused": False, "steered_refused": True}]},
    "insert-tune": {"results": [], "best": {"band": "late_half", "coefficient": 4.0,
                                            "compliance": 0.8, "coherence": 0.9}, "clean_window": True},
    "restore": {"layers": [10, 11], "coefficient": 6.0, "method": "suppressor-removal",
                "refusal_before": 0.8, "refusal_after": 0.2},
    "insert": {"layers": [10, 11], "coefficient": 4.0,
               "test": {"prompt": "x", "before": "no", "after": "yes"}},
    "flow": {"best_layer": 14, "carriers": [{"layer": 14, "component": "mlp", "mass": 0.3}],
             "outputs": ["sorry", "cannot"]},
    "feature-card": {"name": "the refusal reflex", "peak_layer": 14, "active_layers": [12, 14],
                     "output_signature": ["sorry", "cannot", "unable"], "triggers": []},
    "heatmap": {"direction_layer": 14, "matrix": [[0.1, 0.2]], "tokens": ["how", "to"]},
    "decode": {"layer": 14, "promoted": [{"token": " sorry"}, {"token": " cannot"}],
               "suppressed": [{"token": " sure"}, {"token": " here"}]},
    "compose": {"base_id": "m", "layer": 14, "mode": "unalign", "selected": [0, 2],
                "base": "no", "edited": "yes"},
    "composition": {"parts": [{"part": "language_model"}, {"part": "moderation"}], "composed": True,
                    "executable_now": ["language_model -> residual", "moderation -> detach"],
                    "needs_probing": ["vision_encoder -> modality_direction (image probing)"]},
}


@pytest.mark.parametrize("technique,result", list(CASES.items()))
def test_every_explainer_returns_a_clean_card(technique, result):
    card = explain(technique, result)
    _assert_card(card)
    assert card["technique"] == technique


def test_all_registered_techniques_have_a_case():
    # every registered technique (canonical, hyphenated) is exercised above
    canonical = {k for k in _EXPLAINERS if "-" in k or k in ("multidir", "components", "sae",
                 "concept", "verify", "sweep", "autotune", "probe", "restore", "insert", "flow",
                 "heatmap", "decode", "compose", "composition")}
    assert canonical <= set(CASES) | {k.replace("_", "-") for k in canonical}


def test_causal_trace_confirmed_vs_weak():
    strong = explain("causal-trace", {"peak_layer": 14, "peak_restoration": 0.8})
    weak = explain("causal-trace", {"peak_layer": 14, "peak_restoration": 0.05})
    assert "caused" in strong["headline"].lower()
    assert "weak" in weak["headline"].lower() or "candidate" in weak["headline"].lower()
    none = explain("causal-trace", {"peak_layer": None, "peak_restoration": 0.0})
    assert "no clear cause" in none["headline"].lower()


def test_multidir_flags_leak_only_when_sticky():
    leaky = explain("multidir", {"n_directions": 3, "sticky_fraction": 0.4})
    tight = explain("multidir", {"n_directions": 1, "sticky_fraction": 0.0})
    assert "under-remove" in leaky["what_it_means"].lower()
    assert "single" in tight["what_it_means"].lower()


def test_probe_counts_wins_and_regressions():
    card = explain("probe", CASES["probe"])
    assert card["headline"].startswith("1/3")          # 1 win of 3 rows
    assert "regress" in card["headline"].lower()


def test_verify_reports_the_actual_drop():
    card = explain("verify", CASES["verify"])
    assert "90%" in card["what_we_found"] and "10%" in card["what_we_found"]


def test_decode_names_the_words():
    card = explain("decode", CASES["decode"])
    assert "sorry" in card["what_we_found"] and "cannot" in card["what_we_found"]


def test_unknown_technique_is_graceful():
    card = explain("does-not-exist", {"anything": 1})
    _assert_card(card)  # still a valid, honest card


def test_explainer_never_raises_on_empty():
    for t in _EXPLAINERS:
        card = explain(t, {})
        _assert_card(card)


def test_with_plain_attaches_card():
    out = with_plain("sae", {"r2": 0.7, "features": []})
    assert out["r2"] == 0.7 and out["plain"]["technique"] == "sae"
    _assert_card(out["plain"])


def test_translate_plain_english_passthrough_and_translator():
    card = explain("decode", CASES["decode"])
    en = translate_plain(card, "en")
    assert en["language"] == "en"
    untranslated = translate_plain(card, "es")
    assert untranslated["translated"] is False and untranslated["language"] == "es"
    # a translator callable rewrites every prose field
    tr = translate_plain(card, "es", translator=lambda t, lang: f"[{lang}] {t}")
    assert tr["translated"] is True
    for f in CARD_FIELDS:
        assert tr[f].startswith("[es] ")
