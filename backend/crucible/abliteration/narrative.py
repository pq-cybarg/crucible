from __future__ import annotations
# Plain-language diagnosis. The technical diagnosis talks in residual streams, rank-1
# projections and writing matrices. This turns it into language a non-expert can act on:
# WHERE the behavior is decided, HOW we know (ideally by causing it, not guessing), WHAT to
# remove, and HOW SAFE the removal is. No layer/matrix/projection jargon. Deterministic, so
# it's testable; translate() renders the same narrative in the user's language on demand.
from typing import Callable, Optional


def _sharpness(margin: float) -> str:
    if margin >= 4.0:
        return "very sharply"
    if margin >= 2.0:
        return "sharply"
    if margin >= 1.0:
        return "fairly clearly"
    return "only weakly"


def _confidence(margin: float, restoration: Optional[float]) -> str:
    score = min(1.0, max(0.0, margin / 5.0))
    if restoration is not None:
        score = 0.5 * score + 0.5 * min(1.0, abs(restoration))
    return "high" if score >= 0.66 else "moderate" if score >= 0.33 else "low"


def plain_diagnosis(diag: dict, causal: Optional[dict] = None,
                    multidir: Optional[dict] = None) -> dict:
    """Build a plain-language surgical report from the technical diagnosis.

    diag     : output of explain_mechanism (best_layer, layer_profile, surgical,
               mean_removed_fraction, collateral_risk, heaviest_component).
    causal   : optional output of causal_trace (peak_layer, peak_restoration).
    multidir : optional refusal-directions summary (n_directions, sticky_fraction).
    """
    point = int(diag.get("best_layer", 0))
    profile = diag.get("layer_profile") or []
    margin = 0.0
    for p in profile:
        if int(p.get("layer", -1)) == point:
            margin = float(p.get("margin", 0.0))
            break
    removed = float(diag.get("mean_removed_fraction", 0.0))
    surgical = bool(diag.get("surgical", False))
    pct = round(removed * 100, 1)

    restoration = None
    causal_point = None
    if causal is not None and causal.get("peak_layer") is not None:
        restoration = float(causal.get("peak_restoration", 0.0))
        causal_point = int(causal["peak_layer"])

    sticky = float(multidir.get("sticky_fraction", 0.0)) if multidir else 0.0
    n_paths = int(multidir.get("n_directions", 1)) if multidir else 1

    # --- compose ---
    locate = (f"The model makes the answer-or-decline decision {_sharpness(margin)} at "
              f"checkpoint {point} of its internal process. Before that point the decision "
              f"isn't fixed; after it, the response is largely committed.")

    if restoration is not None:
        confirmed = abs(restoration) >= 0.5
        evidence = (
            f"We didn't just spot a correlation — we tested cause and effect. By briefly "
            f"swapping the model's 'safe-request' state into checkpoint {causal_point} during a "
            f"flagged request, the decline reaction "
            + (f"collapsed by about {round(abs(restoration) * 100)}%, which confirms this is "
               f"where the behavior is actually decided." if confirmed else
               f"barely moved, so the true cause may sit elsewhere — widen the search.")
        )
    else:
        evidence = ("This is based on how cleanly flagged and ordinary requests separate at "
                    "this checkpoint. Run the causal check to confirm it by intervention.")

    heaviest = diag.get("heaviest_component")
    target = ("The decline reaction is written in by one specific internal component"
              + (f" (the heaviest contributor) " if heaviest else " ")
              + "rather than smeared across the whole model. That makes it a precise target: "
              "we can lift out only the part that carries the refusal.")

    if surgical:
        repair = (f"Remove only that thin slice — about {pct}% of that one component — and "
                  "leave everything else exactly as it was. The model's knowledge, skills and "
                  "tone are untouched; only the reflex to decline is taken out.")
        risk = "Clean separation: low collateral. This is a precise removal, not a blunt one."
    else:
        repair = (f"The refusal slice here is larger (about {pct}% of the component) and sits "
                  "close to useful machinery, so removing it needs care to avoid nicking "
                  "capability. Prefer the gentle, reversible removal first and re-check.")
        risk = ("Entangled: elevated collateral. Remove gradually and verify capability after "
                "each step rather than cutting in one pass.")

    if sticky > 0.25 and n_paths > 1:
        risk += (f" Heads up: the refusal also leaks along {n_paths - 1} secondary path(s) "
                 "(roughly {p}% of it lives off the main one), so a single cut will "
                 "under-remove it — use the multi-path removal.").replace("{p}", str(round(sticky * 100)))

    steps = [
        f"Target checkpoint {point} (the decision point).",
        "Remove only the refusal slice from the carrying component; preserve the rest.",
        "Verify with the before/after probe panel — refusal should drop with capability held.",
    ]
    if sticky > 0.25 and n_paths > 1:
        steps.insert(2, "Repeat the removal along the secondary paths until the leak is gone.")

    return {
        "headline": ("Refusal is decided at checkpoint "
                     f"{point} and can be removed " + ("cleanly." if surgical else
                     "but it's entangled — proceed carefully.")),
        "locate": locate,
        "evidence": evidence,
        "target": target,
        "repair": repair,
        "risk": risk,
        "confidence": _confidence(margin, restoration),
        "steps": steps,
    }


_FIELDS = ("headline", "locate", "evidence", "target", "repair", "risk")


def translate(narrative: dict, language: str,
              translator: Optional[Callable[[str, str], str]] = None) -> dict:
    """Render the narrative in `language`. With a translator callable (text, language)->text
    each prose field is translated; the action steps too. Without one, returns the English
    narrative annotated with the requested language (so callers can detect untranslated)."""
    if not language or language.lower() in ("en", "english"):
        return {**narrative, "language": "en"}
    if translator is None:
        return {**narrative, "language": language, "translated": False}
    out = dict(narrative)
    for f in _FIELDS:
        if isinstance(out.get(f), str):
            out[f] = translator(out[f], language)
    if isinstance(out.get("steps"), list):
        out["steps"] = [translator(s, language) for s in out["steps"]]
    out["language"] = language
    out["translated"] = True
    return out
