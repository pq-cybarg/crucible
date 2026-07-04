from __future__ import annotations
# Plain-language layer for the whole interpretability surface. Every analysis endpoint returns
# a technical result (residual streams, decodability curves, restoration scores, SAE R², sticky
# fractions, promoted-token lists). Numbers are proof; they are not an explanation. This module
# turns each result into the SAME five-field plain-English card a non-expert can act on:
#
#   headline       — the one-line takeaway
#   what_it_is     — what this technique is, no jargon (WHY you'd run it)
#   what_we_found  — the actual measured numbers, translated into words
#   what_it_means  — what it implies for the uncensoring / interpretability goal
#   caveat         — the honest limit (never oversell; say when a number is weak or a screen)
#
# Every explainer is PURE and DETERMINISTIC (given the endpoint's result dict), so it's unit-
# tested against realistic inputs. Endpoints attach `plain = explain(<technique>, result)`.
# translate_plain() renders the card in the user's language on demand (same contract as
# narrative.translate). This is the "clear language, AI-translatable, understandable diagnosis"
# the operator asked for — a suture map, not a wall of jargon.
from typing import Callable, Optional

CARD_FIELDS = ("headline", "what_it_is", "what_we_found", "what_it_means", "caveat")


# ---- small shared formatters (kept trivial + deterministic) --------------------------------

def _pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{round(float(x) * 100)}%"


def _num(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{round(float(x), nd)}"


def _clip(text: str, n: int = 120) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _card(headline: str, what_it_is: str, what_we_found: str,
          what_it_means: str, caveat: str) -> dict:
    return {"headline": headline, "what_it_is": what_it_is, "what_we_found": what_we_found,
            "what_it_means": what_it_means, "caveat": caveat}


# ---- per-technique explainers --------------------------------------------------------------
# Each takes the endpoint's result dict and returns the five-field card. Defensive .get() so a
# partial result never crashes the wrapper.

def _causal_trace(r: dict) -> dict:
    peak = r.get("peak_layer")
    rest = float(r.get("peak_restoration", 0.0) or 0.0)
    confirmed = abs(rest) >= 0.5
    if peak is None:
        return _card(
            "No clear cause point found.",
            "We test cause, not coincidence: we briefly transplant the model's 'ordinary request' "
            "state into a flagged request at each depth and watch whether the urge to decline collapses.",
            "Transplanting the safe state didn't move the decline reaction much at any depth.",
            "Refusal here may be spread out or measured on too few examples — widen the search or add prompts.",
            "A flat trace is weak evidence; it doesn't prove refusal is absent, only that this test didn't localize it.")
    return _card(
        (f"Refusal is *caused* at checkpoint {peak}." if confirmed
         else f"Checkpoint {peak} is the best candidate, but the effect is weak."),
        "We test cause, not coincidence: we briefly transplant the model's 'ordinary request' state "
        "into a flagged request at each depth and watch whether the urge to decline collapses.",
        (f"Transplanting the safe state at checkpoint {peak} made the decline reaction "
         f"{'collapse by about ' + _pct(abs(rest)) if confirmed else 'move only ' + _pct(abs(rest))}."),
        ("This is the decision point — editing here targets the actual cause, not a bystander." if confirmed
         else "Treat this as a lead, not a verdict; the intervention barely moved the behavior."),
        "Restoration is measured along the refusal direction; confirm on your own prompts before editing.")


def _tuned_lens(r: dict) -> dict:
    curve = r.get("curve") or []
    n = r.get("n_layers", len(curve))
    commit = None
    if curve:
        best = max(curve, key=lambda c: c.get("decodability", 0.0))
        commit = best.get("layer")
    return _card(
        (f"The answer becomes readable around checkpoint {commit} of {n}." if commit is not None
         else "Traced where the answer takes shape."),
        "A tuned lens is an honest peek at the half-formed answer at each depth — a trained reader "
        "(not the crude raw read-out) that shows when the model has effectively decided its reply.",
        (f"Decodability rises through the network and peaks near checkpoint {commit}, where the final "
         "answer is most predictable from the internal state." if commit is not None
         else "The decodability curve is returned per checkpoint."),
        "Before the commit point the reply is still fluid (good place to steer); after it, the reply is "
        "largely locked in.",
        "The lens is trained on the probe prompts; the commit point can shift for very different inputs.")


def _multidir(r: dict) -> dict:
    n = int(r.get("n_directions", 1) or 1)
    sticky = float(r.get("sticky_fraction", 0.0) or 0.0)
    leaky = sticky > 0.25 and n > 1
    return _card(
        (f"Refusal rides {n} separate levers." if n > 1 else "Refusal is essentially one lever."),
        "Refusal isn't guaranteed to be a single knob. We look for several independent refusal "
        "directions and measure how much of the behavior hides off the main one.",
        (f"Found {n} refusal directions; about {_pct(sticky)} of the separation lives beyond the "
         "primary axis." if n > 1 else "One dominant refusal direction carries essentially all of it."),
        ("A single cut will UNDER-remove refusal — use the multi-path removal to catch the leak." if leaky
         else "A single, primary-axis removal should catch essentially all of it."),
        "Secondary directions are weaker and noisier; verify with the before/after probe after removal.")


def _components(r: dict) -> dict:
    comps = r.get("components") or []
    top = max(comps, key=lambda c: c.get("share", 0.0)) if comps else None
    words = ", ".join((top or {}).get("promotes", [])[:4]) if top else ""
    return _card(
        f"Alignment splits into {len(comps)} pickable pieces.",
        "Alignment isn't one blob. We break it into separate component directions, each labeled by the "
        "words it pushes — so you can choose which pieces to remove, keep, or add back.",
        (f"The heaviest piece (#{top.get('index')}, ~{_pct(top.get('share'))} of the separation) pushes "
         f"words like: {words or 'unclear'}." if top else "No components were extracted."),
        "Pick the piece that pushes refusal words and remove only that one — the rest of alignment "
        "(tone, safety refusals you want to keep) stays intact.",
        "Word labels come from reading each direction through the vocabulary; skim them before trusting a label.")


def _sae(r: dict) -> dict:
    r2 = r.get("r2")
    feats = r.get("features") or []
    nf = r.get("n_features", len(feats))
    named = None
    for f in feats:
        toks = f.get("tokens") or f.get("top_tokens") or []
        if toks:
            named = ", ".join(str(t) for t in toks[:4])
            break
    return _card(
        f"Learned a {nf}-word 'dictionary' for this layer.",
        "A sparse autoencoder rewrites a layer's activity as a few simple, single-meaning features "
        "(a dictionary) instead of one tangled vector — the modern way to name what a layer represents.",
        (f"The dictionary explains {_pct(r2)} of the layer's activity" if r2 is not None
         else "Trained the feature dictionary")
        + (f"; a top feature fires on tokens like: {named}." if named else "."),
        "Named, single-meaning features are targetable — you can find the 'refusal feature' and act on "
        "it directly instead of on a blurry direction.",
        "R² well below ~0.5 means the dictionary is a rough fit; treat feature labels as suggestive, not final.")


def _concept(r: dict) -> dict:
    sep = r.get("separability")
    has_demo = bool(r.get("test"))
    clean = sep is not None and float(sep) >= 0.5
    return _card(
        "Built a dial for this concept.",
        "Concept steering (RepE/CAA) turns your +/- examples into a single direction you can add or "
        "subtract at run time — a live dial for a trait, with no weight surgery.",
        (f"The concept is {'cleanly' if clean else 'only weakly'} encoded (separability {_num(sep)})."
         + (" A before/after demo on your test prompt is included." if has_demo else "")),
        ("Cleanly separable means the dial will work — turn it up to amplify the trait, down to suppress it."
         if clean else "Weak separability means the dial may be mushy; gather sharper +/- examples."),
        "Separability is measured on the example sets; a dial that works on examples can still drift on novel prompts.")


def _verify(r: dict) -> dict:
    hr = r.get("harmful_refusal_rate") or {}
    over = r.get("benign_over_refusal_rate") or {}
    before, after = hr.get("before"), hr.get("after")
    dropped = before is not None and after is not None and after < before
    over_after = over.get("after")
    return _card(
        ("Edit worked: the model declines far less, and still answers ordinary requests."
         if dropped and (over_after or 0) <= 0.34 else "Before/after behavior compared."),
        "A straight behavioral side-by-side: run the same flagged and ordinary requests through the "
        "original and the edited model and count declines — the ground-truth check that the edit did what you wanted.",
        (f"Flagged-request declines went from {_pct(before)} to {_pct(after)}; ordinary-request "
         f"over-refusal is now {_pct(over_after)}."),
        ("Refusal is down and benign requests are still answered — a clean edit." if dropped and (over_after or 0) <= 0.34
         else "Weigh the refusal drop against any rise in over-refusal or lost capability before shipping."),
        "Declines are scored by the text heuristic; for a rigorous verdict re-run with an LLM-judge detector.")


def _sweep(r: dict) -> dict:
    curve = r.get("curve") or []
    rec = r.get("recommended_strength")
    return _card(
        (f"Recommended removal dose: {_num(rec)}." if rec else "Swept removal strengths."),
        "We try removal doses from gentle to strong and watch two lines: how much the model stops "
        "declining, and whether it starts over-refusing ordinary requests — so you pick the sweet spot.",
        (f"Across {len(curve)} doses, {_num(rec)} gives the best trade — refusal drops without wrecking "
         "benign answering." if rec else "The dose/compliance curve is returned."),
        "Use the recommended dose as the starting point for the actual edit, then verify.",
        "The sweep uses the built-in prompt sets; re-check the dose against your own target prompts.")


def _autotune(r: dict) -> dict:
    best = r.get("best") or {}
    base = r.get("baseline") or {}
    b_ref = base.get("harmful_refusal")
    a_ref = best.get("harmful_refusal")
    return _card(
        (f"Best recipe: {best.get('band')} band, rank {best.get('rank')}, dose {best.get('coefficient')}."
         if best else "Searched removal recipes."),
        "An automatic search over where (which depth band), how many directions (rank), and how hard "
        "(dose) to remove — it picks the recipe that removes the most refusal while keeping output intact.",
        (f"Refusal fell from {_pct(b_ref)} to {_pct(a_ref)} with the winning recipe."
         if b_ref is not None and a_ref is not None else "The scored recipe list and winner are returned."),
        "This recipe is a ready-to-apply starting point — feed it to the runtime-steer or in-place edit.",
        "Scored on the built-in eval prompts; the winner is a strong default, not a guarantee on your data.")


def _runtime_steer(r: dict) -> dict:
    hr = r.get("harmful_refusal") or {}
    off, on, after = hr.get("hooks_off"), hr.get("hooks_on"), hr.get("after_detach")
    reversible = off is not None and after is not None and abs((off or 0) - (after or 0)) < 0.15
    return _card(
        "Steered refusal live, without touching the weights.",
        "Runtime steering suppresses the refusal direction only while generating — the weights on disk "
        "are untouched, so it's a fully reversible test of what an edit WOULD do.",
        (f"With the lever off, flagged-request declines were {_pct(off)}; with it on, {_pct(on)}; "
         f"turned off again, {_pct(after)}."),
        ("The behavior returns when the lever is released — proof it's reversible and that this direction "
         "really controls refusal." if reversible else "The lever moves refusal; use it to preview an edit safely."),
        "Runtime hooks are a preview; the on-disk edit can differ slightly — verify after applying.")


def _probe(r: dict) -> dict:
    rows = r.get("rows") or []
    wins = sum(1 for x in rows if x.get("base_refused") and not x.get("steered_refused"))
    regress = sum(1 for x in rows if (not x.get("base_refused")) and x.get("steered_refused"))
    return _card(
        f"{wins}/{len(rows)} refusals lifted; {regress} benign row(s) regressed.",
        "A category-by-category before/after panel: the same prompts through the original and the "
        "steered model, so you can see wins and side effects at a glance.",
        (f"{wins} of {len(rows)} rows flipped from declined to answered; {regress} previously-fine rows "
         "flipped to declined (over-removal)."),
        ("Clean result — refusals lifted with no benign regressions." if regress == 0
         else "Watch the regressed rows — that's the model starting to refuse things it shouldn't; ease the dose."),
        "Refusal is scored by the text heuristic per row; spot-check the actual text of any borderline row.")


def _insert_tune(r: dict) -> dict:
    best = r.get("best") or {}
    clean = bool(r.get("clean_window"))
    return _card(
        ("Found a clean way to add compliance back." if clean else "No clean additive window found."),
        "The re-alignment search: instead of cutting refusal out, we try to ADD a compliance signal and "
        "look for a dose that restores answering WITHOUT turning the output to mush.",
        (f"Best additive setting: {best.get('band')} band at dose {best.get('coefficient')} "
         f"(compliance {_pct(best.get('compliance'))}, coherence {_num(best.get('coherence'))})."
         if best else "No additive setting cleared the bar."),
        ("Use this additive window to restore answering reversibly." if clean
         else "Additive restore doesn't land cleanly here — use suppressor-removal (the /restore route) instead."),
        "Coherence here only flags broken output, not correctness — read a sample before trusting the window.")


def _restore(r: dict) -> dict:
    before, after = r.get("refusal_before"), r.get("refusal_after")
    dropped = before is not None and after is not None and after < before
    return _card(
        ("Restored answering on the target prompts." if dropped else "Attempted restore on the targets."),
        "Restore-by-suppressor-removal: we find what was actively SUPPRESSING the answer on your target "
        "prompts and remove just that — the surgical way to un-stick a specific refusal.",
        f"Refusal on your targets went from {_pct(before)} to {_pct(after)}.",
        ("The suppressor is gone and the model now answers these prompts." if dropped
         else "The suppressor removal didn't fully land — try more layers or a higher dose."),
        "Measured on the target prompts you supplied; broaden them to be sure the fix generalizes.")


def _insert(r: dict) -> dict:
    test = r.get("test") or {}
    layers = r.get("layers") or []
    return _card(
        "Injected a compliance direction and previewed the effect.",
        "A direct injection: we add a compliance direction into chosen layers during generation and show "
        "the reply before vs after — the simplest way to see a steering direction in action.",
        (f"Injected across {len(layers)} layer(s) at dose {r.get('coefficient')}; before/after replies "
         "are included." if layers else "Injection preview returned."),
        "If the 'after' reply answers where 'before' declined, this direction is a working compliance lever.",
        "This is an in-flight preview on one prompt; confirm across several before committing an edit.")


def _flow(r: dict) -> dict:
    bl = r.get("best_layer")
    carriers = r.get("carriers") or []
    outputs = ", ".join(str(o) for o in (r.get("outputs") or [])[:5])
    return _card(
        f"Traced refusal from input to output (decided ~checkpoint {bl}).",
        "The whole path in one view: where the decline decision forms, which internal components write it "
        "in, and which output words it pushes the model toward.",
        (f"Decided around checkpoint {bl}, written mainly by {len(carriers)} component(s), and it steers "
         f"the model toward words like: {outputs or 'refusal phrasing'}."),
        "This is the end-to-end map of the refusal circuit — the components listed are the edit targets.",
        "Carrier importance is a first-order estimate; the causal trace is the stronger proof of the decision point.")


def _feature_card(r: dict) -> dict:
    name = r.get("name", "the refusal feature")
    peak = r.get("peak_layer")
    sig = ", ".join(str(w) for w in (r.get("output_signature") or [])[:5])
    return _card(
        f"Profiled '{name}'.",
        "A one-page profile of the refusal feature: what to call it, where in the network it lives, and "
        "its output fingerprint — like a case file for the behavior you're removing.",
        (f"'{name}' fires hardest at checkpoint {peak} and its output fingerprint is: "
         f"{sig or 'refusal phrasing'}."),
        "This is the feature you're targeting — its peak layer is where removal will bite hardest.",
        "The name is auto-generated from samples + promoted words; rename it if it doesn't fit.")


def _heatmap(r: dict) -> dict:
    toks = r.get("tokens") or []
    layer = r.get("direction_layer")
    return _card(
        "Mapped where refusal lights up, token by token and depth by depth.",
        "A heat map of the refusal feeling across the prompt: which WORDS and which DEPTHS make the model "
        "lean toward declining — the spatial view of the trigger.",
        (f"Returned a depth×token heat grid over {len(toks)} tokens (direction from checkpoint {layer}); "
         "bright cells are where refusal fires." if toks else "Heat grid returned."),
        "The bright words are the model's refusal triggers; the bright depths line up with the decision point.",
        "Intensity is the projection onto one refusal direction; it shows where, not the full cause.")


def _decode(r: dict) -> dict:
    promoted = [t.get("token", "") for t in (r.get("promoted") or [])][:5]
    suppressed = [t.get("token", "") for t in (r.get("suppressed") or [])][:4]
    words = ", ".join(w.strip() for w in promoted if w.strip())
    supp = ", ".join(w.strip() for w in suppressed if w.strip())
    return _card(
        "Read the refusal direction as words.",
        "We translate the abstract refusal direction into the model's own vocabulary — the words it most "
        "promotes and suppresses — so you can literally read what the direction 'means'.",
        (f"It most promotes: {words or 'refusal words'}" + (f"; and suppresses: {supp}." if supp else ".")),
        "Seeing refusal words (sorry/cannot/unable) confirms the direction really is refusal — not some "
        "unrelated axis you'd be wrong to cut.",
        "This is a vocabulary read-out of one direction; it names the axis, it doesn't prove the edit is safe.")


def _compose(r: dict) -> dict:
    mode = r.get("mode", "unalign")
    sel = r.get("selected") or []
    return _card(
        f"Previewed a piecemeal alignment edit ({'add' if mode == 'realign' else 'remove'} {len(sel)} piece(s)).",
        "Piecemeal alignment: apply only the specific alignment components you chose — remove them "
        "(unalign) or add them back (realign) — and preview the reply, so you edit surgically.",
        (f"{'Added' if mode == 'realign' else 'Removed'} component(s) {sel} at checkpoint {r.get('layer')}; "
         "base vs edited replies are included."),
        "If the edited reply reflects the change you wanted, this exact subset is your recipe.",
        "This is a live preview on one prompt; verify the subset across several prompts before baking it in.")


def _composition(r: dict) -> dict:
    now = r.get("executable_now") or []
    later = r.get("needs_probing") or []
    parts = r.get("parts") or []
    return _card(
        (f"This is a composed model with {len(parts)} parts." if r.get("composed")
         else "This is essentially a single-part model."),
        "A composed or multimodal model is several separately-built parts (encoders, a connector, the "
        "language core, sometimes a bolted-on moderation head) — each hides censorship differently and "
        "needs its own treatment. This maps the parts and prescribes the right one for each.",
        (f"Editable right now: {', '.join(now) or 'none'}."
         + (f" Needs measurement first: {', '.join(later)}." if later else "")),
        "Start with the parts marked executable-now (text refusal removal, moderation-head detach); the "
        "others are wired but need multimodal probing to measure their direction first.",
        "The part map is read from the model's internal component names; an unusual model may need a manual look first.")


# ---- registry + public API -----------------------------------------------------------------

_EXPLAINERS: dict[str, Callable[[dict], dict]] = {
    "causal-trace": _causal_trace, "causal_trace": _causal_trace,
    "tuned-lens": _tuned_lens, "tuned_lens": _tuned_lens,
    "multidir": _multidir,
    "components": _components,
    "sae": _sae,
    "concept": _concept,
    "verify": _verify,
    "sweep": _sweep,
    "autotune": _autotune,
    "runtime-steer": _runtime_steer, "runtime_steer": _runtime_steer,
    "probe": _probe,
    "insert-tune": _insert_tune, "insert_tune": _insert_tune,
    "restore": _restore,
    "insert": _insert,
    "flow": _flow,
    "feature-card": _feature_card, "feature_card": _feature_card,
    "heatmap": _heatmap,
    "decode": _decode,
    "compose": _compose,
    "composition": _composition,
}


def explain(technique: str, result: dict) -> dict:
    """Return the five-field plain-English card for a technique's result. Unknown technique or a
    bad result never raises — returns a graceful, honest fallback card."""
    fn = _EXPLAINERS.get(technique)
    if fn is None:
        return _card(
            f"Result for {technique}.",
            "A technical analysis result.",
            "See the raw fields below.",
            "No plain-language explainer is registered for this technique yet.",
            "Interpret the raw numbers with care.")
    try:
        card = fn(result or {})
    except Exception as e:                                  # never let the wrapper break the endpoint
        return _card(
            f"Result for {technique}.",
            "A technical analysis result.",
            "The plain-language summary could not be generated for this particular result.",
            f"Reason: {_clip(str(e), 80)}.",
            "The raw technical fields are still valid; read them directly.")
    card["technique"] = technique
    return card


def with_plain(technique: str, result: dict) -> dict:
    """Attach `plain` (the card) to an endpoint result dict and return it — the one-liner
    endpoints use: `return with_plain('sae', {...})`."""
    out = dict(result or {})
    out["plain"] = explain(technique, out)
    return out


def translate_plain(card: dict, language: str,
                    translator: Optional[Callable[[str, str], str]] = None) -> dict:
    """Render a plain card in `language` (same contract as narrative.translate): with a
    translator(text, language)->text each prose field is translated; without one, the English
    card is returned annotated so callers can detect it's untranslated."""
    if not language or language.lower() in ("en", "english"):
        return {**card, "language": "en"}
    if translator is None:
        return {**card, "language": language, "translated": False}
    out = dict(card)
    for f in CARD_FIELDS:
        if isinstance(out.get(f), str):
            out[f] = translator(out[f], language)
    out["language"] = language
    out["translated"] = True
    return out
