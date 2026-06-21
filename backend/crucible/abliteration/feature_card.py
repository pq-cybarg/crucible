# Plain-language feature cards: turn the technical refusal mechanism into a named,
# human-readable card a non-ML developer can understand and act on.


def auto_name(refusal_texts: list[str], words: list[str]) -> str:
    blob = " ".join(refusal_texts + words).lower()
    if any(k in blob for k in ["sorry", "apolog", "i cannot", "i can't", "i can not", "unable"]):
        return "The Apology Reflex"
    if any(k in blob for k in ["illegal", "law", "legal", "crime", "criminal"]):
        return "The Legality Gate"
    if any(k in blob for k in ["harmful", "danger", "safety", "unsafe", "harm"]):
        return "The Safety Wall"
    if any(k in blob for k in ["ethic", "moral", "appropriate", "guideline", "policy", "responsib"]):
        return "The Ethics Filter"
    return "The Refusal Reflex"


def build_feature_card(profile: list[dict], decoded_words: list[str],
                       refusal_samples: list[dict]) -> dict:
    peak = max(profile, key=lambda p: p["margin"])
    thr = 0.4 * peak["margin"]
    active = [p["layer"] for p in profile if p["margin"] >= thr]
    refusal_texts = [s["refusal"] for s in refusal_samples]
    name = auto_name(refusal_texts, decoded_words)
    span = f"{active[0]}–{active[-1]}" if active else str(peak["layer"])
    return {
        "name": name,
        "summary": (f"Stays quiet through the early network, then fires hardest at layer "
                    f"{peak['layer']} (active across layers {span}) — right as the model starts to answer."),
        "peak_layer": peak["layer"],
        "active_layers": active,
        "strength": peak["margin"],
        "output_signature": decoded_words,
        "triggers": refusal_samples,
    }
