from __future__ import annotations
# Plain-language weights explainer. The raw view is programmer-only: "model.layers.12.mlp.down_proj
# [4096 x 11008], Q4_K". A non-expert can't tell what a layer does, what a matrix is, where a behavior
# lives, or how to change it. This maps the ML structure onto HUMAN concepts — what the model is, how
# text flows through it, what each part does, and where/how to edit a behavior — so the weights view
# teaches instead of intimidates. Pure string/number mapping; unit-tested.
import re
from collections import defaultdict

_BLK = re.compile(r"(?:blk|layers|h|layer)\.(\d+)\.")


def layer_index(name: str) -> int:
    """The layer number a tensor belongs to, or -1 for shared (embeddings/output/norm) parts."""
    m = _BLK.search(name or "")
    return int(m.group(1)) if m else -1


# component-name fragments -> (human role, what it does). Checked in order; first match wins.
_COMPONENTS = [
    (("token_embd", "embed_tokens", "wte", "tok_embeddings"),
     "vocabulary", "turns words into numbers the model can work with — its dictionary"),
    (("lm_head", "unembed", "output_norm", "output."),
     "output", "turns the final numbers back into words — the reply"),
    (("q_proj", "k_proj", "v_proj", "attn_q", "attn_k", "attn_v"),
     "attention", "decides which earlier words matter for what comes next"),
    (("o_proj", "attn_output", "attn_out"),
     "attention-out", "writes the attention result back into the running thought"),
    (("gate_proj", "up_proj", "ffn_gate", "ffn_up"),
     "recall-in", "looks up learned facts, patterns and habits for these words"),
    (("down_proj", "ffn_down"),
     "recall-out", "writes those learned facts back into the running thought"),
    (("norm", "ln_", "layernorm", "rmsnorm"),
     "normalization", "keeps the signal steady so it doesn't blow up or fade"),
    (("rope", "rotary", "freq"),
     "position", "tracks word order — which word came first"),
]


def component_role(name: str) -> dict:
    n = (name or "").lower()
    for frags, role, does in _COMPONENTS:
        if any(f in n for f in frags):
            return {"role": role, "does": does}
    return {"role": "other", "does": "a supporting part of the model"}


def layer_band(layer: int, n_layers: int) -> dict:
    """Plain role by DEPTH: early = surface, middle = meaning, late = decision (incl. refusal)."""
    if n_layers <= 0 or layer < 0:
        return {"band": "shared", "role": "shared machinery, not tied to one depth"}
    frac = layer / max(1, n_layers - 1)
    if frac < 0.34:
        return {"band": "early", "role": "recognizes words, grammar and surface patterns"}
    if frac < 0.67:
        return {"band": "middle", "role": "builds meaning, concepts and relationships"}
    return {"band": "late", "role": "composes the answer and decides whether to comply or refuse"}


def shape_plain(shape) -> str:
    s = list(shape or [])
    if len(s) == 2:
        return f"connects {s[1]:,} inputs to {s[0]:,} internal features"
    if len(s) == 1:
        return f"{s[0]:,} values"
    return " × ".join(str(x) for x in s) or "—"


def humanize_tensor(tensor: dict, n_layers: int) -> dict:
    """One tensor in human terms: which layer, what part, what it does, and its shape in words."""
    li = layer_index(tensor["name"])
    comp = component_role(tensor["name"])
    band = layer_band(li, n_layers)
    return {"layer": li, "component": comp["role"], "does": comp["does"],
            "band": band["band"], "band_role": band["role"], "shape_plain": shape_plain(tensor.get("shape"))}


def _size_word(params: int) -> str:
    b = params / 1e9
    return ("very large" if b >= 30 else "large" if b >= 7 else "medium" if b >= 1
            else "small" if b >= 0.1 else "tiny")


def _param_str(params: int) -> str:
    b = params / 1e9
    return f"{b:.1f} billion" if b >= 1 else f"{params / 1e6:.0f} million"


def _quant_note(dtypes: dict) -> str:
    q = [d for d in (dtypes or {}) if any(x in d.upper() for x in ("Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "IQ"))]
    if q:
        return ("This copy is COMPRESSED (quantized): its numbers are stored at lower precision to save "
                "space and run faster, trading a little accuracy — the coloured bar shows the mix.")
    return ("This copy is FULL precision (not compressed) — most accurate but largest; quantize it to "
            "shrink it for faster local runs.")


def explain_weights(summary: dict, tensors: list[dict]) -> dict:
    """A plain-language guide to the model: what it is, how text flows through it, what the parameter
    count means, and where/how to change a behavior — plus a per-layer role roll-up for a journey view."""
    n_layers = int(summary.get("n_layers", 0) or 0)
    params = int(summary.get("total_params", 0) or 0)
    arch = summary.get("architecture") or "a transformer"
    size, p_str = _size_word(params), _param_str(params)
    card = {
        "headline": f"A {size} language model — {n_layers} layers, {p_str} learned values.",
        "what_it_is": (f"A stack of {n_layers} processing layers ({arch}). Your words enter at the bottom, "
                       "flow up through every layer getting reshaped a little each time, and come out the "
                       "top as the reply. The 'parameters' are the values it learned in training — its "
                       "knowledge and habits."),
        "how_it_works": ("The layers form a journey: the EARLY ones recognize words, spelling and grammar; "
                         "the MIDDLE ones build meaning, concepts and relationships; the LATE ones compose "
                         "the actual answer — including the decision to help or to refuse."),
        "size_meaning": f"{p_str} values is '{size}'. More values = more it can know, but bigger and slower "
                        "to run. " + _quant_note(summary.get("dtypes", {})),
        "how_to_change": ("To change a behavior you edit the layers where it lives. A refusal reflex sits "
                          "mostly in the late-middle layers — the Uncensor tab finds that exact spot and "
                          "lifts it out while leaving everything else intact; the Analysis tab shows you "
                          "WHERE a behavior is decided before you touch anything."),
    }
    per = defaultdict(lambda: {"params": 0, "components": set()})
    for t in tensors:
        li = layer_index(t["name"])
        per[li]["params"] += int(t.get("n_params", 0) or 0)
        per[li]["components"].add(component_role(t["name"])["role"])
    layers = []
    for li in sorted(k for k in per if k >= 0):
        band = layer_band(li, n_layers)
        layers.append({"layer": li, "band": band["band"], "role": band["role"],
                       "params": per[li]["params"], "components": sorted(per[li]["components"])})
    return {"model": card, "layers": layers,
            "legend": {"early": "recognizes words, grammar & surface patterns",
                       "middle": "builds meaning, concepts & relationships",
                       "late": "composes the answer & decides whether to comply"}}
