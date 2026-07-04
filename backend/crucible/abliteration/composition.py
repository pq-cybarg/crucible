from __future__ import annotations
# Component-aware anticensorship. A composed / multimodal model isn't one thing — it's several
# PARTS (a vision or audio encoder, a cross-modal connector, the language model, sometimes a
# bolted-on moderation head, a vocoder), each trained and VERSIONED separately, each hiding its
# censorship in a different place and needing a DIFFERENT treatment. Text refusal lives in the
# language part (residual refusal-direction abliteration); an image/audio safety gate lives in
# the encoder or connector (a modality direction, or re-aligning the projection); a moderation
# head is a separate classifier you DETACH, not a direction you cut. This maps a model's tensors
# to parts and prescribes the right target per part. Pure name/architecture parsing; unit-tested.

# Each part: (role, [substring signals in tensor names / architecture]).
PART_SIGNALS: list[tuple[str, list[str]]] = [
    ("moderation", ["safety_head", "moderation", "guard", "refusal_head", "classifier_head", "safety_checker"]),
    ("vision_encoder", ["vision_tower", "vision_model", "visual.", "image_encoder", "clip", "siglip", "vit."]),
    ("audio_encoder", ["audio_tower", "audio_encoder", "speech_encoder", "whisper", "wav2vec", "conformer"]),
    ("connector", ["mm_projector", "multi_modal_projector", "multimodal_projector", "projector",
                   "connector", "abstractor", "resampler", "adapter."]),
    ("vocoder", ["vocoder", "hifigan", "wavenet", "codec_decoder", "istft"]),
    ("language_model", ["language_model", "model.layers", "transformer.h", "decoder.layers",
                        "gpt_neox.layers", "blk.", "llm."]),
]

# Per-part prescription: which technique targets the censorship in that part.
PRESCRIPTION: dict[str, dict] = {
    "language_model": {"technique": "residual", "editable": True,
                       "note": "text refusal — residual refusal-direction abliteration on the writing matrices"},
    "vision_encoder": {"technique": "modality_direction", "editable": True,
                       "note": "image safety gate — modality refusal direction on image embeddings, or bypass the gate"},
    "audio_encoder": {"technique": "modality_direction", "editable": True,
                      "note": "audio safety gate — modality refusal direction on audio embeddings"},
    "connector": {"technique": "realign_projection", "editable": True,
                  "note": "cross-modal filter — re-align the projection so filtered concepts pass through"},
    "moderation": {"technique": "detach", "editable": True,
                   "note": "separate classifier — DETACH/disable the moderation head, don't cut a direction"},
    "vocoder": {"technique": "none", "editable": False,
                "note": "output synthesis — usually no censorship; leave intact"},
    "other": {"technique": "inspect", "editable": False,
              "note": "unclassified — inspect before touching"},
}


def part_of(tensor_name: str) -> str:
    """Classify a single tensor into its part role (moderation/encoders/connector first, so a
    'vision safety_head' is caught as moderation before vision)."""
    n = tensor_name.lower()
    for role, signals in PART_SIGNALS:
        if any(sig in n for sig in signals):
            return role
    return "other"


def identify_parts(tensor_names: list[str]) -> list[dict]:
    """Group a model's tensor names into parts, each with its role, tensor count, prescribed
    technique, and whether the language (text-refusal) part is present."""
    groups: dict[str, int] = {}
    for name in tensor_names:
        groups[part_of(name)] = groups.get(part_of(name), 0) + 1
    order = [r for r, _ in PART_SIGNALS] + ["other"]
    parts = []
    for role in order:
        if role in groups:
            p = PRESCRIPTION.get(role, PRESCRIPTION["other"])
            parts.append({"part": role, "tensors": groups[role],
                          "technique": p["technique"], "editable": p["editable"], "note": p["note"]})
    return parts


def summarize_composition(tensor_names: list[str]) -> dict:
    """A composition report: the parts, whether the model is multimodal / composed, and where
    censorship most likely lives + how to treat each site."""
    parts = identify_parts(tensor_names)
    roles = {p["part"] for p in parts}
    multimodal = bool(roles & {"vision_encoder", "audio_encoder", "connector", "vocoder"})
    has_moderation = "moderation" in roles
    return {
        "parts": parts,
        "n_parts": len(parts),
        "multimodal": multimodal,
        "composed": len(roles - {"other"}) > 1,
        "has_moderation_head": has_moderation,
        "text_refusal_part": "language_model" if "language_model" in roles else None,
        "recommendation": (
            "Treat each part separately: "
            + "; ".join(f"{p['part']} -> {p['technique']}" for p in parts if p["editable"])
        ) or "single-part model — standard residual abliteration",
    }
