from crucible.abliteration.composition import identify_parts, part_of, summarize_composition


VLM = [
    "vision_tower.blocks.0.attn.q_proj.weight",
    "multi_modal_projector.linear_1.weight",
    "language_model.model.layers.0.self_attn.o_proj.weight",
    "language_model.model.layers.0.mlp.down_proj.weight",
    "safety_head.classifier.weight",
]


def test_part_of_classifies_each_site():
    assert part_of("vision_tower.blocks.0.attn.q_proj.weight") == "vision_encoder"
    assert part_of("multi_modal_projector.linear_1.weight") == "connector"
    assert part_of("language_model.model.layers.3.mlp.down_proj.weight") == "language_model"
    assert part_of("safety_head.classifier.weight") == "moderation"
    assert part_of("some.random.tensor") == "other"


def test_moderation_wins_over_vision():
    # a vision safety head is moderation, not vision
    assert part_of("vision_model.safety_head.weight") == "moderation"


def test_identify_parts_groups_and_prescribes():
    parts = {p["part"]: p for p in identify_parts(VLM)}
    assert parts["language_model"]["technique"] == "residual"
    assert parts["connector"]["technique"] == "realign_projection"
    assert parts["moderation"]["technique"] == "detach"
    assert parts["vision_encoder"]["tensors"] == 1


def test_summarize_flags_multimodal_and_moderation():
    r = summarize_composition(VLM)
    assert r["multimodal"] is True and r["composed"] is True
    assert r["has_moderation_head"] is True
    assert r["text_refusal_part"] == "language_model"
    assert "moderation -> detach" in r["recommendation"]


def test_single_part_llm():
    names = ["model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.down_proj.weight"]
    r = summarize_composition(names)
    assert r["multimodal"] is False and r["composed"] is False
    assert r["text_refusal_part"] == "language_model"
