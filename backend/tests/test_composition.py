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


def test_part_writing_matrices_scopes_to_part():
    from crucible.abliteration.composition import part_writing_matrices
    names = [
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "vision_tower.blocks.0.attn.o_proj.weight",     # vision o_proj -> excluded
        "multi_modal_projector.linear_1.weight",         # connector, not a writing matrix
    ]
    got = part_writing_matrices(names, "language_model")
    assert len(got) == 2 and all("language_model" in n for n in got)
    assert part_writing_matrices(names, "vision_encoder") == ["vision_tower.blocks.0.attn.o_proj.weight"]


def _min_gguf(path, name, arr):
    import struct
    out, inn = arr.shape
    def gstr(x):
        b = x.encode(); return struct.pack("<Q", len(b)) + b
    buf = bytearray()
    buf += struct.pack("<I", 0x46554747) + struct.pack("<I", 3)
    buf += struct.pack("<Q", 1) + struct.pack("<Q", 1)
    buf += gstr("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", 32)
    buf += gstr(name) + struct.pack("<I", 2) + struct.pack("<Q", inn) + struct.pack("<Q", out)
    buf += struct.pack("<I", 0) + struct.pack("<Q", 0)
    buf += b"\x00" * ((-len(buf)) % 32)
    buf += arr.astype("<f4").tobytes()
    path.write_bytes(buf)


def test_part_scoped_gguf_abliteration(tmp_path):
    import numpy as np
    from crucible.weights.gguf_edit import abliterate_gguf
    _write_min_gguf = _min_gguf
    W = np.random.default_rng(0).standard_normal((16, 6)).astype(np.float32)
    r = np.random.default_rng(1).standard_normal(16); r /= np.linalg.norm(r)
    p = tmp_path / "m.gguf"
    _write_min_gguf(p, "blk.0.attn_output.o_proj.weight", W)   # 'blk.' -> language_model
    # scoping to language_model edits it; scoping to vision skips it (untouched)
    assert abliterate_gguf(str(p), r, name_filter=("o_proj",), part="language_model")["n_edited"] == 1
    _write_min_gguf(p, "blk.0.attn_output.o_proj.weight", W)   # rewrite fresh
    res = abliterate_gguf(str(p), r, name_filter=("o_proj",), part="vision_encoder")
    assert res["n_edited"] == 0 and res["n_skipped"] == 1
