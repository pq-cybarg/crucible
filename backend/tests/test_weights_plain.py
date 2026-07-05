"""Plain-language weights explainer: maps ML structure (layers, tensors, shapes) to human concepts
so a non-expert understands what the model is and how to change it. Pure mapping — unit-tested."""
from crucible.weights.plain import (component_role, explain_weights, humanize_tensor, layer_band,
                                    layer_index, shape_plain)


def test_layer_index():
    assert layer_index("blk.12.attn_q.weight") == 12
    assert layer_index("model.layers.3.mlp.down_proj.weight") == 3
    assert layer_index("token_embd.weight") == -1


def test_component_role_maps_parts_to_human_language():
    assert component_role("blk.5.attn_q.weight")["role"] == "attention"
    assert component_role("blk.5.ffn_down.weight")["role"] == "recall-out"
    assert component_role("token_embd.weight")["role"] == "vocabulary"
    assert component_role("output.weight")["role"] == "output"
    assert "supporting" in component_role("something.odd.weight")["does"]


def test_layer_band_by_depth():
    assert layer_band(0, 30)["band"] == "early"
    assert layer_band(15, 30)["band"] == "middle"
    assert layer_band(29, 30)["band"] == "late"
    assert "refuse" in layer_band(29, 30)["role"]        # late layers decide compliance
    assert layer_band(-1, 30)["band"] == "shared"


def test_shape_plain_is_readable():
    assert shape_plain([4096, 11008]) == "connects 11,008 inputs to 4,096 internal features"
    assert shape_plain([32000]) == "32,000 values"


def test_humanize_tensor():
    h = humanize_tensor({"name": "blk.29.ffn_down.weight", "shape": [4096, 11008], "n_params": 45e6}, 30)
    assert h["layer"] == 29 and h["component"] == "recall-out" and h["band"] == "late"
    assert "connects" in h["shape_plain"]


def test_explain_weights_gives_a_human_guide():
    summary = {"n_layers": 32, "total_params": 1_500_000_000, "architecture": "qwen2",
               "dtypes": {"Q4_K": 200, "F32": 20}}
    tensors = [{"name": f"blk.{i}.attn_q.weight", "shape": [2048, 2048], "n_params": 4_000_000}
               for i in range(32)]
    tensors += [{"name": f"blk.{i}.ffn_down.weight", "shape": [2048, 5504], "n_params": 11_000_000}
                for i in range(32)]
    out = explain_weights(summary, tensors)
    card = out["model"]
    assert "1.5 billion" in card["headline"] and "32 layers" in card["headline"]
    assert "COMPRESSED" in card["size_meaning"]           # Q4_K -> quantized note
    assert "refuse" in card["how_it_works"] or "refuse" in card["how_to_change"]
    # per-layer journey roll-up: early/middle/late bands present
    bands = {layer["band"] for layer in out["layers"]}
    assert {"early", "middle", "late"} <= bands
    assert len(out["layers"]) == 32
    assert out["legend"]["late"]


def test_explain_full_precision_note():
    out = explain_weights({"n_layers": 4, "total_params": 200_000_000, "dtypes": {"F16": 10}}, [])
    assert "FULL precision" in out["model"]["size_meaning"]
    assert "small" in out["model"]["headline"]
