from crucible.weights.explorer import group_by_layer, layer_of, summarize


def test_layer_of():
    assert layer_of("blk.5.attn_q.weight") == 5
    assert layer_of("output.weight") == -1


def test_group_by_layer():
    g = group_by_layer([{"name": "blk.0.a"}, {"name": "blk.0.b"}, {"name": "output"}])
    assert len(g[0]) == 2 and len(g[-1]) == 1


def test_summarize():
    parsed = {"metadata": {"general.architecture": "glm"}, "tensors": [
        {"name": "blk.0.x", "shape": [2, 2], "dtype": "F16", "n_params": 4, "offset": 0},
        {"name": "blk.1.x", "shape": [2, 2], "dtype": "Q4_K", "n_params": 4, "offset": 0},
        {"name": "output.weight", "shape": [2], "dtype": "F32", "n_params": 2, "offset": 0}]}
    s = summarize(parsed)
    assert s["n_tensors"] == 3 and s["n_layers"] == 2 and s["total_params"] == 10
    assert s["architecture"] == "glm" and s["dtypes"]["F16"] == 1
