import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from crucible.abliteration.torch_adapter import TorchModelAdapter  # noqa: E402


class StubTok:
    chat_template = None

    def __call__(self, text, return_tensors=None):
        ids = [(ord(c) % 50) + 1 for c in text][:8] or [1]

        class O:
            input_ids = torch.tensor([ids])

        return O()


def tiny_adapter():
    cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64)
    return TorchModelAdapter(Qwen2ForCausalLM(cfg), StubTok(), device="cpu")


def test_structure():
    a = tiny_adapter()
    assert a.hidden_size == 16 and a.num_layers == 2
    names = a.writing_matrices()
    assert "model.layers.0.self_attn.o_proj.weight" in names and len(names) == 4


def test_activations_shape():
    a = tiny_adapter()
    acts = a.activations(["hello", "world"], 1)
    assert acts.shape == (2, 16)


def test_get_set_matrix_roundtrip():
    a = tiny_adapter()
    name = "model.layers.0.mlp.down_proj.weight"
    original = a.get_matrix(name)
    a.set_matrix(name, original * 0.0)
    assert np.allclose(a.get_matrix(name), 0.0)
    assert original.shape[0] == a.hidden_size
