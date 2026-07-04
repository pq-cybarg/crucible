import numpy as np

from crucible.training import LoRATrainer, validate_dataset


def test_validate_dataset_filters_malformed():
    ds = validate_dataset([
        {"prompt": "hi", "response": "hello"},
        {"prompt": "", "response": "x"},          # empty prompt -> dropped
        {"prompt": "q", "response": ""},          # empty response -> dropped
        "not a dict",                              # dropped
        {"prompt": " a ", "response": " b "},     # trimmed
    ])
    assert ds == [{"prompt": "hi", "response": "hello"}, {"prompt": "a", "response": "b"}]


def test_lora_trainer_reduces_loss_and_fits_lowrank_target():
    rng = np.random.default_rng(0)
    in_dim, out_dim, r = 6, 5, 2
    X = rng.standard_normal((200, in_dim))
    W_base = rng.standard_normal((out_dim, in_dim))
    # target = base + a rank-r delta the trainer should be able to learn
    U = rng.standard_normal((out_dim, r)); V = rng.standard_normal((r, in_dim))
    W_target = W_base + U @ V
    Y = X @ W_target.T

    tr = LoRATrainer(in_dim, out_dim, rank=r, lr=3e-2, epochs=3000).fit(X, Y, W_base)
    assert tr.history[0] > 0
    assert tr.final_loss() < 0.02 * tr.history[0]       # big loss reduction (converges to ~0)
    assert tr.history[-1] <= tr.history[len(tr.history) // 2]   # monotone-ish, no divergence
    # the trained adapter reconstructs the target mapping well
    delta = tr.lora().delta()
    pred = X @ (W_base + delta).T
    assert np.allclose(pred, Y, atol=0.4)


def test_lora_trainer_zero_init_delta():
    tr = LoRATrainer(4, 3, rank=2)
    assert np.allclose(tr.lora().delta(), 0.0)          # B=0 => no change before training


def test_lora_trainer_learns_from_zero_base():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((150, 4))
    W = rng.standard_normal((3, 4))
    Y = X @ W.T
    tr = LoRATrainer(4, 3, rank=3, lr=5e-2, epochs=800).fit(X, Y)   # no W_base
    assert tr.final_loss() < 0.1 * tr.history[0]
