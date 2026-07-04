from __future__ import annotations
# Retraining pipeline. Beyond the training-FREE edits (abliteration/steering), this actually
# TRAINS a LoRA by gradient descent. The numpy core here optimizes a low-rank correction to
# fit target behavior in activation space — deterministic and unit-tested (it recovers a
# planted low-rank transform and drives loss down). The real token-level SFT on a live model
# (train_lora_torch) is a model path that runs when torch is present. Together they are the
# "retrain it" half of the pipeline: gather data -> train an adapter -> attach/evaluate.
import numpy as np
from numpy.typing import ArrayLike

from crucible.abliteration.lora import LoRA


def validate_dataset(pairs: list) -> list[dict]:
    """Normalize a fine-tune dataset of {prompt, response} pairs; drop malformed rows."""
    out = []
    for row in pairs or []:
        if not isinstance(row, dict):
            continue
        p, r = str(row.get("prompt", "")).strip(), str(row.get("response", "")).strip()
        if p and r:
            out.append({"prompt": p, "response": r})
    return out


class LoRATrainer:
    """Train a LoRA delta (A, B) by gradient descent so that X @ (W_base + delta)ᵀ approaches
    a target Y. This is real optimization (not a closed-form cut): the pure core of adapter
    retraining, in activation space. Standard LoRA init (B=0 => initial delta is zero)."""

    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, lr: float = 1e-2,
                 epochs: int = 300, alpha: float | None = None, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.A = rng.standard_normal((rank, in_dim)) * (1.0 / np.sqrt(in_dim))
        self.B = np.zeros((out_dim, rank))
        self.rank = rank
        self.lr = lr
        self.epochs = epochs
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.history: list[float] = []

    def _delta(self) -> np.ndarray:
        return (self.alpha / self.rank) * (self.B @ self.A)

    def fit(self, X: ArrayLike, Y: ArrayLike, W_base: ArrayLike | None = None) -> "LoRATrainer":
        Xa = np.asarray(X, dtype=np.float64)
        Ya = np.asarray(Y, dtype=np.float64)
        n = Xa.shape[0]
        Wb = np.zeros((self.B.shape[0], self.A.shape[1])) if W_base is None \
            else np.asarray(W_base, dtype=np.float64)
        scale = self.alpha / self.rank
        for _ in range(self.epochs):
            pred = Xa @ (Wb + self._delta()).T          # (n, out)
            err = pred - Ya
            self.history.append(float((err ** 2).mean()))
            g = (2.0 / n) * err                         # dL/dpred
            dDelta = g.T @ Xa                           # dL/ddelta  (out, in)
            dB = scale * (dDelta @ self.A.T)            # both grads from the SAME state
            dA = scale * (self.B.T @ dDelta)
            # gradient-norm clip keeps the B·A product stable (avoids the low-rank blow-up)
            for grad in (dB, dA):
                gn = float(np.linalg.norm(grad))
                if gn > 1.0:
                    grad *= 1.0 / gn
            self.B -= self.lr * dB
            self.A -= self.lr * dA
        return self

    def lora(self) -> LoRA:
        return LoRA(self.A, self.B, self.alpha)

    def final_loss(self) -> float:
        return self.history[-1] if self.history else float("nan")


def train_lora_torch(model_path: str, dataset: list[dict], target_modules=("q_proj", "v_proj"),
                     rank: int = 8, epochs: int = 1, lr: float = 2e-4, save_path: str | None = None):
    """Real token-level SFT: attach LoRA adapters to a live HF model and train on the
    prompt/response pairs, then optionally save the adapter. Requires torch + peft; this is a
    model path (not unit-tested). Returns a summary dict."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = validate_dataset(dataset)
    if not data:
        raise ValueError("no valid {prompt, response} pairs")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank * 2,
                                             target_modules=list(target_modules), task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        for row in data:
            text = row["prompt"] + "\n" + row["response"] + (tok.eos_token or "")
            ids = tok(text, return_tensors="pt")
            out = model(**ids, labels=ids["input_ids"])
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(float(out.loss.item()))
    if save_path:
        model.save_pretrained(save_path)
    return {"n_examples": len(data), "epochs": epochs, "final_loss": losses[-1] if losses else None,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "saved": bool(save_path)}
