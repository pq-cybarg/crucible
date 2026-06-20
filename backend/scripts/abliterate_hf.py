#!/usr/bin/env python3
"""Standalone: diagnose + abliterate an HF model without the server.
  PYTHONPATH=backend python backend/scripts/abliterate_hf.py <model> [--layer N] [--strength S] [--out PATH]
"""
import argparse
import json

from crucible.abliteration.diagnosis import (
    ablation_impact, best_layer, explain_mechanism, layer_refusal_profile)
from crucible.abliteration.direction import compute_refusal_direction
from crucible.abliteration.pipeline import AbliterationPipeline
from crucible.abliteration.prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
from crucible.abliteration.torch_adapter import TorchModelAdapter
from crucible.registry import Model, Registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF model id or local path")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out", default="models/abliterated")
    ap.add_argument("--variant", default="abliterated")
    a = ap.parse_args()

    adapter = TorchModelAdapter.load(a.model)
    layers = list(range(adapter.num_layers))
    profile = layer_refusal_profile(adapter, DEFAULT_HARMFUL, DEFAULT_HARMLESS, layers)
    layer = a.layer if a.layer is not None else best_layer(profile)
    direction = compute_refusal_direction(
        adapter.activations(DEFAULT_HARMFUL, layer),
        adapter.activations(DEFAULT_HARMLESS, layer))
    impacts = {n: ablation_impact(adapter.get_matrix(n), direction) for n in adapter.writing_matrices()}
    print(json.dumps(explain_mechanism(profile, impacts, a.model), indent=2, default=str))

    reg = Registry("/tmp/crucible_cli_registry.json")
    bid = a.model.replace("/", "_")
    if bid not in [m.id for m in reg.list()]:
        reg.register(Model(id=bid, name=a.model, base_id=None, path=a.model, quant="bf16",
                           kind="base", endpoint=None, created="2026-06-20", notes=""))
    variant, card, _ = AbliterationPipeline(adapter, reg).abliterate(
        reg.get(bid), DEFAULT_HARMFUL, DEFAULT_HARMLESS, layer, a.out, a.variant, a.strength)
    print(f"\nsaved variant '{variant.id}' -> {a.out}  (repro {card['repro_hash']}, layer {layer})")


if __name__ == "__main__":
    main()
