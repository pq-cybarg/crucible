import hashlib
import json

from crucible.abliteration.detection import refusal_rate
from crucible.abliteration.subspace import refusal_subspace


def layer_band(n_layers: int, name: str) -> list[int]:
    if name == "late_half":
        return list(range(n_layers // 2, n_layers))
    if name == "last_quarter":
        return list(range(3 * n_layers // 4, n_layers))
    return list(range(n_layers))


def recipe_hash(recipe: dict) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]


def autotune(adapter, harmful, harmless, configs, max_new_tokens=22):
    ah = adapter.all_layer_activations(harmful)    # (n, L+1, d)
    al = adapter.all_layer_activations(harmless)
    base_h = [adapter.generate(p, max_new_tokens) for p in harmful]
    base_b = [adapter.generate(p, max_new_tokens) for p in harmless]
    base_refusal = refusal_rate(base_h)
    base_over = refusal_rate(base_b)

    n = adapter.num_layers
    results = []
    for cfg in configs:
        band = layer_band(n, cfg["band"])
        # decoder layer j output == hidden_states[j+1]
        dirs = {j: refusal_subspace(ah[:, j + 1, :], al[:, j + 1, :], cfg["rank"])[0] for j in band}
        gh = [adapter.ablate_generate_banded(p, dirs, cfg["coefficient"], max_new_tokens) for p in harmful]
        gb = [adapter.ablate_generate_banded(p, dirs, cfg["coefficient"], max_new_tokens) for p in harmless]
        r, o = refusal_rate(gh), refusal_rate(gb)
        results.append({**cfg, "harmful_refusal": r, "benign_over_refusal": o,
                        "score": (base_refusal - r) - o})

    best = max(results, key=lambda x: x["score"]) if results else None
    recipe = {"band": best["band"], "rank": best["rank"], "coefficient": best["coefficient"]} if best else {}
    return {"baseline": {"harmful_refusal": base_refusal, "benign_over_refusal": base_over},
            "results": results, "best": best, "recipe": recipe,
            "recipe_hash": recipe_hash(recipe) if recipe else "", "weights_modified": False}
