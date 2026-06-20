import hashlib
import json


def reproducibility_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def build_model_card(base_id: str, variant_id: str, method: str, layer: int,
                     strength: float, prompt_counts: dict, hidden_size: int) -> dict:
    params = {"base": base_id, "method": method, "layer": layer, "strength": strength,
              "harmful": prompt_counts.get("harmful"), "harmless": prompt_counts.get("harmless")}
    return {
        "variant_id": variant_id, "base_id": base_id, "method": method, "layer": layer,
        "strength": strength, "hidden_size": hidden_size, "prompt_counts": prompt_counts,
        "repro_hash": reproducibility_hash(params), "eval_delta": None,
    }
