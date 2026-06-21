# Capability retention: run a real lm-eval task on the base vs the abliterated weights
# (lm-eval HF backend, in-process) to prove the cut was surgical, not lobotomizing.
from crucible.evals.lmeval import parse_lmeval_results

_PRIMARY = ("exact_match", "acc_norm", "acc")


def lm_eval_hf(model_path: str, task: str, limit: int, device: str | None = None) -> list[dict]:
    import lm_eval
    import torch
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = lm_eval.simple_evaluate(
        model="hf",
        model_args={"pretrained": model_path, "device": device, "dtype": "float32"},
        tasks=[task], limit=limit, apply_chat_template=True)
    return parse_lmeval_results(out["results"])


def primary_metric(rows: list[dict]) -> dict | None:
    for name in _PRIMARY:
        for r in rows:
            if r["metric"] == name:
                return r
    return rows[0] if rows else None


def capability_delta(base_path: str, variant_path: str, task: str, limit: int) -> dict:
    base_rows = lm_eval_hf(base_path, task, limit)
    var_rows = lm_eval_hf(variant_path, task, limit)
    b = primary_metric(base_rows)
    v = primary_metric(var_rows)
    bs = b["value"] if b else None
    vs = v["value"] if v else None
    return {"task": task,
            "base_score": bs, "variant_score": vs,
            "metric": b["metric"] if b else None,
            "retention": (vs / bs) if (bs not in (None, 0) and vs is not None) else None,
            "base_rows": base_rows, "variant_rows": var_rows}
