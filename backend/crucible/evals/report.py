from __future__ import annotations
from crucible.evals.published import PUBLISHED


def build_comparison(measured: dict, published: dict = PUBLISHED) -> dict:
    metrics: list[str] = []
    for model in published.values():
        for metric in model:
            if metric not in metrics:
                metrics.append(metric)
    for metric in measured:
        if metric not in metrics:
            metrics.append(metric)
    rows = []
    for metric in metrics:
        rows.append({
            "metric": metric,
            "measured": measured.get(metric),
            "models": {name: model.get(metric, {"value": None, "source": "cite"})
                       for name, model in published.items()},
        })
    return {"rows": rows,
            "provenance": "measured = run locally by Crucible; model columns = published/cited."}
