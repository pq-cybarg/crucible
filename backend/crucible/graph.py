from __future__ import annotations
# Model graphs — "model systems / subsystems". A single request rarely needs one model: real
# systems chain subsystems (speech-to-text -> language model -> text-to-speech; image -> VLM ->
# text -> image), each stage a different model/modality/tool, edges possibly asymmetric. This
# is a small DAG engine: define stages with typed inputs, topologically order them, and run
# each with an injected stage runner (so the control flow is pure + unit-tested; the runner
# does the actual model/tool calls). Cascades, verifier ensembles, and mixture-of-models all
# express as graphs.
from typing import Callable


def topo_order(stages: list[dict]) -> list[str]:
    """Topological order of stage ids (Kahn). Raises ValueError on a cycle or a missing dep."""
    ids = [s["id"] for s in stages]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate stage id")
    by_id = {s["id"]: s for s in stages}
    indeg = {i: 0 for i in ids}
    for s in stages:
        for dep in s.get("inputs", []):
            if dep not in by_id:
                raise ValueError(f"stage '{s['id']}' depends on unknown stage '{dep}'")
            indeg[s["id"]] += 1
    ready = [i for i in ids if indeg[i] == 0]
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for s in stages:
            if n in s.get("inputs", []):
                indeg[s["id"]] -= 1
                if indeg[s["id"]] == 0:
                    ready.append(s["id"])
    if len(order) != len(ids):
        raise ValueError("graph has a cycle")
    return order


def execute_graph(stages: list[dict], run_stage: Callable[[dict, dict], object],
                  initial: object = None) -> dict:
    """Execute a stage DAG. run_stage(stage, inputs) -> output, where inputs maps each input
    stage id to its output (root stages — no inputs — receive {'__initial__': initial}).
    Returns {stage_id: output} for every stage. A stage that raises stops the whole run."""
    order = topo_order(stages)
    by_id = {s["id"]: s for s in stages}
    outputs: dict = {}
    for sid in order:
        stage = by_id[sid]
        deps = stage.get("inputs", [])
        inputs = {d: outputs[d] for d in deps} if deps else {"__initial__": initial}
        outputs[sid] = run_stage(stage, inputs)
    return outputs


def final_outputs(stages: list[dict], outputs: dict) -> dict:
    """The outputs of the terminal stages (no other stage depends on them) — the graph result."""
    depended = set()
    for s in stages:
        depended.update(s.get("inputs", []))
    terminals = [s["id"] for s in stages if s["id"] not in depended]
    return {t: outputs.get(t) for t in terminals}
