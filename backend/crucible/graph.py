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


# --- graph node PATTERNS: cascade (cheap -> escalate) + verifier ensemble (fan out -> vote) ---
# These are the two workhorse compositions. They're pure (producers/outputs are passed in), so the
# control logic is unit-tested here; the endpoint supplies producers that actually call models.

def make_acceptor(min_len: int = 1, must_include=None, must_exclude=None) -> Callable[[str], bool]:
    """Build an accept predicate for a cascade: an output passes if it's at least `min_len` chars,
    contains every `must_include` substring, and none of `must_exclude` (e.g. exclude "i don't
    know" / "as an ai" so a weak/hedged answer escalates to the stronger model)."""
    inc = [s.lower() for s in (must_include or [])]
    exc = [s.lower() for s in (must_exclude or [])]

    def accept(out: str) -> bool:
        o = (out or "").strip()
        if len(o) < max(1, min_len):
            return False
        lo = o.lower()
        if any(s not in lo for s in inc):
            return False
        if any(s in lo for s in exc):
            return False
        return True

    return accept


def cascade(candidates: list[tuple], accept: Callable[[str], bool]) -> dict:
    """Cheap -> escalate. Run producers in order until one is ACCEPTED; return that output. If none
    pass, return the last (strongest) attempt. candidates: [(label, producer)] where producer()->str.
    Pure given the producers, so the escalation logic is testable without a live model."""
    tried: list[str] = []
    label, out = None, ""
    for label, producer in candidates:
        out = producer() or ""
        tried.append(label)
        if accept(out):
            return {"chosen": label, "output": out, "tried": tried,
                    "escalated": len(tried) > 1, "accepted": True}
    return {"chosen": label, "output": out, "tried": tried,
            "escalated": len(tried) > 1, "accepted": False}


def majority(outputs: list[str]) -> str:
    """Most-common output (normalized on stripped text); ties resolve to the first seen. [] -> ''."""
    from collections import Counter
    norm = [(o or "").strip() for o in outputs if (o or "").strip()]
    if not norm:
        return ""
    counts = Counter(norm)
    return max(norm, key=lambda o: (counts[o], -norm.index(o)))


def vote(outputs: list[str], strategy: str = "majority") -> dict:
    """Verifier-ensemble merge over N outputs. 'majority' -> most common (with an agreement score);
    'concat' -> all joined; 'first' -> first non-empty. Returns {strategy, result, n, agreement}."""
    clean = [(o or "").strip() for o in outputs]
    nonempty = [o for o in clean if o]
    if strategy == "concat":
        result = "\n\n".join(clean)
    elif strategy == "first":
        result = nonempty[0] if nonempty else ""
    else:
        result = majority(outputs)
    agreement = (sum(1 for o in nonempty if o == result) / len(nonempty)) if nonempty else 0.0
    return {"strategy": strategy, "result": result, "n": len(outputs), "agreement": round(agreement, 3)}


def stage_text(value) -> str:
    """Extract the passable text from a stage output — cascade/vote stages return rich dicts, so
    downstream stages read their 'output'/'result' field rather than stringifying the whole dict."""
    if isinstance(value, dict):
        return str(value.get("output", value.get("result", "")))
    return "" if value is None else str(value)
