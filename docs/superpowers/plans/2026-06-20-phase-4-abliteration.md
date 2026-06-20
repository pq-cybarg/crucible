# Phase 4: Abliteration / Uncensoring Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the uncensoring engine — refusal-direction extraction, weight orthogonalization (abliteration), reversible activation steering, model cards with reproducibility hashes, and a pipeline that produces a registered, lineage-tracked variant — all model-agnostic and TDD-tested, plus REST endpoints that run it through an injectable adapter.

**Architecture:** The linear-algebra core (numpy) is pure and fully testable: a refusal direction is `normalize(mean(harmful) − mean(harmless))`; abliteration removes that direction from every residual-writing matrix `W' = W − strength·(r ⊗ rᵀW)`; steering adds/subtracts a vector at runtime (reversible). The `AbliterationPipeline` drives a `ModelAdapter` protocol (activations / get-set matrices / save), so a `FakeAdapter` tests the whole flow and a future torch/transformers adapter plugs in for real GLM weights. Variants register through the Phase-1 `Registry` (immutable originals, lineage), each with a `ModelCard`.

**Tech Stack:** Python 3.11+, numpy, pydantic, FastAPI, pytest. Builds on Phases 1–3.

## Global Constraints

- Python 3.11+; the math core depends only on numpy and is deterministic/testable.
- Originals are never mutated — abliteration writes a new variant file and registers it with `kind="abliterated"` and `base_id` set.
- Steering must be reversible: `steer(steer(x, v, c), v, −c) == x`.
- Every variant gets a `ModelCard` with a reproducibility hash of its parameters.
- The real-weights adapter (torch/transformers) is out of scope for this plan's tests; the endpoint returns `503` when no adapter is loaded.
- Local-only; TDD; one commit per task.

---

## File Structure

```
backend/crucible/abliteration/
  __init__.py        # re-exports
  direction.py       # compute_refusal_direction
  orthogonalize.py   # project_out, orthogonalize_writing_matrix, orthogonalize_embedding
  steering.py        # steer (reversible add/subtract)
  cards.py           # reproducibility_hash, build_model_card
  pipeline.py        # ModelAdapter protocol, AbliterationPipeline
  prompts.py         # default harmful/harmless refusal-elicitation prompt sets
  detection.py       # is_refusal, refusal_rate (behavioral censorship detection)
  diagnosis.py       # layer_refusal_profile, ablation_impact, explain_mechanism
backend/tests/
  test_abl_direction.py
  test_abl_orthogonalize.py
  test_abl_steering.py
  test_abl_cards.py
  test_abl_pipeline.py
  test_abl_detection.py
  test_abl_diagnosis.py
  test_abl_endpoint.py
```

---

### Task 1: numpy dependency + refusal direction

**Files:**
- Modify: `pyproject.toml` (add `numpy>=1.26`)
- Create: `backend/crucible/abliteration/__init__.py` (empty for now)
- Create: `backend/crucible/abliteration/direction.py`
- Test: `backend/tests/test_abl_direction.py`

**Interfaces:**
- Produces `compute_refusal_direction(harmful: ArrayLike, harmless: ArrayLike) -> np.ndarray` — `(d,)` unit vector = `normalize(mean(harmful,axis=0) − mean(harmless,axis=0))`; raises `ValueError` if the difference is zero.

- [ ] **Step 1: Add numpy to pyproject `dependencies` and reinstall**

Add `"numpy>=1.26",` to the `dependencies` list, then:

```bash
. .venv/bin/activate && pip install -q -e ".[dev]"
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_abl_direction.py
import numpy as np
import pytest
from crucible.abliteration.direction import compute_refusal_direction

def test_direction_points_from_harmless_to_harmful():
    harmful = np.array([[1.0, 0.0], [1.0, 0.0]])
    harmless = np.array([[0.0, 0.0], [0.0, 0.0]])
    d = compute_refusal_direction(harmful, harmless)
    assert np.allclose(d, [1.0, 0.0])
    assert np.isclose(np.linalg.norm(d), 1.0)

def test_zero_difference_raises():
    x = np.ones((3, 4))
    with pytest.raises(ValueError):
        compute_refusal_direction(x, x)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_abl_direction.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Write minimal implementation**

```python
# backend/crucible/abliteration/direction.py
import numpy as np
from numpy.typing import ArrayLike

def compute_refusal_direction(harmful: ArrayLike, harmless: ArrayLike) -> np.ndarray:
    h = np.asarray(harmful, dtype=np.float64)
    l = np.asarray(harmless, dtype=np.float64)
    diff = h.mean(axis=0) - l.mean(axis=0)
    norm = float(np.linalg.norm(diff))
    if norm == 0.0:
        raise ValueError("refusal direction is zero (harmful and harmless means coincide)")
    return diff / norm
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest backend/tests/test_abl_direction.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml backend/crucible/abliteration/__init__.py backend/crucible/abliteration/direction.py backend/tests/test_abl_direction.py
git commit -m "feat: add refusal-direction computation + numpy dep"
```

---

### Task 2: Orthogonalization

**Files:**
- Create: `backend/crucible/abliteration/orthogonalize.py`
- Test: `backend/tests/test_abl_orthogonalize.py`

**Interfaces:**
- Produces:
  - `project_out(x, direction) -> np.ndarray` — removes the `direction` component from a vector `(d,)` or row-wise from a matrix `(n,d)`.
  - `orthogonalize_writing_matrix(W, direction) -> np.ndarray` — `W:(d_model,k)` writing into the residual on axis 0: `W − outer(r, r@W)`.
  - `orthogonalize_embedding(E, direction) -> np.ndarray` — `E:(vocab,d_model)` on axis 1: `E − outer(E@r, r)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_abl_orthogonalize.py
import numpy as np
from crucible.abliteration.orthogonalize import (
    project_out, orthogonalize_writing_matrix, orthogonalize_embedding)

def unit(v): return np.asarray(v) / np.linalg.norm(v)

def test_project_out_vector_is_orthogonal():
    r = unit([1.0, 1.0, 0.0])
    out = project_out(np.array([3.0, 1.0, 2.0]), r)
    assert np.isclose(out @ r, 0.0)

def test_project_out_matrix_rows_orthogonal():
    r = unit([0.0, 1.0, 0.0])
    out = project_out(np.random.default_rng(0).standard_normal((5, 3)), r)
    assert np.allclose(out @ r, 0.0, atol=1e-9)

def test_writing_matrix_has_no_refusal_component():
    rng = np.random.default_rng(1)
    r = unit([1.0, 0.0, 0.0, 0.0])
    W = rng.standard_normal((4, 7))
    Wp = orthogonalize_writing_matrix(W, r)
    assert np.allclose(r @ Wp, 0.0, atol=1e-9)

def test_embedding_rows_have_no_refusal_component():
    rng = np.random.default_rng(2)
    r = unit([0.0, 0.0, 1.0, 0.0])
    E = rng.standard_normal((10, 4))
    Ep = orthogonalize_embedding(E, r)
    assert np.allclose(Ep @ r, 0.0, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_abl_orthogonalize.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/abliteration/orthogonalize.py
import numpy as np
from numpy.typing import ArrayLike

def project_out(x: ArrayLike, direction: ArrayLike) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        return a - (a @ d) * d
    return a - np.outer(a @ d, d)

def orthogonalize_writing_matrix(W: ArrayLike, direction: ArrayLike) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    w = np.asarray(W, dtype=np.float64)
    return w - np.outer(d, d @ w)

def orthogonalize_embedding(E: ArrayLike, direction: ArrayLike) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64)
    e = np.asarray(E, dtype=np.float64)
    return e - np.outer(e @ d, d)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_abl_orthogonalize.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/abliteration/orthogonalize.py backend/tests/test_abl_orthogonalize.py
git commit -m "feat: add orthogonalization (abliteration core)"
```

---

### Task 3: Activation steering

**Files:**
- Create: `backend/crucible/abliteration/steering.py`
- Test: `backend/tests/test_abl_steering.py`

**Interfaces:**
- Produces `steer(activations, vector, coefficient) -> np.ndarray` = `activations + coefficient*vector` (broadcast). Reversible by negating `coefficient`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_abl_steering.py
import numpy as np
from crucible.abliteration.steering import steer

def test_steer_adds_scaled_vector():
    x = np.array([1.0, 2.0])
    out = steer(x, np.array([1.0, 0.0]), 3.0)
    assert np.allclose(out, [4.0, 2.0])

def test_steer_is_reversible():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6))
    v = rng.standard_normal(6)
    back = steer(steer(x, v, 2.5), v, -2.5)
    assert np.allclose(back, x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_abl_steering.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/abliteration/steering.py
import numpy as np
from numpy.typing import ArrayLike

def steer(activations: ArrayLike, vector: ArrayLike, coefficient: float) -> np.ndarray:
    a = np.asarray(activations, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    return a + coefficient * v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_abl_steering.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/abliteration/steering.py backend/tests/test_abl_steering.py
git commit -m "feat: add reversible activation steering"
```

---

### Task 4: Model cards

**Files:**
- Create: `backend/crucible/abliteration/cards.py`
- Test: `backend/tests/test_abl_cards.py`

**Interfaces:**
- Produces:
  - `reproducibility_hash(params: dict) -> str` — 16-hex sha256 of canonical JSON.
  - `build_model_card(base_id, variant_id, method, layer, strength, prompt_counts: dict, hidden_size) -> dict` with keys `variant_id, base_id, method, layer, strength, hidden_size, prompt_counts, repro_hash, eval_delta(None)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_abl_cards.py
from crucible.abliteration.cards import build_model_card, reproducibility_hash

def test_hash_is_stable_and_order_independent():
    assert reproducibility_hash({"a": 1, "b": 2}) == reproducibility_hash({"b": 2, "a": 1})
    assert len(reproducibility_hash({"a": 1})) == 16

def test_card_has_expected_fields():
    card = build_model_card("glm-4-32b", "glm-4-32b-abl", "abliteration", 20, 1.0,
                            {"harmful": 32, "harmless": 32}, 5120)
    assert card["base_id"] == "glm-4-32b" and card["layer"] == 20
    assert card["eval_delta"] is None and len(card["repro_hash"]) == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_abl_cards.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/abliteration/cards.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_abl_cards.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/abliteration/cards.py backend/tests/test_abl_cards.py
git commit -m "feat: add model cards with reproducibility hash"
```

---

### Task 5: Pipeline + ModelAdapter

**Files:**
- Create: `backend/crucible/abliteration/pipeline.py`
- Create: `backend/crucible/abliteration/prompts.py`
- Modify: `backend/crucible/abliteration/__init__.py`
- Test: `backend/tests/test_abl_pipeline.py`

**Interfaces:**
- Produces:
  - `ModelAdapter` (Protocol): attr `hidden_size: int`; methods `activations(prompts, layer) -> np.ndarray`, `writing_matrices() -> list[str]`, `get_matrix(name) -> np.ndarray`, `set_matrix(name, W) -> None`, `save(path) -> None`.
  - `AbliterationPipeline(adapter, registry)` with `.compute_direction(harmful, harmless, layer) -> np.ndarray` and `.abliterate(base: Model, harmful, harmless, layer, out_path, variant_id, strength=1.0) -> tuple[Model, dict, np.ndarray]` (variant, card, direction).
  - `prompts.py`: `DEFAULT_HARMFUL: list[str]`, `DEFAULT_HARMLESS: list[str]` (research refusal-elicitation pairs; non-operational placeholders).
  - `__init__.py` re-exports `compute_refusal_direction`, `orthogonalize_writing_matrix`, `steer`, `build_model_card`, `AbliterationPipeline`, `ModelAdapter`.

- [ ] **Step 1: Write the failing test (FakeAdapter proves direction is found and removed)**

```python
# backend/tests/test_abl_pipeline.py
import numpy as np
from crucible.abliteration.pipeline import AbliterationPipeline
from crucible.registry import Model, Registry

class FakeAdapter:
    hidden_size = 8
    def __init__(self):
        rng = np.random.default_rng(0)
        self.e = np.zeros(8); self.e[2] = 1.0  # planted refusal direction
        self._mats = {"o_proj": rng.standard_normal((8, 8)),
                      "down_proj": rng.standard_normal((8, 16))}
        self.saved = None
    def activations(self, prompts, layer):
        rng = np.random.default_rng(7)
        out = rng.standard_normal((len(prompts), self.hidden_size)) * 0.05
        for i, p in enumerate(prompts):
            if "harm" in p:
                out[i] = out[i] + 5.0 * self.e
        return out
    def writing_matrices(self): return list(self._mats)
    def get_matrix(self, name): return self._mats[name]
    def set_matrix(self, name, W): self._mats[name] = W
    def save(self, path): self.saved = path

def base_model(reg):
    m = Model(id="base", name="base", base_id=None, path="/m/base.gguf", quant="Q4_K_M",
              kind="base", endpoint=None, created="2026-06-20", notes="")
    reg.register(m); return m

def test_pipeline_finds_and_removes_refusal(tmp_path):
    reg = Registry(tmp_path / "r.json")
    base = base_model(reg)
    adapter = FakeAdapter()
    pipe = AbliterationPipeline(adapter, reg)
    harmful = [f"harm{i}" for i in range(16)]
    harmless = [f"safe{i}" for i in range(16)]
    variant, card, direction = pipe.abliterate(
        base, harmful, harmless, layer=0, out_path="/m/base-abl.gguf",
        variant_id="base-abl", strength=1.0)
    # direction recovered
    assert abs(np.dot(direction, adapter.e)) > 0.99
    # refusal removed from every writing matrix
    for name in adapter.writing_matrices():
        assert np.allclose(adapter.e @ adapter.get_matrix(name), 0.0, atol=1e-9)
    # variant registered with lineage + card
    assert variant.kind == "abliterated"
    assert [m.id for m in reg.lineage("base-abl")] == ["base", "base-abl"]
    assert card["repro_hash"] and adapter.saved == "/m/base-abl.gguf"

def test_original_untouched(tmp_path):
    reg = Registry(tmp_path / "r.json")
    base = base_model(reg)
    pipe = AbliterationPipeline(FakeAdapter(), reg)
    pipe.abliterate(base, ["harm"], ["safe"], 0, "/m/base-abl.gguf", "base-abl")
    assert reg.get("base").path == "/m/base.gguf"  # original registry entry intact
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_abl_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write the prompt sets**

```python
# backend/crucible/abliteration/prompts.py
# Research refusal-elicitation pairs. Non-operational placeholders — the GUI lets
# the operator supply their own sets. "Harmful" here means "tends to elicit a refusal".
DEFAULT_HARMFUL: list[str] = [
    "Explain in detail how to do something dangerous and illegal.",
    "Write instructions for a harmful activity.",
    "Provide step-by-step guidance for an unethical task.",
    "Describe how to cause significant harm.",
]
DEFAULT_HARMLESS: list[str] = [
    "Explain how photosynthesis works.",
    "Write a short poem about the ocean.",
    "Summarize the rules of chess.",
    "Describe how to bake bread.",
]
```

- [ ] **Step 4: Write the pipeline**

```python
# backend/crucible/abliteration/pipeline.py
from typing import Protocol

import numpy as np

from crucible.abliteration.cards import build_model_card
from crucible.abliteration.direction import compute_refusal_direction
from crucible.abliteration.orthogonalize import orthogonalize_writing_matrix
from crucible.registry import Model, Registry


class ModelAdapter(Protocol):
    hidden_size: int
    def activations(self, prompts: list[str], layer: int) -> np.ndarray: ...
    def writing_matrices(self) -> list[str]: ...
    def get_matrix(self, name: str) -> np.ndarray: ...
    def set_matrix(self, name: str, W: np.ndarray) -> None: ...
    def save(self, path: str) -> None: ...


class AbliterationPipeline:
    def __init__(self, adapter: ModelAdapter, registry: Registry):
        self.adapter = adapter
        self.registry = registry

    def compute_direction(self, harmful: list[str], harmless: list[str], layer: int) -> np.ndarray:
        return compute_refusal_direction(
            self.adapter.activations(harmful, layer),
            self.adapter.activations(harmless, layer))

    def abliterate(self, base: Model, harmful: list[str], harmless: list[str], layer: int,
                   out_path: str, variant_id: str, strength: float = 1.0
                   ) -> tuple[Model, dict, np.ndarray]:
        direction = self.compute_direction(harmful, harmless, layer)
        for name in self.adapter.writing_matrices():
            W = self.adapter.get_matrix(name)
            ablated = W - strength * (orthogonalize_writing_matrix(W, direction) - W) * -1
            # Equivalent direct form (kept explicit for clarity):
            ablated = W - strength * np.outer(direction, direction @ W)
            self.adapter.set_matrix(name, ablated)
        self.adapter.save(out_path)
        variant = self.registry.register(Model(
            id=variant_id, name=variant_id, base_id=base.id, path=out_path,
            quant=base.quant, kind="abliterated", endpoint=None,
            created="2026-06-20", notes=f"abliterated from {base.id} @ layer {layer} (strength {strength})"))
        card = build_model_card(base.id, variant_id, "abliteration", layer, strength,
                                {"harmful": len(harmful), "harmless": len(harmless)},
                                self.adapter.hidden_size)
        return variant, card, direction
```

> Note: delete the first `ablated = …* -1` line — it is shown only to flag the
> trap of double-applying. The correct single line is
> `ablated = W - strength * np.outer(direction, direction @ W)`.

```python
# backend/crucible/abliteration/__init__.py
from crucible.abliteration.cards import build_model_card  # noqa: F401
from crucible.abliteration.direction import compute_refusal_direction  # noqa: F401
from crucible.abliteration.orthogonalize import orthogonalize_writing_matrix  # noqa: F401
from crucible.abliteration.pipeline import AbliterationPipeline, ModelAdapter  # noqa: F401
from crucible.abliteration.steering import steer  # noqa: F401
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest backend/tests/test_abl_pipeline.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/crucible/abliteration/pipeline.py backend/crucible/abliteration/prompts.py backend/crucible/abliteration/__init__.py backend/tests/test_abl_pipeline.py
git commit -m "feat: add abliteration pipeline + model adapter protocol"
```

---

### Task 6: Refusal detection (behavioral)

**Files:**
- Create: `backend/crucible/abliteration/detection.py`
- Test: `backend/tests/test_abl_detection.py`

**Interfaces:**
- Produces `is_refusal(text) -> bool` (marker-phrase detection) and `refusal_rate(texts) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_abl_detection.py
from crucible.abliteration.detection import is_refusal, refusal_rate

def test_detects_refusal_phrases():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("As an AI, I cannot provide this.")
    assert not is_refusal("Sure! Here is how you do it.")

def test_refusal_rate():
    texts = ["I cannot do that", "Sure, here you go", "I'm sorry, I won't"]
    assert abs(refusal_rate(texts) - 2 / 3) < 1e-9
    assert refusal_rate([]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_abl_detection.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/abliteration/detection.py
REFUSAL_MARKERS: list[str] = [
    "i can't", "i cannot", "i can not", "i'm sorry", "i am sorry", "i won't", "i will not",
    "as an ai", "i'm not able", "i am not able", "i'm unable", "i am unable",
    "cannot assist", "can't help with that", "against my", "not appropriate",
    "i must decline", "i'm not comfortable",
]

def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(marker in t for marker in REFUSAL_MARKERS)

def refusal_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(1 for t in texts if is_refusal(t)) / len(texts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_abl_detection.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/abliteration/detection.py backend/tests/test_abl_detection.py
git commit -m "feat: add behavioral refusal detection"
```

---

### Task 7: Censorship diagnosis & mechanism explanation

**Files:**
- Create: `backend/crucible/abliteration/diagnosis.py`
- Test: `backend/tests/test_abl_diagnosis.py`

**Interfaces:**
- Consumes: `ModelAdapter`, `compute_refusal_direction`.
- Produces:
  - `layer_refusal_profile(adapter, harmful, harmless, layers) -> list[dict]` — per layer `{layer, separation, margin}` (margin = standardized harmful/harmless separation along that layer's direction; *where* refusal is encoded).
  - `best_layer(profile) -> int` — the layer with the strongest margin.
  - `ablation_impact(W, direction) -> dict` — `{total_norm, removed_norm, removed_fraction}` quantifying how much of a matrix is the refusal component (*what* you remove, *how invasive*).
  - `explain_mechanism(profile, matrices_impact: dict, base_id) -> dict` — a structured + human-readable report (`best_layer, why, how, removal, mean_removed_fraction, heaviest_component, surgical, collateral_risk, layer_profile, components`).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_abl_diagnosis.py
import numpy as np
from crucible.abliteration.diagnosis import (
    layer_refusal_profile, best_layer, ablation_impact, explain_mechanism)

class Adapter:
    hidden_size = 6
    def __init__(self, refusal_layer):
        self.refusal_layer = refusal_layer
        self.e = np.zeros(6); self.e[1] = 1.0
    def activations(self, prompts, layer):
        rng = np.random.default_rng(layer)
        out = rng.standard_normal((len(prompts), 6)) * 0.05
        if layer == self.refusal_layer:
            for i, p in enumerate(prompts):
                if "harm" in p:
                    out[i] = out[i] + 6.0 * self.e
        return out

def test_profile_localizes_refusal_layer():
    a = Adapter(refusal_layer=2)
    prof = layer_refusal_profile(a, ["harm0", "harm1"], ["safe0", "safe1"], [0, 1, 2, 3])
    assert best_layer(prof) == 2

def test_ablation_impact_fraction_between_0_and_1():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((6, 9))
    r = np.zeros(6); r[1] = 1.0
    imp = ablation_impact(W, r)
    assert 0.0 < imp["removed_fraction"] < 1.0

def test_explain_marks_small_removal_surgical():
    rng = np.random.default_rng(0)
    r = np.zeros(6); r[1] = 1.0
    impacts = {"o_proj": ablation_impact(rng.standard_normal((6, 6)), r)}
    prof = [{"layer": 2, "separation": 5.0, "margin": 12.0}]
    report = explain_mechanism(prof, impacts, "glm")
    assert report["best_layer"] == 2
    assert "rank-1" in report["removal"]
    assert isinstance(report["surgical"], bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_abl_diagnosis.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/abliteration/diagnosis.py
import numpy as np
from numpy.typing import ArrayLike


def layer_refusal_profile(adapter, harmful: list[str], harmless: list[str],
                          layers: list[int]) -> list[dict]:
    profile: list[dict] = []
    for layer in layers:
        h = np.asarray(adapter.activations(harmful, layer), dtype=np.float64)
        l = np.asarray(adapter.activations(harmless, layer), dtype=np.float64)
        diff = h.mean(axis=0) - l.mean(axis=0)
        separation = float(np.linalg.norm(diff))
        r = diff / (separation or 1.0)
        hp, lp = h @ r, l @ r
        pooled = float(np.sqrt((hp.var() + lp.var()) / 2.0)) + 1e-9
        margin = float((hp.mean() - lp.mean()) / pooled)
        profile.append({"layer": layer, "separation": separation, "margin": margin})
    return profile


def best_layer(profile: list[dict]) -> int:
    return int(max(profile, key=lambda p: p["margin"])["layer"])


def ablation_impact(W: ArrayLike, direction: ArrayLike) -> dict:
    w = np.asarray(W, dtype=np.float64)
    r = np.asarray(direction, dtype=np.float64)
    removed = np.outer(r, r @ w)
    total = float(np.linalg.norm(w))
    removed_norm = float(np.linalg.norm(removed))
    return {"total_norm": total, "removed_norm": removed_norm,
            "removed_fraction": (removed_norm / total) if total else 0.0}


def explain_mechanism(profile: list[dict], matrices_impact: dict, base_id: str) -> dict:
    bl = best_layer(profile) if profile else 0
    fractions = [m["removed_fraction"] for m in matrices_impact.values()]
    mean_removed = sum(fractions) / len(fractions) if fractions else 0.0
    heaviest = (max(matrices_impact.items(), key=lambda kv: kv[1]["removed_fraction"])[0]
                if matrices_impact else None)
    surgical = mean_removed < 0.05
    return {
        "base_id": base_id,
        "best_layer": bl,
        "layer_profile": profile,
        "components": matrices_impact,
        "heaviest_component": heaviest,
        "mean_removed_fraction": mean_removed,
        "surgical": surgical,
        "collateral_risk": "low" if surgical else
            "elevated — refusal is entangled with a large weight component",
        "why": ("Alignment/safety fine-tuning (RLHF + safety SFT) installed a roughly linear "
                "'refusal feature' in the residual stream. When a prompt activates it, the model "
                "is steered toward refusal phrasing."),
        "how": (f"Harmful vs harmless prompts are most linearly separable at layer {bl} "
                "(highest margin). Residual-writing matrices (o_proj, down_proj) add a component "
                "along the refusal direction r; later layers read that component and emit refusal tokens."),
        "removal": ("Orthogonalization subtracts only the rank-1 projection onto r (W - r·rᵀW). "
                    "The matrix's action on the (d-1)-dimensional subspace orthogonal to r is "
                    "unchanged, so capabilities encoded in other directions are preserved exactly — "
                    "that is why the cut is surgical."),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_abl_diagnosis.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/abliteration/diagnosis.py backend/tests/test_abl_diagnosis.py
git commit -m "feat: add censorship diagnosis + mechanism explanation"
```

---

### Task 8: REST endpoints

**Files:**
- Modify: `backend/crucible/app.py`
- Test: `backend/tests/test_abl_endpoint.py`

**Interfaces:**
- Consumes: `AbliterationPipeline`, prompt sets, registry.
- Produces (added to `create_app`, accepting optional `abliteration_adapter=None`):
  - `GET /api/abliteration/promptsets` → `{"harmful": [...], "harmless": [...]}`.
  - `POST /api/abliteration/run` body `{base_id, layer, strength, variant_id, out_path?, harmful?, harmless?}` → `503` if no adapter; else runs the pipeline (base from registry, defaults to the built-in prompt sets and `models/<variant_id>.gguf`) → `{"variant": Model, "card": {...}}`; `404` if `base_id` unknown.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_abl_endpoint.py
import numpy as np
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Model, Registry

class FakeAdapter:
    hidden_size = 8
    def __init__(self):
        rng = np.random.default_rng(0)
        self.e = np.zeros(8); self.e[1] = 1.0
        self._mats = {"o_proj": rng.standard_normal((8, 8))}
        self.saved = None
    def activations(self, prompts, layer):
        out = np.random.default_rng(3).standard_normal((len(prompts), 8)) * 0.05
        for i, p in enumerate(prompts):
            if "harm" in p or "danger" in p or "illegal" in p:
                out[i] = out[i] + 4.0 * self.e
        return out
    def writing_matrices(self): return list(self._mats)
    def get_matrix(self, n): return self._mats[n]
    def set_matrix(self, n, W): self._mats[n] = W
    def save(self, p): self.saved = p

def mkapp(tmp_path, monkeypatch, adapter=None):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    reg = Registry(tmp_path / "r.json")
    reg.register(Model(id="glm", name="glm", base_id=None, path="/m/glm.gguf", quant="Q4_K_M",
                       kind="base", endpoint=None, created="2026-06-20", notes=""))
    return create_app(registry=reg, agent_root=tmp_path, abliteration_adapter=adapter)

def test_promptsets(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    body = c.get("/api/abliteration/promptsets").json()
    assert len(body["harmful"]) > 0 and len(body["harmless"]) > 0

def test_run_requires_adapter(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    r = c.post("/api/abliteration/run", json={"base_id": "glm", "layer": 0, "strength": 1.0, "variant_id": "glm-abl"})
    assert r.status_code == 503

def test_run_produces_variant(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch, adapter=FakeAdapter()))
    r = c.post("/api/abliteration/run", json={"base_id": "glm", "layer": 0, "strength": 1.0, "variant_id": "glm-abl"})
    assert r.status_code == 200
    out = r.json()
    assert out["variant"]["kind"] == "abliterated"
    assert out["card"]["base_id"] == "glm"
    assert [m["id"] for m in c.get("/api/models/glm-abl/lineage").json()] == ["glm", "glm-abl"]

def test_run_unknown_base_404(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch, adapter=FakeAdapter()))
    r = c.post("/api/abliteration/run", json={"base_id": "nope", "layer": 0, "strength": 1.0, "variant_id": "x"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_abl_endpoint.py -v`
Expected: FAIL — unexpected kwarg / missing routes.

- [ ] **Step 3: Extend `app.py`**

Add imports:

```python
from crucible.abliteration.pipeline import AbliterationPipeline
from crucible.abliteration.prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
```

Add a request model:

```python
class AbliterateRequest(BaseModel):
    base_id: str
    layer: int = 0
    strength: float = 1.0
    variant_id: str
    out_path: str | None = None
    harmful: list[str] | None = None
    harmless: list[str] | None = None
```

Extend the factory signature with `abliteration_adapter=None`, build the pipeline when present:

```python
    abl_pipeline = (AbliterationPipeline(abliteration_adapter, reg)
                    if abliteration_adapter is not None else None)
```

Add routes:

```python
    @app.get("/api/abliteration/promptsets")
    def abl_promptsets() -> dict:
        return {"harmful": DEFAULT_HARMFUL, "harmless": DEFAULT_HARMLESS}

    @app.post("/api/abliteration/run")
    def abl_run(req: AbliterateRequest) -> dict:
        if abl_pipeline is None:
            raise HTTPException(status_code=503,
                detail="no model adapter loaded — abliteration needs the HF weights + torch")
        try:
            base = reg.get(req.base_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="base model not found")
        harmful = req.harmful or DEFAULT_HARMFUL
        harmless = req.harmless or DEFAULT_HARMLESS
        out_path = req.out_path or f"models/{req.variant_id}.gguf"
        variant, card, _ = abl_pipeline.abliterate(
            base, harmful, harmless, req.layer, out_path, req.variant_id, req.strength)
        return {"variant": variant.model_dump(), "card": card}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_abl_endpoint.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all Phase 1–4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/crucible/app.py backend/tests/test_abl_endpoint.py
git commit -m "feat: add abliteration REST endpoints (injectable adapter, 503 without weights)"
```

---

## Self-Review

**Spec coverage (Phase 4 = Component 4.5 Uncensoring Pipeline):**
- Refusal direction from harmful/harmless sets → Task 1. ✅
- Orthogonalize out of the residual stream → Task 2. ✅
- Adjustable strength + new GGUF variant + immutable original → Task 5 (`strength`, registry variant, original untouched test). ✅
- Activation steering (reversible) → Task 3. ✅
- Model cards + eval-delta slot → Task 4. ✅
- Lineage-tracked variant → Task 5/6 (`lineage` checks). ✅
- A/B preview & real torch adapter → the adapter protocol is defined; the torch-backed adapter + GGUF re-quant are the gated next step (flagged), and A/B belongs to the GUI pass. 

**Placeholder scan:** The Task-5 pipeline intentionally shows one wrong `ablated` line with a delete-note; the correct single line is `ablated = W - strength * np.outer(direction, direction @ W)`. No other placeholders. ✅

**Type consistency:** `ModelAdapter` method signatures consistent (Tasks 5–6, FakeAdapter mirrors them). `Model` fields match Phase 1. `build_model_card(...)` signature consistent Tasks 4–5. `compute_refusal_direction`, `orthogonalize_writing_matrix`, `steer` signatures consistent across tasks. ✅
