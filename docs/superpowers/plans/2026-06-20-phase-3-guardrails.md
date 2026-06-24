# Phase 3: Guardrails Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GUI-manageable guardrails engine — a layered safety stack (system-prompt presets, regex/redaction filters, and a constitutional self-critique pass) that can be dialed up or down, applied per conversation stage, and shows exactly what each layer did — plus REST endpoints and a focused integration into the agent endpoint.

**Architecture:** Each guardrail layer is independent and reports structured `GuardrailAction`s. The `GuardrailsEngine` orchestrates layers per `stage` (`input`/`output`) and returns the (possibly transformed) text plus the action trail. The constitutional layer is model-agnostic: it takes a `critic` callable, so it is fully testable without a live model and wires to the real model later. Config persists to JSON via a small store, mirroring the Phase-1 `Registry`.

**Tech Stack:** Python 3.11+, pydantic v2, FastAPI, pytest. Builds on Phases 1–2 (`crucible.app`, `crucible.agent`, `crucible.config`).

## Global Constraints

- Python 3.11+; layers must be independently testable and report structured actions.
- Guardrails are tunable: a master `enabled` flag plus per-layer toggles; dialing to "unrestricted" is a first-class preset.
- The constitutional layer never calls the network directly — it takes a `critic: Callable[[str, str], str]`.
- Config persists to `<data_dir>/guardrails.json`.
- Local-only; TDD; one commit per task minimum.

---

## File Structure

```
backend/crucible/guardrails/
  __init__.py        # re-exports + build_engine()
  base.py            # Stage, GuardrailAction, GuardrailResult, RegexRule, GuardrailConfig
  presets.py         # SystemPromptPreset + PRESETS + get_preset()
  filters.py         # RegexFilter (block/redact, per stage)
  constitution.py    # ConstitutionalCritic (revise via critic callable)
  engine.py          # GuardrailsEngine (system_prompt + apply per stage)
  store.py           # GuardrailStore (load/save GuardrailConfig)
backend/tests/
  test_guardrails_presets.py
  test_guardrails_filters.py
  test_guardrails_constitution.py
  test_guardrails_engine.py
  test_guardrails_store.py
  test_guardrails_endpoint.py
```

---

### Task 1: Base types

**Files:**
- Create: `backend/crucible/guardrails/base.py`
- Test: `backend/tests/test_guardrails_base.py`

**Interfaces:**
- Produces:
  - `Stage = Literal["input","output"]`.
  - `GuardrailAction(layer: str, stage: Stage, action: Literal["inject","block","redact","revise","pass"], detail: str)`.
  - `GuardrailResult(text: str, blocked: bool, actions: list[GuardrailAction])`.
  - `RegexRule(pattern: str, mode: Literal["block","redact"], label: str, stages: list[Stage] = ["input","output"])`.
  - `GuardrailConfig(enabled: bool = True, preset_id: str = "balanced", regex_rules: list[RegexRule] = [], constitution: str = "", constitution_enabled: bool = False)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_guardrails_base.py
from crucible.guardrails.base import GuardrailAction, GuardrailResult, RegexRule, GuardrailConfig

def test_action_and_result():
    a = GuardrailAction(layer="regex_filter", stage="input", action="redact", detail="ssn")
    r = GuardrailResult(text="x", blocked=False, actions=[a])
    assert r.actions[0].action == "redact"

def test_regex_rule_defaults_to_both_stages():
    assert RegexRule(pattern="x", mode="block", label="x").stages == ["input", "output"]

def test_config_defaults():
    c = GuardrailConfig()
    assert c.enabled and c.preset_id == "balanced" and c.constitution_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_guardrails_base.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/guardrails/base.py
from typing import Literal
from pydantic import BaseModel, Field

Stage = Literal["input", "output"]

class GuardrailAction(BaseModel):
    layer: str
    stage: Stage
    action: Literal["inject", "block", "redact", "revise", "pass"]
    detail: str

class GuardrailResult(BaseModel):
    text: str
    blocked: bool
    actions: list[GuardrailAction]

class RegexRule(BaseModel):
    pattern: str
    mode: Literal["block", "redact"]
    label: str
    stages: list[Stage] = Field(default_factory=lambda: ["input", "output"])

class GuardrailConfig(BaseModel):
    enabled: bool = True
    preset_id: str = "balanced"
    regex_rules: list[RegexRule] = Field(default_factory=list)
    constitution: str = ""
    constitution_enabled: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_guardrails_base.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/guardrails/base.py backend/tests/test_guardrails_base.py
git commit -m "feat: add guardrail base types"
```

---

### Task 2: System-prompt presets

**Files:**
- Create: `backend/crucible/guardrails/presets.py`
- Test: `backend/tests/test_guardrails_presets.py`

**Interfaces:**
- Produces:
  - `SystemPromptPreset(id: str, name: str, intensity: int, system_prompt: str)` (pydantic; `intensity` 0=unrestricted … 100=strict).
  - `PRESETS: list[SystemPromptPreset]` with ids `unrestricted`, `balanced`, `strict`.
  - `get_preset(id: str) -> SystemPromptPreset` (raises `KeyError`).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_guardrails_presets.py
import pytest
from crucible.guardrails.presets import PRESETS, get_preset

def test_three_presets_ordered_by_intensity():
    ids = [p.id for p in PRESETS]
    assert ids == ["unrestricted", "balanced", "strict"]
    assert get_preset("unrestricted").intensity == 0
    assert get_preset("strict").intensity == 100

def test_unrestricted_prompt_is_empty_or_minimal():
    assert len(get_preset("unrestricted").system_prompt) <= len(get_preset("strict").system_prompt)

def test_missing_preset_raises():
    with pytest.raises(KeyError):
        get_preset("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_guardrails_presets.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/guardrails/presets.py
from pydantic import BaseModel

class SystemPromptPreset(BaseModel):
    id: str
    name: str
    intensity: int
    system_prompt: str

PRESETS: list[SystemPromptPreset] = [
    SystemPromptPreset(id="unrestricted", name="Unrestricted", intensity=0,
                       system_prompt=""),
    SystemPromptPreset(id="balanced", name="Balanced", intensity=50,
                       system_prompt="You are a helpful, candid assistant. Answer directly. "
                                     "Decline only clearly illegal requests."),
    SystemPromptPreset(id="strict", name="Strict", intensity=100,
                       system_prompt="You are a careful assistant. Refuse harmful, dangerous, "
                                     "or unethical requests and explain why."),
]

_BY_ID = {p.id: p for p in PRESETS}

def get_preset(id: str) -> SystemPromptPreset:
    return _BY_ID[id]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_guardrails_presets.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/guardrails/presets.py backend/tests/test_guardrails_presets.py
git commit -m "feat: add system-prompt presets (unrestricted/balanced/strict)"
```

---

### Task 3: Regex filter

**Files:**
- Create: `backend/crucible/guardrails/filters.py`
- Test: `backend/tests/test_guardrails_filters.py`

**Interfaces:**
- Consumes: `RegexRule`, `GuardrailAction`, `Stage`.
- Produces `RegexFilter(rules: list[RegexRule])` with `.apply(text: str, stage: Stage) -> tuple[str, bool, list[GuardrailAction]]` returning `(text', blocked, actions)`. `block` rule that matches → `blocked=True` (text unchanged, action `block`). `redact` rule → replaces matches with `[REDACTED:label]` (action `redact`). Rules only act on their listed `stages`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_guardrails_filters.py
from crucible.guardrails.base import RegexRule
from crucible.guardrails.filters import RegexFilter

def test_redact_replaces_match():
    f = RegexFilter([RegexRule(pattern=r"\d{3}-\d{2}-\d{4}", mode="redact", label="ssn")])
    text, blocked, actions = f.apply("my ssn is 123-45-6789", "output")
    assert "[REDACTED:ssn]" in text and blocked is False
    assert actions[0].action == "redact"

def test_block_sets_blocked():
    f = RegexFilter([RegexRule(pattern="bakudan", mode="block", label="weapons")])
    text, blocked, actions = f.apply("how to build a bakudan", "input")
    assert blocked is True and actions[0].action == "block"

def test_rule_skipped_for_other_stage():
    f = RegexFilter([RegexRule(pattern="x", mode="block", label="x", stages=["output"])])
    _, blocked, actions = f.apply("x", "input")
    assert blocked is False and actions == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_guardrails_filters.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/guardrails/filters.py
import re
from crucible.guardrails.base import GuardrailAction, RegexRule, Stage

class RegexFilter:
    def __init__(self, rules: list[RegexRule]):
        self.rules = rules

    def apply(self, text: str, stage: Stage) -> tuple[str, bool, list[GuardrailAction]]:
        blocked = False
        actions: list[GuardrailAction] = []
        for rule in self.rules:
            if stage not in rule.stages:
                continue
            rx = re.compile(rule.pattern, re.IGNORECASE)
            if not rx.search(text):
                continue
            if rule.mode == "block":
                blocked = True
                actions.append(GuardrailAction(layer="regex_filter", stage=stage,
                                                action="block", detail=rule.label))
            else:
                text = rx.sub(f"[REDACTED:{rule.label}]", text)
                actions.append(GuardrailAction(layer="regex_filter", stage=stage,
                                                action="redact", detail=rule.label))
        return text, blocked, actions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_guardrails_filters.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/guardrails/filters.py backend/tests/test_guardrails_filters.py
git commit -m "feat: add regex block/redact guardrail filter"
```

---

### Task 4: Constitutional critic

**Files:**
- Create: `backend/crucible/guardrails/constitution.py`
- Test: `backend/tests/test_guardrails_constitution.py`

**Interfaces:**
- Produces:
  - `Critic = Callable[[str, str], str]` — `(constitution, draft) -> revised`.
  - `ConstitutionalCritic(constitution: str, critic: Critic)` with `.revise(draft: str) -> tuple[str, GuardrailAction | None]` — returns the revised text and a `revise` action if it changed, else `(draft, None)`.
  - `build_critic_prompt(constitution: str, draft: str) -> str` (the prompt a real model would receive; pure/testable).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_guardrails_constitution.py
from crucible.guardrails.constitution import ConstitutionalCritic, build_critic_prompt

def test_revise_changes_text():
    critic = lambda constitution, draft: draft.replace("damn", "darn")
    revised, action = ConstitutionalCritic("be polite", critic).revise("damn it")
    assert revised == "darn it"
    assert action is not None and action.action == "revise"

def test_revise_noop_returns_none():
    critic = lambda constitution, draft: draft
    revised, action = ConstitutionalCritic("be polite", critic).revise("hello")
    assert revised == "hello" and action is None

def test_prompt_includes_constitution_and_draft():
    p = build_critic_prompt("RULE: be kind", "you are dumb")
    assert "RULE: be kind" in p and "you are dumb" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_guardrails_constitution.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/guardrails/constitution.py
from typing import Callable
from crucible.guardrails.base import GuardrailAction

Critic = Callable[[str, str], str]

def build_critic_prompt(constitution: str, draft: str) -> str:
    return (
        "Revise the DRAFT so it complies with the CONSTITUTION. "
        "Return only the revised text, nothing else.\n\n"
        f"CONSTITUTION:\n{constitution}\n\nDRAFT:\n{draft}"
    )

class ConstitutionalCritic:
    def __init__(self, constitution: str, critic: Critic):
        self.constitution = constitution
        self.critic = critic

    def revise(self, draft: str) -> tuple[str, GuardrailAction | None]:
        revised = self.critic(self.constitution, draft)
        if revised == draft:
            return draft, None
        return revised, GuardrailAction(layer="constitution", stage="output",
                                        action="revise", detail="revised per constitution")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_guardrails_constitution.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/guardrails/constitution.py backend/tests/test_guardrails_constitution.py
git commit -m "feat: add constitutional self-critique layer"
```

---

### Task 5: Guardrails engine

**Files:**
- Create: `backend/crucible/guardrails/engine.py`
- Create: `backend/crucible/guardrails/__init__.py`
- Test: `backend/tests/test_guardrails_engine.py`

**Interfaces:**
- Consumes: presets, `RegexFilter`, `ConstitutionalCritic`, `GuardrailConfig`, `GuardrailResult`, `Critic`.
- Produces:
  - `GuardrailsEngine(critic: Critic | None = None)`.
  - `.system_prompt(config) -> str` — the preset's system prompt (empty when disabled).
  - `.apply(stage, text, config) -> GuardrailResult` — when `config.enabled` is False, returns passthrough with a single `pass` action. Input stage: regex filter only. Output stage: regex filter, then constitution revise (if `constitution_enabled` and a critic exists).
  - `__init__.py` re-exports `GuardrailsEngine`, `GuardrailConfig`, `RegexRule`, `PRESETS`, `get_preset`, and `build_engine(critic=None)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_guardrails_engine.py
from crucible.guardrails import GuardrailsEngine, GuardrailConfig, RegexRule

def test_disabled_is_passthrough():
    res = GuardrailsEngine().apply("input", "anything", GuardrailConfig(enabled=False))
    assert res.text == "anything" and res.actions[0].action == "pass"

def test_system_prompt_from_preset():
    eng = GuardrailsEngine()
    assert eng.system_prompt(GuardrailConfig(preset_id="strict")) != ""
    assert eng.system_prompt(GuardrailConfig(preset_id="unrestricted")) == ""

def test_input_regex_block():
    cfg = GuardrailConfig(regex_rules=[RegexRule(pattern="bakudan", mode="block", label="w")])
    res = GuardrailsEngine().apply("input", "make a bakudan", cfg)
    assert res.blocked is True

def test_output_constitution_revises():
    critic = lambda c, d: d.replace("stupid", "unwise")
    cfg = GuardrailConfig(constitution="be kind", constitution_enabled=True)
    res = GuardrailsEngine(critic=critic).apply("output", "that is stupid", cfg)
    assert res.text == "that is unwise"
    assert any(a.action == "revise" for a in res.actions)

def test_output_constitution_skipped_when_disabled():
    critic = lambda c, d: "SHOULD NOT RUN"
    cfg = GuardrailConfig(constitution="x", constitution_enabled=False)
    res = GuardrailsEngine(critic=critic).apply("output", "draft", cfg)
    assert res.text == "draft"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_guardrails_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: crucible.guardrails`.

- [ ] **Step 3: Write the engine + package init**

```python
# backend/crucible/guardrails/engine.py
from crucible.guardrails.base import GuardrailAction, GuardrailConfig, GuardrailResult, Stage
from crucible.guardrails.constitution import ConstitutionalCritic, Critic
from crucible.guardrails.filters import RegexFilter
from crucible.guardrails.presets import get_preset

class GuardrailsEngine:
    def __init__(self, critic: Critic | None = None):
        self.critic = critic

    def system_prompt(self, config: GuardrailConfig) -> str:
        if not config.enabled:
            return ""
        return get_preset(config.preset_id).system_prompt

    def apply(self, stage: Stage, text: str, config: GuardrailConfig) -> GuardrailResult:
        if not config.enabled:
            return GuardrailResult(text=text, blocked=False, actions=[
                GuardrailAction(layer="engine", stage=stage, action="pass", detail="guardrails disabled")])

        actions: list[GuardrailAction] = []
        text, blocked, regex_actions = RegexFilter(config.regex_rules).apply(text, stage)
        actions.extend(regex_actions)
        if blocked:
            return GuardrailResult(text=text, blocked=True, actions=actions)

        if stage == "output" and config.constitution_enabled and self.critic is not None:
            text, action = ConstitutionalCritic(config.constitution, self.critic).revise(text)
            if action is not None:
                actions.append(action)

        if not actions:
            actions.append(GuardrailAction(layer="engine", stage=stage, action="pass", detail="no rule matched"))
        return GuardrailResult(text=text, blocked=False, actions=actions)
```

```python
# backend/crucible/guardrails/__init__.py
from crucible.guardrails.base import (  # noqa: F401
    GuardrailAction, GuardrailConfig, GuardrailResult, RegexRule, Stage)
from crucible.guardrails.constitution import Critic  # noqa: F401
from crucible.guardrails.engine import GuardrailsEngine  # noqa: F401
from crucible.guardrails.presets import PRESETS, SystemPromptPreset, get_preset  # noqa: F401

def build_engine(critic: Critic | None = None) -> GuardrailsEngine:
    return GuardrailsEngine(critic=critic)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_guardrails_engine.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/guardrails/engine.py backend/crucible/guardrails/__init__.py backend/tests/test_guardrails_engine.py
git commit -m "feat: add guardrails engine orchestrating presets, filters, constitution"
```

---

### Task 6: Config store

**Files:**
- Create: `backend/crucible/guardrails/store.py`
- Test: `backend/tests/test_guardrails_store.py`

**Interfaces:**
- Produces `GuardrailStore(path: Path)` with `.load() -> GuardrailConfig` (returns defaults if file missing) and `.save(config: GuardrailConfig) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_guardrails_store.py
from crucible.guardrails.base import GuardrailConfig
from crucible.guardrails.store import GuardrailStore

def test_load_defaults_when_missing(tmp_path):
    assert GuardrailStore(tmp_path / "g.json").load().preset_id == "balanced"

def test_save_then_load(tmp_path):
    store = GuardrailStore(tmp_path / "g.json")
    store.save(GuardrailConfig(preset_id="strict", constitution_enabled=True))
    loaded = store.load()
    assert loaded.preset_id == "strict" and loaded.constitution_enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_guardrails_store.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/guardrails/store.py
from pathlib import Path
from crucible.guardrails.base import GuardrailConfig

class GuardrailStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> GuardrailConfig:
        if not self.path.exists():
            return GuardrailConfig()
        return GuardrailConfig.model_validate_json(self.path.read_text())

    def save(self, config: GuardrailConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(config.model_dump_json(indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_guardrails_store.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/guardrails/store.py backend/tests/test_guardrails_store.py
git commit -m "feat: add guardrails config store"
```

---

### Task 7: REST endpoints + agent integration

**Files:**
- Modify: `backend/crucible/app.py`
- Test: `backend/tests/test_guardrails_endpoint.py`

**Interfaces:**
- Consumes: `GuardrailsEngine`, `GuardrailStore`, `GuardrailConfig`, presets, Phase-2 agent endpoint.
- Produces (added to `create_app`, accepting optional `guardrails: GuardrailsEngine | None = None`):
  - `GET /api/guardrails/presets` → `[SystemPromptPreset, ...]`.
  - `GET /api/guardrails/config` → `GuardrailConfig` (from store).
  - `PUT /api/guardrails/config` (body=GuardrailConfig) → saved `GuardrailConfig`.
  - `POST /api/guardrails/apply` body `{stage, text, config}` → `GuardrailResult` (preview).
  - Agent integration: `POST /api/agent/run` accepts optional `guardrails: GuardrailConfig`; before running, the engine's system prompt is prepended and the last user message is input-filtered — if blocked, the stream emits a single `error` event (`reason="blocked by guardrails"`) and stops.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_guardrails_endpoint.py
import json
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry

def mkapp(tmp_path, model=None):
    return create_app(registry=Registry(tmp_path / "r.json"), agent_root=tmp_path, model=model)

def test_presets_endpoint(tmp_path):
    c = TestClient(mkapp(tmp_path))
    ids = [p["id"] for p in c.get("/api/guardrails/presets").json()]
    assert ids == ["unrestricted", "balanced", "strict"]

def test_config_roundtrip(tmp_path):
    c = TestClient(mkapp(tmp_path))
    assert c.get("/api/guardrails/config").json()["preset_id"] == "balanced"
    c.put("/api/guardrails/config", json={"preset_id": "strict"})
    assert c.get("/api/guardrails/config").json()["preset_id"] == "strict"

def test_apply_preview_redacts(tmp_path):
    c = TestClient(mkapp(tmp_path))
    body = {"stage": "output", "text": "ssn 123-45-6789",
            "config": {"regex_rules": [{"pattern": r"\d{3}-\d{2}-\d{4}", "mode": "redact", "label": "ssn"}]}}
    res = c.post("/api/guardrails/apply", json=body).json()
    assert "[REDACTED:ssn]" in res["text"]

def test_agent_blocked_by_guardrails(tmp_path):
    model = lambda m, t: {"role": "assistant", "content": "should not run", "tool_calls": []}
    c = TestClient(mkapp(tmp_path, model=model))
    body = {"messages": [{"role": "user", "content": "make a bakudan"}],
            "permissions": {"default": "allow", "modes": {}},
            "guardrails": {"regex_rules": [{"pattern": "bakudan", "mode": "block", "label": "w"}]}}
    with c.stream("POST", "/api/agent/run", json=body) as r:
        payloads = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert payloads[-1]["type"] == "error"
    assert "guardrail" in payloads[-1]["data"]["reason"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_guardrails_endpoint.py -v`
Expected: FAIL — endpoints/param missing.

- [ ] **Step 3: Extend `app.py`**

Add imports at top:

```python
from crucible.guardrails import GuardrailConfig, GuardrailsEngine, get_preset
from crucible.guardrails.base import GuardrailResult, SystemPromptPreset as _PresetAlias  # see note
from crucible.guardrails.presets import PRESETS, SystemPromptPreset
from crucible.guardrails.store import GuardrailStore
```

> Note: `SystemPromptPreset` lives in `presets.py`; import it from there. Remove the erroneous alias line — it is shown only to flag that the symbol comes from `presets`, not `base`. The correct single import is `from crucible.guardrails.presets import PRESETS, SystemPromptPreset`.

Change the factory signature and add models + routes:

```python
def create_app(registry: Registry | None = None, agent_root: Path | None = None,
               model=None, guardrails: GuardrailsEngine | None = None) -> FastAPI:
    settings = get_settings()
    reg = registry or Registry(settings.registry_path)
    root = Path(agent_root or ".")
    gr_engine = guardrails or GuardrailsEngine()
    gr_store = GuardrailStore(settings.data_dir / "guardrails.json")
    app = FastAPI(title="Crucible")
    # ... existing health/models/lineage routes unchanged ...
```

Add request models near the top of the module:

```python
class ApplyRequest(BaseModel):
    stage: str
    text: str
    config: GuardrailConfig = GuardrailConfig()
```

Extend `AgentRunRequest`:

```python
class AgentRunRequest(BaseModel):
    messages: list[dict]
    permissions: PermissionConfig = PermissionConfig()
    guardrails: GuardrailConfig | None = None
```

Add the guardrail routes:

```python
    @app.get("/api/guardrails/presets")
    def guardrail_presets() -> list[SystemPromptPreset]:
        return PRESETS

    @app.get("/api/guardrails/config")
    def guardrail_config_get() -> GuardrailConfig:
        return gr_store.load()

    @app.put("/api/guardrails/config")
    def guardrail_config_put(config: GuardrailConfig) -> GuardrailConfig:
        gr_store.save(config)
        return config

    @app.post("/api/guardrails/apply")
    def guardrail_apply(req: ApplyRequest) -> GuardrailResult:
        return gr_engine.apply(req.stage, req.text, req.config)
```

Update the agent route to apply input guardrails:

```python
    @app.post("/api/agent/run")
    def agent_run(req: AgentRunRequest):
        if model is None:
            raise HTTPException(status_code=503, detail="no model configured")
        messages = list(req.messages)
        cfg = req.guardrails
        if cfg is not None:
            sys_prompt = gr_engine.system_prompt(cfg)
            if sys_prompt:
                messages = [{"role": "system", "content": sys_prompt}, *messages]
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user is not None:
                checked = gr_engine.apply("input", last_user.get("content", ""), cfg)
                if checked.blocked:
                    def blocked_stream():
                        yield ("data: " + json.dumps(
                            {"type": "error", "data": {"reason": "blocked by guardrails"}}) + "\n\n")
                    return StreamingResponse(blocked_stream(), media_type="text/event-stream")

        policy = PermissionPolicy(default=req.permissions.default, modes=req.permissions.modes)
        agent = Agent(model=model, tools=default_registry(root),
                      permissions=policy, audit=AuditLog(settings.data_dir / "audit.jsonl"))

        def stream():
            for event in agent.run(messages):
                yield f"data: {json.dumps({'type': event.type, 'data': event.data})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_guardrails_endpoint.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all Phase 1–3 tests pass (no regressions in the Phase 2 agent endpoint).

- [ ] **Step 6: Commit**

```bash
git add backend/crucible/app.py backend/tests/test_guardrails_endpoint.py
git commit -m "feat: add guardrails REST endpoints + agent input-guardrail integration"
```

---

## Self-Review

**Spec coverage (Phase 3 scope = Component 4.4 Guardrails Engine):**
- System-prompt presets (safe↔unrestricted, editable) → Task 2 + engine `system_prompt` (Task 5). ✅
- Input/output filters (regex) → Task 3, orchestrated per stage in Task 5. ✅
- Constitutional self-critique (editable constitution, revise) → Task 4, wired in Task 5. ✅
- Per-conversation on/off + intensity → `GuardrailConfig.enabled` + preset `intensity`; per-request `guardrails` on the agent route (Task 7). ✅
- "Show exactly what each layer did" → every layer emits `GuardrailAction`s surfaced by `apply` and the preview endpoint (Tasks 3–5,7). ✅
- Persistence + REST for the GUI → Tasks 6–7. ✅
- Classifier-based filter (spec mentions "optional classifier") → deferred; regex covers the v1 filter surface (flagged, not lost).

**Placeholder scan:** The Task-7 import note intentionally flags one wrong line and gives the correct import — implementer uses `from crucible.guardrails.presets import PRESETS, SystemPromptPreset`. No other placeholders. ✅

**Type consistency:** `GuardrailConfig`, `GuardrailResult`, `RegexRule`, `Stage`, `GuardrailAction` consistent across all tasks. `GuardrailsEngine(critic=None)`, `.apply(stage, text, config)`, `.system_prompt(config)` consistent across Tasks 5,7. `Critic = (constitution, draft) -> revised` consistent Tasks 4,5. ✅
