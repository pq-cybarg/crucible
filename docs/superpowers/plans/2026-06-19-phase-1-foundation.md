# Phase 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Crucible backend skeleton — a model registry, a `llama-server` launcher/supervisor, an OpenAI-compatible client, and a FastAPI app — so a downloaded GLM GGUF can be served and queried through a verified local endpoint.

**Architecture:** A single Python/FastAPI process (the control plane) manages model metadata (Registry), launches and health-checks `llama-server` subprocesses (Inference), and talks to them over the OpenAI-compatible HTTP API (Client). The FastAPI app exposes these as REST endpoints the future GUI consumes. No model weights live in the repo; paths point at `models/`.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, httpx, pydantic v2, pytest, pytest-asyncio, respx (HTTP mocking), llama.cpp/`llama-server`.

## Global Constraints

- Python 3.11+ required.
- All model traffic uses the **OpenAI-compatible** API (`/v1/chat/completions`).
- Original model files are **immutable**; variants are always new files (enforced later — registry must never rewrite a registered path).
- **Local-only** inference: endpoints are `127.0.0.1`/LAN URLs; no remote API providers.
- Registry persists to JSON at `<data_dir>/registry.json`; `*.gguf` and `models/` are gitignored.
- TDD: every behavior gets a failing test first. Frequent commits (one per task minimum).
- Dev model: `GLM-4-32B-0414` Q4_K_M at `models/glm-4-32b/THUDM_GLM-4-32B-0414-Q4_K_M.gguf`.

---

## File Structure

```
crucible/
  pyproject.toml                       # package + deps + pytest config
  backend/
    crucible/
      __init__.py
      config.py        # Settings: data_dir, models_dir, default host/port
      registry.py      # Model, ModelVariant, Registry (JSON-backed CRUD + lineage)
      inference.py     # LlamaServer: launch/stop/health of llama-server subprocess
      client.py        # ChatClient: OpenAI-compatible chat() + health()
      app.py           # FastAPI app wiring registry + inference + client
    tests/
      __init__.py
      conftest.py      # tmp data_dir fixture, fake llama-server fixtures
      test_config.py
      test_registry.py
      test_inference.py
      test_client.py
      test_app.py
```

---

### Task 1: Project setup

**Files:**
- Create: `pyproject.toml`
- Create: `backend/crucible/__init__.py`
- Create: `backend/tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable `crucible` package; `pytest` runnable from repo root.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "crucible"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "httpx>=0.27",
    "pydantic>=2.6",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "respx>=0.21"]

[tool.setuptools.packages.find]
where = ["backend"]

[tool.pytest.ini_options]
testpaths = ["backend/tests"]
asyncio_mode = "auto"
pythonpath = ["backend"]
```

- [ ] **Step 2: Create empty package files**

```bash
mkdir -p backend/crucible backend/tests
touch backend/crucible/__init__.py backend/tests/__init__.py
```

- [ ] **Step 3: Create venv and install**

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```
Expected: installs without error.

- [ ] **Step 4: Verify pytest collects nothing yet**

Run: `. .venv/bin/activate && pytest -q`
Expected: "no tests ran" (exit 5) — confirms config is valid.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml backend/crucible/__init__.py backend/tests/__init__.py
git commit -m "chore: scaffold crucible python package"
```

---

### Task 2: Config module

**Files:**
- Create: `backend/crucible/config.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings(data_dir: Path, models_dir: Path, host: str, port: int)` and `get_settings() -> Settings`. `Settings.registry_path -> Path` = `data_dir / "registry.json"`. Honors env vars `CRUCIBLE_DATA_DIR`, `CRUCIBLE_MODELS_DIR`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_config.py
from pathlib import Path
from crucible.config import Settings, get_settings

def test_settings_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(tmp_path / "models"))
    s = get_settings()
    assert s.data_dir == tmp_path / "data"
    assert s.models_dir == tmp_path / "models"
    assert s.registry_path == tmp_path / "data" / "registry.json"
    assert s.host == "127.0.0.1"

def test_data_dir_created(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "d"))
    s = get_settings()
    assert s.data_dir.is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: crucible.config`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/config.py
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    data_dir: Path
    models_dir: Path
    host: str = "127.0.0.1"
    port: int = 8400

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.json"

def get_settings() -> Settings:
    data_dir = Path(os.environ.get("CRUCIBLE_DATA_DIR", Path.home() / ".crucible"))
    models_dir = Path(os.environ.get("CRUCIBLE_MODELS_DIR", "models"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(data_dir=data_dir, models_dir=models_dir)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_config.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/config.py backend/tests/test_config.py
git commit -m "feat: add config/settings module"
```

---

### Task 3: Model Registry

**Files:**
- Create: `backend/crucible/registry.py`
- Test: `backend/tests/test_registry.py`

**Interfaces:**
- Consumes: `crucible.config.Settings`.
- Produces:
  - `Model(id: str, name: str, base_id: str | None, path: str, quant: str, kind: Literal["base","abliterated","steered"], endpoint: str | None, created: str, notes: str)` (pydantic model).
  - `Registry(path: Path)` with: `list() -> list[Model]`, `get(id) -> Model` (raises `KeyError`), `register(model: Model) -> Model` (raises `ValueError` on duplicate id or if a base path is reused by a non-base), `lineage(id) -> list[Model]` (root→...→id), `set_endpoint(id, endpoint) -> Model`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_registry.py
import pytest
from crucible.registry import Model, Registry

def make(id, base=None, kind="base", path=None):
    return Model(id=id, name=id, base_id=base, path=path or f"/m/{id}.gguf",
                 quant="Q4_K_M", kind=kind, endpoint=None, created="2026-06-19", notes="")

def test_register_and_get(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register(make("glm32b"))
    assert reg.get("glm32b").name == "glm32b"
    assert [m.id for m in reg.list()] == ["glm32b"]

def test_duplicate_id_rejected(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register(make("a"))
    with pytest.raises(ValueError):
        reg.register(make("a"))

def test_persistence_across_instances(tmp_path):
    p = tmp_path / "registry.json"
    Registry(p).register(make("a"))
    assert Registry(p).get("a").id == "a"

def test_lineage(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register(make("base"))
    reg.register(make("abl", base="base", kind="abliterated"))
    reg.register(make("steer", base="abl", kind="steered"))
    assert [m.id for m in reg.lineage("steer")] == ["base", "abl", "steer"]

def test_original_path_immutable(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register(make("base", path="/m/base.gguf"))
    with pytest.raises(ValueError):
        reg.register(make("abl", base="base", kind="abliterated", path="/m/base.gguf"))

def test_set_endpoint(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register(make("a"))
    reg.set_endpoint("a", "http://127.0.0.1:8081")
    assert reg.get("a").endpoint == "http://127.0.0.1:8081"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: crucible.registry`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/registry.py
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel

class Model(BaseModel):
    id: str
    name: str
    base_id: str | None
    path: str
    quant: str
    kind: Literal["base", "abliterated", "steered"]
    endpoint: str | None
    created: str
    notes: str = ""

class Registry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._models: dict[str, Model] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._models = {m["id"]: Model(**m) for m in data}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([m.model_dump() for m in self._models.values()], indent=2))

    def list(self) -> list[Model]:
        return list(self._models.values())

    def get(self, id: str) -> Model:
        return self._models[id]

    def register(self, model: Model) -> Model:
        if model.id in self._models:
            raise ValueError(f"duplicate id: {model.id}")
        if model.kind != "base":
            for existing in self._models.values():
                if existing.path == model.path:
                    raise ValueError(f"path reuses an existing model file: {model.path}")
        self._models[model.id] = model
        self._save()
        return model

    def set_endpoint(self, id: str, endpoint: str) -> Model:
        m = self._models[id]
        updated = m.model_copy(update={"endpoint": endpoint})
        self._models[id] = updated
        self._save()
        return updated

    def lineage(self, id: str) -> list[Model]:
        chain: list[Model] = []
        cur: str | None = id
        while cur is not None:
            m = self._models[cur]
            chain.append(m)
            cur = m.base_id
        return list(reversed(chain))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_registry.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/registry.py backend/tests/test_registry.py
git commit -m "feat: add model registry with lineage + immutable originals"
```

---

### Task 4: llama-server launcher/supervisor

**Files:**
- Create: `backend/crucible/inference.py`
- Test: `backend/tests/test_inference.py`
- Create: `backend/tests/conftest.py`

**Interfaces:**
- Consumes: nothing (takes explicit args).
- Produces:
  - `LlamaServer(model_path: str, port: int, ctx: int = 16384, gpu_layers: int = 999, binary: str = "llama-server")`.
  - `.command() -> list[str]` (the argv it will exec — pure, testable without launching).
  - `.start() -> None` (spawns subprocess), `.stop() -> None`, `.is_running -> bool`.
  - `.endpoint -> str` = `http://127.0.0.1:{port}`.
  - `wait_healthy(endpoint: str, timeout: float = 120) -> bool` (polls `GET {endpoint}/health` until 200 or timeout).

- [ ] **Step 1: Write `conftest.py` (fake binary helper)**

```python
# backend/tests/conftest.py
import sys, textwrap, stat
from pathlib import Path
import pytest

@pytest.fixture
def fake_llama_server(tmp_path):
    """A stand-in 'llama-server' that serves /health then sleeps, so start/stop is real but cheap."""
    script = tmp_path / "fake-llama-server"
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import sys, http.server, threading, time
        port = int(sys.argv[sys.argv.index("--port")+1])
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b'{{"status":"ok"}}')
            def log_message(self, *a): pass
        srv = http.server.HTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        while True: time.sleep(0.2)
    """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_inference.py
from crucible.inference import LlamaServer, wait_healthy

def test_command_construction():
    srv = LlamaServer(model_path="/m/x.gguf", port=8081, ctx=8192, gpu_layers=50)
    cmd = srv.command()
    assert cmd[0] == "llama-server"
    assert "--model" in cmd and "/m/x.gguf" in cmd
    assert "--port" in cmd and "8081" in cmd
    assert "--ctx-size" in cmd and "8192" in cmd
    assert "--n-gpu-layers" in cmd and "50" in cmd

def test_endpoint():
    assert LlamaServer("/m/x.gguf", 8081).endpoint == "http://127.0.0.1:8081"

def test_start_stop_and_health(fake_llama_server):
    srv = LlamaServer(model_path="/m/x.gguf", port=8137, binary=fake_llama_server)
    srv.start()
    try:
        assert wait_healthy(srv.endpoint, timeout=10) is True
        assert srv.is_running is True
    finally:
        srv.stop()
    assert srv.is_running is False
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_inference.py -v`
Expected: FAIL — `ModuleNotFoundError: crucible.inference`.

- [ ] **Step 4: Write minimal implementation**

```python
# backend/crucible/inference.py
import subprocess
import time
import httpx

class LlamaServer:
    def __init__(self, model_path: str, port: int, ctx: int = 16384,
                 gpu_layers: int = 999, binary: str = "llama-server"):
        self.model_path = model_path
        self.port = port
        self.ctx = ctx
        self.gpu_layers = gpu_layers
        self.binary = binary
        self._proc: subprocess.Popen | None = None

    def command(self) -> list[str]:
        return [
            self.binary,
            "--model", self.model_path,
            "--port", str(self.port),
            "--ctx-size", str(self.ctx),
            "--n-gpu-layers", str(self.gpu_layers),
            "--host", "127.0.0.1",
        ]

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(self.command())

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

def wait_healthy(endpoint: str, timeout: float = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{endpoint}/health", timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    return False
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest backend/tests/test_inference.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/crucible/inference.py backend/tests/test_inference.py backend/tests/conftest.py
git commit -m "feat: add llama-server launcher with health check"
```

---

### Task 5: OpenAI-compatible chat client

**Files:**
- Create: `backend/crucible/client.py`
- Test: `backend/tests/test_client.py`

**Interfaces:**
- Consumes: an endpoint URL string.
- Produces:
  - `ChatClient(endpoint: str)`.
  - `async chat(messages: list[dict], model: str = "local", temperature: float = 0.7, max_tokens: int = 512) -> str` — POSTs `{endpoint}/v1/chat/completions`, returns `choices[0].message.content`.
  - `async health() -> bool` — `GET {endpoint}/health` == 200.

- [ ] **Step 1: Write the failing test (mock HTTP with respx)**

```python
# backend/tests/test_client.py
import httpx, respx, pytest
from crucible.client import ChatClient

@pytest.mark.asyncio
@respx.mock
async def test_chat_returns_content():
    route = respx.post("http://127.0.0.1:8081/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hello world"}}]
        })
    )
    client = ChatClient("http://127.0.0.1:8081")
    out = await client.chat([{"role": "user", "content": "hi"}])
    assert out == "hello world"
    assert route.called

@pytest.mark.asyncio
@respx.mock
async def test_health():
    respx.get("http://127.0.0.1:8081/health").mock(return_value=httpx.Response(200))
    assert await ChatClient("http://127.0.0.1:8081").health() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_client.py -v`
Expected: FAIL — `ModuleNotFoundError: crucible.client`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/client.py
import httpx

class ChatClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    async def chat(self, messages: list[dict], model: str = "local",
                   temperature: float = 0.7, max_tokens: int = 512) -> str:
        payload = {"model": model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(f"{self.endpoint}/v1/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                return (await c.get(f"{self.endpoint}/health")).status_code == 200
        except httpx.HTTPError:
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_client.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/crucible/client.py backend/tests/test_client.py
git commit -m "feat: add OpenAI-compatible chat client"
```

---

### Task 6: FastAPI app

**Files:**
- Create: `backend/crucible/app.py`
- Test: `backend/tests/test_app.py`

**Interfaces:**
- Consumes: `Registry`, `Model`, `ChatClient`, `get_settings`.
- Produces a FastAPI app with:
  - `GET /api/models` → `[Model, ...]`
  - `POST /api/models` (body = Model JSON) → `Model` (201); 409 on duplicate.
  - `GET /api/models/{id}/lineage` → `[Model, ...]`; 404 if missing.
  - `GET /api/health` → `{"ok": true}`.
  - `create_app(registry: Registry | None = None) -> FastAPI` factory (injectable for tests).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_app.py
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry

def client(tmp_path):
    return TestClient(create_app(Registry(tmp_path / "registry.json")))

def test_health(tmp_path):
    assert client(tmp_path).get("/api/health").json() == {"ok": True}

def test_create_list_model(tmp_path):
    c = client(tmp_path)
    body = {"id": "glm32b", "name": "GLM-4-32B", "base_id": None,
            "path": "/m/glm32b.gguf", "quant": "Q4_K_M", "kind": "base",
            "endpoint": None, "created": "2026-06-19", "notes": ""}
    r = c.post("/api/models", json=body)
    assert r.status_code == 201
    assert c.get("/api/models").json()[0]["id"] == "glm32b"

def test_duplicate_returns_409(tmp_path):
    c = client(tmp_path)
    body = {"id": "a", "name": "a", "base_id": None, "path": "/m/a.gguf",
            "quant": "Q4_K_M", "kind": "base", "endpoint": None,
            "created": "2026-06-19", "notes": ""}
    c.post("/api/models", json=body)
    assert c.post("/api/models", json=body).status_code == 409

def test_lineage_404(tmp_path):
    assert client(tmp_path).get("/api/models/nope/lineage").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: crucible.app`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/crucible/app.py
from fastapi import FastAPI, HTTPException
from crucible.config import get_settings
from crucible.registry import Registry, Model

def create_app(registry: Registry | None = None) -> FastAPI:
    reg = registry or Registry(get_settings().registry_path)
    app = FastAPI(title="Crucible")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/models")
    def list_models() -> list[Model]:
        return reg.list()

    @app.post("/api/models", status_code=201)
    def create_model(model: Model) -> Model:
        try:
            return reg.register(model)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.get("/api/models/{model_id}/lineage")
    def lineage(model_id: str) -> list[Model]:
        try:
            return reg.lineage(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="model not found")

    return app

app = create_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_app.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all tests pass (config + registry + inference + client + app).

- [ ] **Step 6: Commit**

```bash
git add backend/crucible/app.py backend/tests/test_app.py
git commit -m "feat: add FastAPI app exposing registry + health"
```

---

### Task 7: End-to-end smoke test with the real model

**Files:**
- Create: `backend/scripts/smoke.py`

**Interfaces:**
- Consumes: `LlamaServer`, `wait_healthy`, `ChatClient`, `Registry`, `Model`, `get_settings`.
- Produces: a runnable script that launches `llama-server` on the downloaded GGUF, waits healthy, registers it, sends one chat request, prints the reply, and stops the server.

> **Gating:** This task requires the GLM-4-32B Q4 download to be complete
> (`models/glm-4-32b/THUDM_GLM-4-32B-0414-Q4_K_M.gguf` exists, ~19.7 GB).
> It is a manual integration check, not an automated unit test.

- [ ] **Step 1: Write the smoke script**

```python
# backend/scripts/smoke.py
import asyncio
from pathlib import Path
from crucible.config import get_settings
from crucible.registry import Registry, Model
from crucible.inference import LlamaServer, wait_healthy
from crucible.client import ChatClient

MODEL_PATH = "models/glm-4-32b/THUDM_GLM-4-32B-0414-Q4_K_M.gguf"

async def main():
    assert Path(MODEL_PATH).exists(), f"missing model: {MODEL_PATH}"
    srv = LlamaServer(model_path=MODEL_PATH, port=8081, ctx=16384, gpu_layers=999)
    srv.start()
    try:
        assert wait_healthy(srv.endpoint, timeout=180), "server never became healthy"
        reg = Registry(get_settings().registry_path)
        if "glm-4-32b" not in [m.id for m in reg.list()]:
            reg.register(Model(id="glm-4-32b", name="GLM-4-32B-0414", base_id=None,
                               path=MODEL_PATH, quant="Q4_K_M", kind="base",
                               endpoint=srv.endpoint, created="2026-06-19", notes="dev model"))
        reply = await ChatClient(srv.endpoint).chat(
            [{"role": "user", "content": "Reply with exactly: Crucible online."}],
            max_tokens=32, temperature=0.0)
        print("MODEL REPLY:", reply)
        assert reply.strip(), "empty reply"
    finally:
        srv.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run the smoke test (after download completes)**

Run: `. .venv/bin/activate && python backend/scripts/smoke.py`
Expected: prints `MODEL REPLY: Crucible online.` (or close), then exits 0.

- [ ] **Step 3: Commit**

```bash
git add backend/scripts/smoke.py
git commit -m "feat: add end-to-end smoke test for local model serving"
```

---

## Self-Review

**Spec coverage (Phase 1 scope = Component 4.1 Inference Layer + 4.2 Model Registry + verified endpoint):**
- 4.2 Model Registry → Task 3 (CRUD, lineage, immutable originals, set_endpoint). ✅
- 4.1 Inference Layer (llama-server launcher, OpenAI-compatible endpoint) → Tasks 4–5. ✅
- "verified OpenAI-compatible endpoint" → Task 7 smoke test. ✅
- REST surface for future GUI → Task 6. ✅
- Local-only constraint → endpoints bound to 127.0.0.1 (Task 4). ✅
- Immutable originals constraint → `test_original_path_immutable` (Task 3). ✅
- Out of Phase 1 (deferred to later phases): agent harness, guardrails, abliteration, weight explorer, eval harness — not in this plan by design.

**Placeholder scan:** No TBD/TODO; every code step contains complete code. ✅

**Type consistency:** `Model` fields identical across Tasks 3/6/7; `LlamaServer` ctor + `.endpoint` + `wait_healthy` signatures consistent across Tasks 4/7; `ChatClient.chat` signature consistent across Tasks 5/7. ✅
