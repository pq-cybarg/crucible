# Crucible

A local workbench for running, **uncensoring**, **diagnosing**, and **benchmarking** open-weight
LLMs — wrapped in a Claude-Code-style agent harness, with a GUI for everything.

Built around one principle: **you can't responsibly remove guardrails you can't measure.** So Crucible
doesn't just abliterate — it detects the censorship, localizes *where* it lives, explains *why/how* it
works, quantifies *exactly* what a cut removes, and verifies the removal was surgical.

---

## Quick start

```bash
./run.sh                       # llama-server (if a GGUF is present) + backend + frontend
# → frontend  http://localhost:5273
# → backend   http://127.0.0.1:8400

# enable live torch abliteration on real HF weights:
CRUCIBLE_HF_MODEL="$PWD/models/qwen-hf" ./run.sh
```

Run the test suite:

```bash
source .venv/bin/activate && pytest -q          # 123 backend tests
cd frontend && npm run build                     # hardened TypeScript, zero errors
```

---

## What's inside

| Surface | What it does |
|---|---|
| **Agent** | Claude-Code-style tool loop: read/write/edit/grep/glob/bash, allow/ask/deny permissions, audit log, SSE streaming |
| **Guardrails** | System-prompt presets + regex/redaction filters + constitutional self-critique — **full editorial CRUD** over every rail, built-ins included; live test bench |
| **Uncensor** | Censorship **diagnosis** (per-layer refusal localization, per-component impact, why/how/removal, surgical verdict) → abliteration → reversible activation steering → lineage-tracked variants |
| **Weights** | GGUF tensor browser — architecture, layers, shapes, quantization mix — read straight from the header |
| **Benchmarks** | EleutherAI **lm-evaluation-harness** (the real thing) over the canonical suite, with per-metric standard error, vs cited frontier numbers |
| **Models** | Registry with immutable originals + variant lineage |

## Architecture — control plane + inference node

Everything talks to an **OpenAI-compatible endpoint**, so the backing model is just a URL:

```
Web GUI (React, hardened TS) --HTTP/SSE--> FastAPI backend (control plane)
                                              |  registry . agent . guardrails
                                              |  abliteration+diagnosis . evals . weights
                              +---------------+---------------+
                       llama-server (GGUF)              torch adapter (HF safetensors)
                       Mac / Windows / Linux            real abliteration on real weights
```

- **Backend:** Python 3.11+ / FastAPI. Core is numpy-only; torch/transformers are optional (only the live abliteration adapter needs them).
- **Frontend:** React + Vite + TypeScript under `strict` + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`.
- **Inference:** llama.cpp / `llama-server`.
- **OpenCode:** `~/.config/opencode/opencode.json` registers a `crucible-local` provider → `opencode --model crucible-local/local`.

## The censorship-diagnosis pipeline (live, on real weights)

```
POST /api/abliteration/diagnose  -> {best_layer, layer_profile[], components{removed_fraction},
                                     surgical, collateral_risk, why, how, removal}
```

Refusal is a near-linear direction in the residual stream (Arditi et al. 2024). Crucible measures
harmful/harmless **separability per layer** (where it's decided), how much each writing matrix
**projects onto that direction** (`||r.rTW|| / ||W||` — what you'd remove), and confirms abliteration
removes **only that rank-1 component** (everything orthogonal to `r` is untouched -> capabilities
preserved). Verified end-to-end on Qwen2.5-0.5B: refusal margin at the peak layer **13.93 -> 7.03**
after a single surgical pass.

## Hardware reality

GLM-5.2 is a **743B MoE** — it needs ~170-390 GB of RAM at any usable quant. It does **not** run on a
32 GB Mac. The realistic targets:

| Node | Runs |
|---|---|
| Laptop / 32 GB | GLM-4-9B/32B (GGUF) . small HF models for torch abliteration . the whole control plane + GUI |
| High-RAM workstation (128-256 GB) + GPU | GLM-5.2 at 1.58-bit (paging) now -> Q2_K fully in RAM after a 256 GB upgrade |
| Future Linux (8-channel server CPU, 256-512 GB, 24-32 GB GPU) | GLM-5.2 Q4 at quality |

### Pointing Crucible at the Windows GLM-5.2 node

```powershell
# on the inference node, once GLM-5.2 weights are down:
llama-server --model glm-5.2-IQ2.gguf --port 8081 --ctx-size 16384 --n-gpu-layers 40 --host 0.0.0.0
```
Then register it in Crucible (Models tab or API) with `endpoint=http://<windows-ip>:8081`, and the Mac
GUI drives the real 5.2 — agent, benchmarks, and guardrails all over the network.

## Honest scope

Crucible abliterates **local, open-weight, MIT-licensed** models for the operator's own research/use —
mainstream practice, made instrumented and reversible. The tooling is yours; it does not author harmful
content. Benchmark numbers are labeled **measured** (run locally via lm-eval) vs **cited** (published,
sourced); frontier numbers that can't be reliably sourced are left blank rather than guessed.

## Layout

```
backend/crucible/   registry . inference . agent . tools . permissions . audit
                    guardrails/ . abliteration/ (+torch_adapter) . evals/ (+lmeval) . weights/
backend/tests/      123 tests
backend/scripts/    smoke.py . abliterate_hf.py
frontend/src/       App + components (Agent/Guardrails/Uncensor/Weights/Benchmarks/Models)
docs/superpowers/   specs/ + plans/ (design + per-phase implementation plans)
```

## Live demo & wiki
- App (static demo, real sample data): **https://pq-cybarg.github.io/crucible/**
- Wiki: **https://pq-cybarg.github.io/crucible/docs/**

Connect the **node** field (top-right) to a running Crucible backend to go live.

## Security
The server runs tools (`bash`, file edits) and serves models, so when you expose it
beyond `127.0.0.1` (Docker `0.0.0.0`, or the remote Windows node), **set a token**:

```bash
CRUCIBLE_API_TOKEN=$(openssl rand -hex 24) crucible-serve
```

When set, every `/api` and `/v1` request needs `Authorization: Bearer <token>`
(`/api/health` stays open for probes). The GUI has a token field next to the node URL;
the CLI takes `--token` (or saves it in `~/.crucible/settings.json`). Unset = open
(fine for local-only `127.0.0.1`). License: MIT.
