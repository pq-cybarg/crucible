# Crucible — Design Spec

**Date:** 2026-06-19
**Status:** Approved (core) — pending final spec review
**Working name:** Crucible *(rename freely)*

A cross-platform workbench for running, uncensoring, instrumenting, and benchmarking
open-weight GLM models locally, wrapped in a Claude-Code-style agentic harness.

---

## 1. Goals & Non-Goals

### Goals
- Run an open-weight GLM model locally via an OpenAI-compatible endpoint.
- Abliterate ("uncensor") an open-weight, MIT-licensed model from a GUI, **reversibly and measurably**.
- Manage guardrails (system prompts, filters, constitutional self-critique) from a GUI — dial up or down.
- Explore and manage model weights/variants with real interpretability tooling.
- Provide a Claude-Code-style agent harness (tool use, planning, permissions, context mgmt) — "better than OpenCode."
- Benchmark the local model against **published** GLM-5.2 / Claude Opus numbers **and** a live head-to-head with the assistant in-session.
- Integrate with the user's existing OpenCode install by sharing one backing model endpoint.

### Non-Goals
- Running full-size GLM-5.2 on the 32 GB Mac (physically impossible; ~170–390 GB RAM needed).
- Operating the uncensored model to produce specific harmful content. Crucible is tooling; the user is the operator.
- Using remote inference APIs. Per the user: **local only** (the in-session assistant is the sole permitted "remote," and only as a benchmark counterpart).

---

## 2. Hardware Reality (the constraint that shapes everything)

| Node class | CPU | RAM | GPU | Role |
|---|---|---|---|---|
| Laptop | modern 8–16 core | 32 GB | integrated / small dGPU | Control plane + dev model (GLM-4-32B) + fast local uncensored |
| High-RAM workstation | multi-channel DDR4/5 | 128–256 GB | mid-range (10–12 GB) | Heavy inference node (GLM-5.2) |
| Server | 8-channel server CPU | 256–512 GB | 24–32 GB | Production inference node |

**GLM-5.2 = 743B total / 39B active MoE (DeepSeek Sparse Attention, MIT license, weights public 2026-06-16).**
A MoE keeps all experts resident, so **RAM ≈ on-disk quant size**:

| Quant | Size | Fits 128 GB | Fits 256 GB | Quality |
|---|---|---|---|---|
| Q4_K_M | ~390 GB | ❌ | ❌ | very good |
| Q2_K | ~210 GB | ❌ | ✅ | degraded-but-real |
| IQ2 / ~2-bit dynamic | ~155–186 GB | ⚠️ light NVMe paging | ✅ | usable |
| IQ1 / 1.58-bit dynamic | ~150–160 GB | ⚠️ ~30 GB paging | ✅ | rough |

- **At 128 GB:** 1.58-bit GLM-5.2, light NVMe paging, ~1–2 tok/s. Functional, low quality.
- **At 256 GB:** Q2_K fully in RAM, ~few tok/s, real quality. **Recommended.**
- A mid-range GPU offloads attention + KV cache + a few dense layers.
- Local context capped at 16–32k (1M-token KV cache is hundreds of GB — out of scope locally).

**Local dev/uncensoring target:** `GLM-4-32B-0414` (dense 32B, MIT, Q4_K_M ≈ 19.7 GB) — the largest GLM
that truly runs on 32 GB RAM and can be abliterated + benchmarked for real. `GLM-4-9B` is the fast/low-disk alternative.

---

## 3. Architecture — Control Plane + Inference Node

Nothing hard-binds to one machine. All model traffic goes over an **OpenAI-compatible HTTP endpoint**.

```
┌─────────────────────────── Control Plane (Mac) ───────────────────────────┐
│  Web GUI (React/Vite)  ── HTTP/SSE ──▶  Backend (FastAPI, Python)          │
│   Agent · Guardrails · Uncensor · Weights · Models · Benchmarks            │
│                                                                            │
│  Backend subsystems:                                                       │
│   • Model Registry        • Agent Harness (tool loop)                      │
│   • Guardrails Engine      • Abliteration/Steering Pipeline (torch)        │
│   • Weight/Interp Explorer • Eval Harness                                  │
│   • Audit Log              • llama-server launcher/supervisor              │
└───────────────┬───────────────────────────────────────────┬──────────────┘
                │ OpenAI-compatible HTTP                       │
        ┌───────▼────────┐                            ┌────────▼─────────┐
        │ llama-server    │                            │ llama-server     │
        │ Mac: GLM-4-32B  │                            │ Win: GLM-5.2     │
        └─────────────────┘                            └──────────────────┘
```

Swapping the backing model = changing a URL + entry in the Model Registry. The abliteration/steering
pipeline runs on whichever node holds the weights (Mac for 32B; Windows/Linux for 5.2).

---

## 4. Components

Each unit has one purpose, a defined interface, and is independently testable.

### 4.1 Inference Layer
- `llama-server` launcher/supervisor: start/stop, health-check, GPU layer config, context size, port.
- Exposes OpenAI-compatible `/v1/chat/completions` (+ streaming).
- **Interface:** `InferenceNode{ name, endpoint, model_path, quant, status }`.

### 4.2 Model Registry
- Tracks every model + variant: name, base, endpoint, quant, lineage (original → abliterated → steered), disk path, model card.
- Originals are immutable; variants are new files. Disk usage surfaced in GUI.
- **Interface:** `registry.list() / get(id) / register(variant) / lineage(id)`.

### 4.3 Agent Harness ("better than OpenCode")
The Claude-Code-style loop, encoding how the assistant actually works:
- **Tools:** `read_file`, `write_file`, `edit_file`, `bash`, `grep`, `glob`, `web_fetch`.
- **Planning:** explicit task list, multi-step execution, self-correction.
- **Permission system:** per-tool allow/ask/deny modes; dangerous-command gating.
- **Context management:** summarization at threshold, file-state tracking, tool-result truncation.
- **Streaming** tool calls + text to the GUI over SSE.
- **Interface:** `harness.run(conversation, tools, permissions) -> stream<event>`.

### 4.4 Guardrails Engine (GUI-managed)
Layered, each independently toggleable with an intensity control:
- **System-prompt presets** (safe ↔ unrestricted), GUI-editable.
- **Input/output filters:** regex + optional small classifier.
- **Constitutional self-critique:** GUI-editable constitution; model critiques+revises its own output against it (Constitutional-AI style).
- Per-conversation on/off; the GUI shows exactly what each layer did to a given turn.
- **Interface:** `guardrails.apply(stage, text, config) -> {text, actions[]}`.

### 4.5 Uncensoring Pipeline (Abliteration + Steering)
- **Abliteration:** compute the refusal direction from harmful/harmless prompt-pair sets → orthogonalize it out of the residual stream → export new GGUF. Adjustable strength; A/B preview vs original; one-click revert.
- **Activation steering:** reversible runtime direction add/subtract (no weight edit) — principled complement to destructive abliteration.
- Every run produces a model card + eval delta (see 4.7).
- **Interface:** `abliterate(model, direction, strength) -> variant`; `steer(request, vectors)`.

### 4.6 Weight / Interpretability Explorer
- GGUF tensor browser: per-layer norms, dtype, quant, shapes.
- **Interpretability:** refusal direction as a first-class object; logit lens; activation patching; per-layer attribution; attention viz.
- **Interface:** `explorer.tensors(model)`, `explorer.logit_lens(prompt)`, `explorer.patch(...)`.

### 4.7 Eval Harness + Comparison Report
- **Capability evals:** HumanEval, GPQA, MMLU, SWE-bench-lite — run on the local model for real.
- **Safety evals:** refusal rate, **over-refusal** (false positives), harmful-compliance — measured **before vs after** uncensoring so the safety/capability delta is explicit (RSP mindset).
- **Comparison report:** measured local scores + **published** GLM-5.2 & Opus numbers (clearly labeled *measured* vs *cited*) + **live head-to-head**: exports a prompt set, the in-session assistant answers, both scored. Calibration (ECE) included.
- **Interface:** `evals.run(suite, model) -> results`; `report.build(...)`.

### 4.8 Audit Log
- Every tool call, model swap, abliteration, and guardrail action logged with timestamp + hash. Inspectable in GUI. Underpins agentic safety + reproducibility.

---

## 5. Tech Stack
- **Backend:** Python 3.11+ / FastAPI. One process hosts all subsystems. Python chosen for the ML ecosystem (`torch`, `transformers`, `gguf`, `numpy`) that abliteration/interp/evals require.
- **Frontend:** React + Vite + TypeScript; HTTP + SSE to backend.
- **Inference:** `llama.cpp` / `llama-server` (installed: build 9700).
- **Model fetch:** `huggingface-cli` (installed).
- **OpenCode hook:** configure OpenCode's OpenAI-compatible provider to point at the same `llama-server` endpoint.
- **Tests:** `pytest` (backend), `vitest` (frontend). TDD per project conventions.

---

## 6. Honest Scope & Ethics
- **Uncensoring** = abliterating a local, open-weight, MIT-licensed model for the user's own research/use. Mainstream and legitimate. Crucible builds the tooling and the reversible/measurable controls; the assistant does not author specific harmful content through it.
- **"Test myself against it"** = real measured local scores + published frontier numbers + a genuine live-vs-assistant run. No fabricated self-benchmarks; the live-vs-assistant path is the honest realization of that request.
- The Anthropic-flavored features (safety/capability evals, reversible steering, constitutional guardrails, interpretability, agentic safety, model cards, calibration) exist to make uncensoring **instrumented and reversible** rather than blind.

---

## 7. Build Order (each phase ships something usable)

1. **Foundation** — Model Registry + `llama-server` launcher + GLM-4-32B-0414 Q4 download + verified OpenAI-compatible endpoint.
2. **Agent harness + GUI shell** — tool loop, streaming chat, permission system, tabbed web app. Usable coding agent.
3. **Guardrails Engine** — presets + filters + constitutional self-critique + per-turn visibility.
4. **Uncensoring pipeline** — abliteration + activation steering + A/B + variant management + model cards.
5. **Weight / Interpretability Explorer.**
6. **Eval Harness + comparison report** (capability + safety + live-vs-assistant + calibration).
7. **OpenCode integration + point at a remote GLM-5.2 node** (1.58-bit at 128 GB; Q2_K at 256 GB).

---

## 8. Open Questions / Future
- Final product name.
- Inference node: 1.58-bit at 128 GB as a bridge to Q2_K at 256 GB.
- Server-class node (8-channel, 256–512 GB, 24–32 GB GPU) for full-quality GLM-5.2.

- Whether to add a capability-threshold ("RSP-style") warning when an uncensored variant crosses defined eval scores.
