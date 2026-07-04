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

Optional extras: `pip install -e ".[torch]"` for live weight editing/diagnosis on real HF
weights, or `".[train]"` to also get the **retraining** pipeline (LoRA SFT via `peft`). The
numpy control plane, agent, guardrails, evals, registry, and GUI need none of these.

Run the test suite:

```bash
source .venv/bin/activate && pytest -q          # 382 backend tests
cd frontend && npm run build                     # hardened TypeScript, zero errors
```

---

## What's inside

| Surface | What it does |
|---|---|
| **Agent** | Claude-Code-style tool loop with a full skillset — **read · write · edit · multi_edit · list_dir · grep · glob · bash · web_fetch · todo_write** — allow/ask/deny permissions, audit log, token-streamed SSE with a **Stop** button. Works with **any** model (native tool-calls or text ReAct, auto). |
| **Guardrails** | System-prompt presets + regex/redaction filters + constitutional self-critique — **full editorial CRUD** over every rail, built-ins included; live test bench |
| **Uncensor** | Censorship **diagnosis** in **plain language** (where it's decided, proven by causal **activation patching**, what to remove, how safe) → **abliteration / un-alignment / re-alignment** (in-place *or* portable LoRA) → reversible steering. Plus **SAEs** (monosemantic features), **tuned lens**, **multiple refusal directions**, **CAA concept steering**, and **piecemeal alignment** (decompose → pick → preview). |
| **Pipeline** | The rest of the dev loop: **quantization** fidelity analysis, **alignment components** (decompose/pick/preview remove-or-add), and **retraining** — real gradient **LoRA SFT** on your `{prompt,response}` data, saved + auto-registered as a variant |
| **Weights** | GGUF tensor browser + **direct GGUF abliteration** (edit the quantized model in place, no HF round-trip) |
| **Benchmarks** | EleutherAI **lm-evaluation-harness** + standardized **safety suites** (XSTest over-refusal, HarmBench/AdvBench/StrongREJECT loaders), **LLM-as-judge**, trained refusal classifier, pass@k, contamination |
| **Models** | Registry with immutable originals + variant lineage; **online/offline autodetection**; **runtime manager** (load/stop, multi-active **round-robin**, tok/s speed test; llama.cpp **or vLLM**); **import from Ollama** (grab the raw GGUF blobs → editable/retrainable) |
| **Provider** | Crucible is itself an **OpenAI-compatible provider** — point OpenCode at it and it routes to your chosen / preferred / nearest-available model, with tool-calling for *every* backing model (native relay or ReAct bridge) |

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
typical 32 GB laptop. The realistic targets, by capability tier:

| Node class | Runs |
|---|---|
| Laptop / 32 GB | GLM-4-9B/32B (GGUF) . small HF models for torch abliteration . the whole control plane + GUI |
| High-RAM workstation (128–256 GB) + GPU | GLM-5.2 at low quant — 1.58-bit with NVMe paging, → Q2_K fully in RAM around 256 GB |
| Server (8-channel, 256–512 GB, 24–32 GB GPU) | GLM-5.2 Q4 at quality |

### Pointing Crucible at a remote GLM-5.2 node

```bash
# on the inference node, once GLM-5.2 weights are down:
llama-server --model glm-5.2-IQ2.gguf --port 8081 --ctx-size 16384 --n-gpu-layers 40 --host 0.0.0.0
```
Then register it in Crucible (Models tab or API) with `endpoint=http://<node-ip>:8081`, and the laptop
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
backend/tests/      382 tests
backend/scripts/    smoke.py . abliterate_hf.py
frontend/src/       App + components (Agent/Guardrails/Uncensor/Weights/Benchmarks/Models/Pipeline)
docs/superpowers/   specs/ + plans/ (design + per-phase implementation plans)
```

## Live demo & wiki
- App (static demo, real sample data): **https://pq-cybarg.github.io/crucible/**
- Wiki: **https://pq-cybarg.github.io/crucible/docs/**

Connect the **node** field (top-right) to a running Crucible backend to go live.

## BYO-AI — bring your own backend

The page (even the static demo) can drive **any AI service you already run**. In the **Models** tab,
hit **scan**: Crucible probes localhost — and any remote you name — for Crucible, **Ollama**
(`:11434`), **llama.cpp** / **vLLM** / any OpenAI-compatible `/v1` (`:8080/:8081/:8000`), and
**ComfyUI** (`:8188`). Detected services show capability badges (and a **model picker** when a service
exposes several); any chat-capable one can be driven from the **forge** console two ways (a **Stop**
button aborts an in-flight run in either mode):

- **chat (direct)** — browser → service `/v1`, a plain chat call (streams token-by-token via the
  service's own SSE). Works from the static page, no Crucible backend required; no tool-loop.
- **+ tools (via Crucible)** — registers the endpoint as a Crucible model (`POST /api/models/connect`)
  and routes the **full agent tool-loop** through your Crucible backend: Crucible executes the tools
  (read/write/edit/grep/bash, with permissions) locally and relays generation to the service. Needs a
  Crucible node online; the service just generates. Replies **stream token-by-token** (SSE
  `assistant_delta` events) — fragmented tool-calls are reassembled server-side.

Two caveats, by design:

- **Chat vs. edit.** Ollama/llama.cpp/vLLM are **chat-only** here — fine for talking to a model.
  To **diagnose / abliterate / edit** weights you need a **Crucible node with write access** to the
  model files: run Crucible locally (it can wrap any of these as its inference endpoint), or point it
  at a remote you can write to. ComfyUI is detected but isn't a chat backend.
- **Browser CORS.** Calling a local service from a `https://…github.io` page is cross-origin. Crucible
  sets permissive CORS itself; **Ollama** needs `OLLAMA_ORIGINS` set (`OLLAMA_ORIGINS=*`, or the page's
  origin) before it will answer a browser. llama.cpp/vLLM generally allow it; for a remote, expose the
  port and (if you set `CRUCIBLE_API_TOKEN`) supply the token in the GUI.

## Crucible as a model provider (for OpenCode)

Crucible exposes an **OpenAI-compatible `/v1`**, so point OpenCode (or any client) straight at
it and Crucible decides which backing model serves each request: the model you named if it's
up, else a **preference order** you set in advance, else the **nearest available** model —
availability tested live at request time. `/v1/models` lists the choices; the response's
`system_fingerprint` says which was used and why.

```jsonc
// ~/.config/opencode/opencode.json
{ "provider": { "crucible": {
    "npm": "@ai-sdk/openai-compatible",
    "options": { "baseURL": "http://127.0.0.1:8400/v1" },
    "models": { "auto": {}, "crucible": {} } } } }
// then: opencode --model crucible/auto   (auto = nearest/ preferred)
```

Set the fallback order via `POST /api/provider/preferences {"preferences": ["glm-5.2","crucible"]}`.

**Tools through the gateway — for every backing model.** When OpenCode drives Crucible-as-provider
it sends its own tool definitions and expects `tool_calls` back. Crucible delivers them two ways:
- **Proxied endpoints** (llama.cpp/vLLM/remote): `tools`/`tool_choice` are forwarded and the
  model's native `tool_calls` relayed. Verified end-to-end — `llama-server` returns proper
  `tool_calls` (even a tiny abliterated GGUF) and Crucible passes them straight through.
- **The local abliterated adapter** (`crucible`): it has no *native* function-calling, so
  Crucible bridges it — describes the tools in a ReAct format, generates, and converts the text
  action into a real OpenAI `tool_call` (`finish_reason: tool_calls`). So an uncensored local
  model gets tool-calling too, without a tool-capable runtime.

Either way the client sees standard OpenAI tool-calling. Crucible's own **forge** additionally
ships the full toolset above and gives tools to any model via its hybrid loop.

## Drive Crucible from an agent (MCP)

Crucible ships an **MCP server** (`crucible-mcp`) that exposes its capabilities as tools, so
Claude Code — or any MCP-speaking agent — can drive the whole system: list models, diagnose
censorship, get the plain-language surgical report, run causal tracing, run safety suites,
chat with a model, and manage the runtime. Point it at a running backend and register it:

```jsonc
// ~/.claude/mcp.json (or your client's MCP config)
{ "mcpServers": {
    "crucible": {
      "command": "crucible-mcp",
      "env": { "CRUCIBLE_ENDPOINT": "http://127.0.0.1:8400", "CRUCIBLE_API_TOKEN": "" }
    } } }
```

Tools: `crucible_list_models`, `crucible_diagnose`, `crucible_explain` (plain-language,
translatable), `crucible_causal_trace`, `crucible_safety_suite`, `crucible_chat`,
`crucible_runtime`. The agent diagnoses, abliterates, benchmarks, and iterates — you stay in
the loop. (Crucible is also an MCP *client*, so the agent it runs can use your other MCP servers.)

## Security
The server runs tools (`bash`, file edits) and serves models, so when you expose it
beyond `127.0.0.1` (Docker `0.0.0.0`, or a remote inference node), **set a token**:

```bash
CRUCIBLE_API_TOKEN=$(openssl rand -hex 24) crucible-serve
```

When set, every `/api` and `/v1` request needs `Authorization: Bearer <token>`
(`/api/health` stays open for probes). The GUI has a token field next to the node URL;
the CLI takes `--token` (or saves it in `~/.crucible/settings.json`). Unset = open
(fine for local-only `127.0.0.1`). License: MIT.

## Install from CI artifacts
- **Docker (any OS/arch, incl. Raspberry Pi):**
  ```bash
  docker pull ghcr.io/pq-cybarg/crucible:latest
  docker run -p 8400:8400 ghcr.io/pq-cybarg/crucible:latest   # → http://localhost:8400
  ```
- **Native binaries / packages / mobile:** download from the latest tagged
  [release](https://github.com/pq-cybarg/crucible/releases). Verify integrity with the
  published **SHA3-256** sums (this project uses SHA-3 — no legacy hashes):
  ```bash
  shasum -a 3-256 -c SHA3-256SUMS.txt    # or: openssl dgst -sha3-256 <file>
  ```
