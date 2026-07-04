# Crucible — Task Backlog

Deferred / planned work, captured so nothing is lost.

## In progress
- [x] Git-like edit version control (history / revert / branch) with per-tensor delta snapshots
- [x] Clone-before-edit (pristine backup + active copy)
- [x] GUI changelog view (history timeline, revert/clone buttons)

## Understandable visualization (4 of 4 DONE)
- [x] Live activation heatmap (token × layer)
- [x] Plain-language feature cards (auto-named mechanism + trigger/output/location)
- [x] Causal flow graph (input concepts → carrier components → output tokens)
- [x] Before/after behavior diff (multi-probe results assessment)

## Modification styles
- [x] Single-layer / multi-layer banded ablation
- [x] Reversible runtime ablation (forward hooks, nondestructive)
- [x] In-place weight editing (no copy, model stays live)
- [x] Feature insertion mechanism (additive runtime steering)
- [x] Insertion auto-tuner (coherence-guarded search; FOUND a clean additive window — corrects earlier "no window")
- [x] Restoration via remove-the-suppressor (target-prompt suppressor direction, proven removal)

## crucible CLI — full Claude-Code parity
- [x] Functional CLI (agent loop, remote-aware --endpoint, slash commands)
- [x] Persistent sessions
- [x] Settings file (~/.crucible/settings.json)
- [x] MCP server support (stdio client, tool wrapping, /mcp command)
- [x] Full slash-command set + status line + config

## Remote orchestration
- [x] Remote-aware endpoints (--endpoint / CRUCIBLE_ENDPOINT)
- [~] Remote orchestration plumbing done (CORS + configurable GUI/CLI node URL); live 1.5TB run runs on a high-RAM inference node

## BYO-AI — bring your own backend
- [x] Service auto-detection (Crucible/Ollama/llama.cpp/vLLM/ComfyUI) on localhost + named remotes
- [x] Capability badges + per-service notes (full / chat-only / no-chat) in the Models tab
- [x] Drive the forge in "chat (direct)" mode — browser → service /v1, works on the static page
- [x] Drive the forge in "+ tools (via Crucible)" mode — register endpoint, full agent tool-loop over it
- [x] POST /api/models/connect — register an OpenAI-compatible endpoint as a first-class model
- [x] Token-level streaming on the tool-loop path (assistant_delta SSE; fragmented tool-calls reassembled)
- [x] Token-level streaming on the direct-chat path (browser → service SSE, static-page friendly)
- [x] Stop/cancel an in-flight run (AbortController through runAgent + chatDirectStream; Stop button)
- [x] Per-service model picker on BYO cards (when a service exposes several; persisted, used by both modes)
- [x] Demo-mode simulated token streaming (static page shows the animation + caret with no backend)
- [x] CORS / OLLAMA_ORIGINS + write-permission docs

## Benchmarks
- [x] lm-eval integration (canonical suite, capability retention)
- [x] loglikelihood MC tasks via HF backend (MMLU/ARC/HellaSwag)


## Full AI dev & analysis pipeline (all DONE)
- [x] Causal interpretability: activation patching / causal trace (proves WHERE, not just correlates)
- [x] Multiple refusal directions (refusal isn't rank-1) + CAA/RepE concept steering
- [x] Sparse autoencoders — monosemantic feature dictionaries (+ token labels)
- [x] Tuned lens — faithful per-layer decodability curve
- [x] Plain-language surgical diagnosis (where/how-we-know/target/repair/risk; translatable)
- [x] Piecemeal alignment: decompose into components → pick → preview remove/add
- [x] Un-alignment AND re-alignment — in-place OR portable LoRA (exact inverse; reversible)
- [x] Direct GGUF abliteration (edit the quantized model in place; F16/BF16/F32/Q8_0)
- [x] Quantization analysis (per-tensor fidelity/compression for a target type)
- [x] Retraining pipeline — real gradient LoRA SFT (peft), saved + auto-registered as a variant
- [x] Eval rigor: safety suites (XSTest/HarmBench/AdvBench/StrongREJECT), LLM-judge, refusal
      classifier, pass@k, contamination

## Serving, models, ops (all DONE)
- [x] Runtime manager: load/stop, multi-active round-robin, tok/s speed test; llama.cpp OR vLLM
- [x] Online/offline autodetection; health-checked routing; GGUF detected by magic bytes
- [x] Import from Ollama (grab the raw GGUF blobs → editable/retrainable/servable)
- [x] Crucible as an OpenAI-compatible provider/gateway (chosen/preferred/nearest routing)
- [x] Tool-calling for EVERY backing model (native relay + ReAct bridge + name coercion)
- [x] Interactive 'ask' tool approval; server-side cancellation; live + pre-flight tok/s
- [x] Crucible MCP server — drive/evolve the whole system from any agent

## Harness (all DONE)
- [x] Full skillset: read/write/edit/multi_edit/list_dir/grep/glob/bash/web_fetch/web_search/todo_write
- [x] Hybrid tool loop (native + text ReAct) so tools work with any model, no toggle
