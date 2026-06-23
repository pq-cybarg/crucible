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
- [x] CORS / OLLAMA_ORIGINS + write-permission docs

## Benchmarks
- [x] lm-eval integration (canonical suite, capability retention)
- [x] loglikelihood MC tasks via HF backend (MMLU/ARC/HellaSwag)
