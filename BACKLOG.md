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


# Roadmap — next phase (from design brainstorms)
Legend: [x] shipped · [~] partial/foundation laid · [ ] open

## Multimodality — I/O across audio, voice, video, image, document
- [x] Provider passthrough of multimodal content (image_url / input_audio) to VLM/audio backing models
- [x] Media endpoints: /v1/audio/transcriptions (STT), /v1/audio/speech (TTS), /v1/images/generations, /v1/embeddings
- [~] Route image/video generation to the already-detected ComfyUI backend (generate_image tool + routes)
- [x] Media tools for the agent + /api/tools (transcribe, speak, generate_image, describe_image)
- [ ] Document I/O (parse PDFs/docs in, render documents out)
- [ ] Mixed / asymmetric / changing input-output structures (per-request modality shapes)

## Multi-model systems / subsystems (model graphs)
- [x] Model-graph engine: compose routed model calls into pipelines/DAGs (e.g. STT -> LLM -> TTS; image -> VLM -> text -> image)
- [x] Cascades (cheap -> escalate) and verifier / judge ensembles as graph nodes — cascade()/make_acceptor() + vote()/majority(); kinds in /api/graph/run; Graph tab builder UI
- [ ] Per-subsystem config + versioning; asymmetric modality edges between stages
- [~] Mixture-of-models by task (task_router) — DONE as the node selector; graph edges TODO

## Swarm / fractal AI (orchestration primitives)
- [x] Subagents: /api/agent/swarm runs+merges sub-agents
- [x] Recursive agent trees (agents spawning agents) — 'fractal': spawn_agent tool, bounded by a shared depth+total fork-bomb budget; wired into /api/agent/run and the swarm
- [ ] Task/model sharding across the distributed runtime (multi-node + gateway substrate exists)
- [ ] Coordination/merge strategies (map-reduce, tournament, debate, blackboard)

## Component-aware / multimodal anticensorship
- [x] Composition map: identify parts (vision/audio encoder, connector, language model, moderation head, vocoder) + prescribe per-part technique
- [x] Per-part abliteration execution (scope edits to a named part) — abliterate_gguf part= filter + part_writing_matrices; composition reports executable_now vs needs_probing
- [x] Modality refusal directions (image/audio embedding space), not just text — modality.py contrastive direction on encoder embeddings, HELD-OUT (cross-validated) separability so the score is honest; /api/abliteration/modality-direction + Analysis-tab control (honest 503 when no embeddings/multimodal adapter)
- [ ] Cross-modal connector re-alignment (let filtered concepts pass the projection)
- [x] Moderation-head DETACH (disable a bolted-on classifier rather than cut a direction) — detach_part_gguf zeros the classifier tensors; /api/abliteration/detach
- [ ] Per-part versioning + lineage (each part independently versionable/revertable)
- [ ] Modality-specific in/out censorship diagnosis (refuses-to-describe-image, refuses-audio, etc.)

## Model intelligence / routing
- [x] Task-aware routing (classify -> route; user level fast/balanced/max); /api/route + auto:task
- [~] Model-based task classifier (replace heuristics); eval-driven selection using measured safety/capability
- [ ] Hand-labeled model tags/tiers in the registry (override inferred tags)

## Integration / exposure
- [x] MCP server (crucible-mcp) + MCP client (bidirectional)
- [x] Embeddable tool exposers: GET /api/tools catalog + POST /api/tools/invoke
- [ ] Tool-manifest export + JS/Python client SDKs for embedding in other apps
- [ ] Richer MCP server surface (expose more of the pipeline as MCP tools)
- [ ] Slash-command sets importable from other harnesses/models (Claude Code / OpenCode packs)

## Context & memory
- [x] Context compaction (summarize old turns when the window fills) — context.py (heuristic token estimate + summarize old / keep recent); /api/agent/compact + auto_compact on runs; composer meter + "compact" button + "auto" toggle
- [ ] Persistent project memory (CLAUDE.md-style) surfaced to the agent
- [ ] Web-forge session persistence (history survives reload; CLI already has sessions)

## Harness parity (Claude Code / OpenCode)
- [ ] Plan mode (read-only 'propose before acting')
- [ ] File-edit checkpointing / undo (revert the agent's file changes)
- [ ] Agent hooks (pre/post-tool)
- [ ] Slash commands in the web GUI (CLI has them)
- [ ] LSP integration; usage/cost/token accounting; vision input; IDE extension

## Platform / infra (worth adding)
- [ ] Prompt/response caching; budget/cost caps
- [ ] RAG / embeddings store
- [ ] Observability / tracing of agent + graph runs
- [ ] Plugin system; batch/queue processing; streaming tool results
