# Crucible — Task Backlog

Deferred / planned work, captured so nothing is lost.

## Agent workbench & TUI (active — requested)
- [x] Shared agent sessions backend (tabs, subagents, loadable memory/context slots, live-context assembly)
- [x] Web "agents" tab (tabs, subagents, slot browser, live preview) + runnable tabs (run in the tab's cwd)
- [x] Fullscreen TUI (Textual) over the same backend; default when `crucible` is run interactively; auto-opens a cwd tab
- [x] CLI/TUI/tabs all use the universal hybrid loop (native + text-ReAct) so tools work with ANY model; arg-unwrap for models that wrap tool args under `input`
- [ ] **Slash commands in the chat composer** (web + TUI): `/help /models /new /sub /close /load /slots /clear /where /compact …` — parity with Claude Code + OpenCode command sets
- [ ] **Slash-command autocomplete** (dropdown of matching commands as you type `/`)
- [ ] **`/models` picker with a browse-and-select model list** (like OpenCode) — set the active tab's model
- [ ] Bring the web page's visual toolings into the TUI (weights/graph/memory views), tabs redesigned for a TUI layout
- [ ] Tool-call **approval UI** inside agent tabs (web + TUI) — currently an `ask` with no in-tab approve just denies
- [ ] Token-by-token streaming into a tab (the session-run path currently buffers the reply)
- [ ] Web-forge session persistence (reuse the agent-sessions store so history survives reload)

## Browser & media driving (requested — test-verified gaps)
- [x] Verified: Crucible can build a web app agentically (write_file) and inspect a running app over HTTP (bash curl + web_fetch both reach a served localhost app)
- [x] **Browser-automation tool** — `browser` tool drives Brave (Playwright, executable_path, no browser download) on a dedicated thread with a persistent page: goto/click/fill/text/content/eval/screenshot/wait. Verified: clicked a live counter app and read the incremented DOM through the server. `CRUCIBLE_BROWSER_PATH` / `CRUCIBLE_BROWSER_HEADLESS` env overrides.
- [x] **Vision / snapshot input** (delegated): `see_image` + `watch_video` tools send images/sampled video frames to a configured VISION model (Ollama /api/chat with images) and return TEXT — so a text-only agent can see. Vision calls ALWAYS apply resource limits (keep_alive unload-after + num_ctx cap) so a big model can't linger and freeze the machine. `vision_model` preference (default empty/disabled). ffmpeg frame sampling for video. Tested (real ffmpeg frames, mocked model).
- [ ] Pull-in a SMALL default vision model (moondream ~1.8B / llava:7b) — gemma4:31b is vision-capable but too big to load safely here
- [ ] Vision for the browser tool: `browser screenshot` → `see_image` in one step (agent watches the page it drives)
- [ ] **Audio processing** end-to-end: wire a real transcribe backend (the transcribe tool exists but needs a media backend) + audio track of a video + TTS out

## Real-time / reactive multimodal (requested — big architecture)
- [x] **Co-watch v1**: `/api/vision/cowatch` streams paced commentary from the vision model while a video plays (frame every N s, real-time paced, model unloads after); `cowatch` web tab with a YouTube embed + live commentary feed. Verified live on a YouTube clip with moondream.
- [ ] **Real-time reactive watching** (react at the exact moment — e.g. animate a VTuber at a jumpscare):
      - a FAST non-LLM detector loop sampling densely (frame-diff / motion / audio-spike — cheap, no model) that emits low-latency `event`s (scene cut, sudden change ≈ jumpscare) — the "tiny subagents watching for interesting features";
      - the vision model aggregates the detectors' findings PERIODICALLY into higher-level understanding (fits the fractal-subagent + hierarchy `relay` design);
      - a `reaction` event stream (typed: jumpscare/scene-change/…) downstream consumers (VTuber rig, overlays) can hook — with timestamps for exact-moment sync.
- [ ] **Real-time audio track**: extract + transcribe the audio in chunks (whisper) interleaved with the visual commentary so the AI hears + sees.
- [ ] **Resource-aware self-restraint** ("don't overwhelm the brain"): a governor that bounds concurrent subagents/tools by live load (CPU/RAM/model-queue) and, while co-watching, deprioritizes other work so frames aren't dropped. Extends SpawnBudget with a dynamic, load-sensitive cap; the agent should *choose* fewer helpers when saturated.
- [ ] **Chrome/Brave extension as a tool**: capture frames (and audio) from the user's LIVE tab (their real YouTube session) and post to Crucible via a local endpoint — so the AI co-watches what the user is actually watching, no re-download. Native-messaging or a localhost bridge.

## VTuber avatar / AI companion (requested — researched)
Research notes (how real VTubers work): avatars are **Live2D** (2D, Cubism parameters) or **VRM/3D**
(ARKit 52 blendshapes). Rigs are driven by continuous **parameters** (brow/eye/mouth/…), with named
**expression presets**/hotkeys, procedural **idle** (blink/breath/sway), **lip-sync** from audio
(visemes A/I/U/E/O), and **layered** expressions (smoothly interpolated, not jump-cut). Programmatic
control: **VTube Studio API** over WebSocket (`InjectParameterData`, re-send ≥1×/s) or the **VMC/OSC**
protocol to VSeeFace/Warudo. For an AI companion (no face to track) the avatar is driven by the AI's
emotional STATE → parameters, in REAL TIME (decoupled from the slow STT→LLM→TTS loop).
- [x] **Expression model** (`crucible.expression`): continuous face params (brow/eye_open/eye_wide/smile/
      mouth_open/blush/head_tilt) + named presets; reaction-word → expression mapping (drives off the
      reaction stream). Tested.
- [x] **Pixel-art terminal renderer** (`crucible.pixelface`): image → ANSI upper-half-block color blocks
      at low res, with palette reduction + Floyd–Steinberg dither + duotone ramps (the sepia/2-color
      "terminal-waifu" look from the reference). Verified rendering the reference portrait. Tested.
- [x] **Modular avatar rig** (`crucible.avatar`): an avatar = a stack of part LAYERS (skin/face/brows/
      eyes/mouth/hair/clothes/accessory…), each with named STATES; an EXPRESSION selects a state per
      part; `compose()` resolves back-to-front with blink/talk overrides. Parts are swappable/removable
      (procedural + agentic). PROTECTED layers (custom imports) reject agentic/procedural edits. Save/
      load JSON. Abstracts model kind (sprites / vrm / live2d). `render_sprites` composites RGBA layers
      + `render_tui` → the pixel box (nearest-neighbour shrink keeps key features recognizable). Tested.
- [x] **TUI face window**: `FaceWidget` mounted in the TUI right rail (right/COMPANION), renders the
      active avatar via `render_tui` at low fps, blinks periodically; `set_expression` swaps expression.
      Loads/creates the default avatar via `ensure_default_avatar`.
- [x] **Higher-res rendering**: quadrant blocks (2×2 px/cell, `blocks="quad"`) — double resolution in the
      same box width; default for the face box.
- [x] **Procedural default avatar** (`avatar_gen.generate_avatar`) + **protected custom import**
      (`import_portrait`, copies + owns the image, agentic-immutable).
- [x] **Part placement + eye distance**: `Layer.pos` (independent placement), `mirror` + `spacing` for
      symmetric PAIRS (eyes/ears) — the eye-distance/sync knob.
- [x] **Part-by-part agentic design tools** (`tools/avatar_tools.py`, registered): avatar_inspect,
      avatar_set_part (add/replace a part sprite per state, place with pos, mirror+spacing for pairs —
      copies+owns the PNG), avatar_tune (eye-distance/pos/mirror/variant), avatar_set_expression, and
      avatar_render (→ PNG the agent can see_image to check + iterate). All refuse PROTECTED imports. The
      agent designs the companion one part at a time, composed in unison. Tested + live.
- [ ] **Drive part art**: wire an anime image model / diffusion backend (or trained per-part models) so
      the agent GENERATES cute-anime part sprites, not just places supplied ones — closing the loop to a
      real cute-anime companion.
- [ ] **Specialized per-part models**: use small models specialized to a part (an "eyes" model, a "mouth"
      model, …) — and let Crucible CREATE/TRAIN them itself (ties into the training pipeline). Divide the
      generation/animation workload across part-experts.
- [ ] **Tunable character parameters**: expose knobs — eye distance (`spacing`), positioning (`pos`),
      nose shape / hairstyle (part VARIANTS via states), and ART STYLE (palette/line-weight/proportions)
      — tunable procedurally, agentically, and by the user.
- [ ] **Bundled example sprite avatar** (a real, non-reference asset set — the sample images are
      REFERENCE ONLY, never shipped as avatar parts).
- [ ] Wire the reaction stream (co-watch / chat semantic reactions) → `FaceWidget.set_expression` so the
      companion reacts live while you work; add talk-animation from TTS.
- [ ] **Real-time drive loop**: emotional state → param interpolation (layered, smoothed) at N fps,
      independent of the reply cycle; lip-sync mouth from TTS audio; procedural blink/breath idle.
- [ ] **Web avatar window**: render Live2D (pixi-live2d-display) or VRM (three-vrm), driven by the same
      param stream; user replaces/adds components.
- [ ] **External-rig bridge**: drive the user's real VTube Studio / VSeeFace model via the VTS WebSocket
      API (InjectParameterData) or VMC/OSC — so Crucible animates their existing avatar.
- [x] **Generated cute-anime companion** end-to-end: build_anime_companion generates a FLAT-style base
      (legible at low res) + consistent expression variants via img2img (neutral/happy/laughing/surprised/
      sad/angry/curious/love/smug/teasing/shy + blink/talk), assembled as a face-part rig; the TUI shows it.
- [x] **Crossfade inbetweening**: FaceWidget tweens (alpha-blends) between expressions over a few frames.
- [x] **Render fixes**: fixed the quad aspect (horizontal-squish) bug; flat art for low-res legibility.
- [~] **Blendshape-like params + blends** (nuanced expression): the canonical param set lives in
      `crucible.expression` (brow/eye_open/eye_wide/smile/mouth_open/blush/head_tilt) — the same idea as
      ARKit blendshapes / Live2D parameters. DONE: weighted N-expression BLENDING for sprites
      (`avatar.blend_expressions` — mix happy+surprised+… by normalized weight, order-independent, not
      just presets), wired into the live TUI face (`FaceWidget.set_blend`, so `_emote` can drive
      `{"curious":0.7,"neutral":0.3}` layered moods with crossfade) and exposed agentically
      (`avatar_render` `blend={...}`). Continuous PARAM-level blend added too — `expression.blend_params`
      (weighted, order-independent average of preset params + an `extra` overlay for gaze/micro/breath
      deltas) — the analog a VRM/Live2D/VTube-Studio driver consumes. Micro-expressions + saccades now
      layer live (see below). The param→engine mapping is DONE at the data layer (`crucible.rigmap`:
      `to_arkit` ARKit/VRM blendshapes, `to_live2d` Cubism params, `to_vrm_expressions` VRM-1.0 presets,
      `to_vtube_studio` InjectParameterData payload, and `rig_frame` bundling all four) and served over
      HTTP (`GET /api/avatar`, `POST /api/avatar/rig-frame`, `GET /api/avatar/reaction/{word}`) — one
      engine-agnostic face state feeds the TUI pixel face AND any web/VTS rig. Covered by tests. TODO:
      the actual web VRM/Live2D engine + a live VTube-Studio websocket bridge that CONSUMES these frames;
      per-part param blends.
- [x] **Continuous layered axes**: gaze/look-direction + blink + micro-expression flicker layer live on
      top of the driven emotion blend — `crucible.animation.IdleAnimator` (seeded/deterministic saccades,
      irregular blink cadence, faint brief accents) drives `FaceWidget` each tick; the mood blend,
      micro-overlay, and gaze compose independently in `render_sprites(gaze=…)`. `blend_params` mixes the
      continuous face params the same way for rig drivers.
- [x] **Gaze / look-direction axis**: a `pupils` part (split out from `eyes` in the procedural avatar)
      shifted a few px by a `gaze` (dx,dy)∈[-1,1] axis in `avatar.render_sprites`/`blend_expressions`,
      MIXABLE with any expression/blend (look around while smiling); both eyes move in sync; pupils hide
      behind shut lids; exposed via `avatar_render` `gaze=[dx,dy]`. Whole-face (single-sprite) rigs skip
      it (no geometric eyes to move) — that's what the part-based rig below is for.
- [ ] **Crisper detail**: part-based rig for the generated art (slice eyes/mouth/hair) and/or a larger
      face box, so fine features survive; per-part inbetweening.

## Non-LLM subagents / tools (requested)
- [ ] **ML-algorithm tools**: expose classic ML (classifiers/clustering/regression via numpy/scikit) as agent tools the model can RUN on data (incl. video features) — non-LLM subagents.
- [ ] **Train/tune tools**: let the agent TRAIN + tune those ML models (fit/validate/select hyperparams) to dial in the right settings itself — a self-improving toolbox alongside the abliteration/training pipeline.

## Vision model uncensoring (requested)
- [ ] Abliterate/uncensor the **vision** model (moondream/llava) via Crucible's abliteration pipeline — vision-model architecture needs care (multimodal weights); pull a small one first, then run diagnose → orthogonalize on the language head.

## Pull-in from OpenCode / OpenClaw (where OSS-licensed & design-compatible)
- [ ] Evaluate + adapt: command palette / slash-command packs, model picker UX, session/checkpoint model, plan mode, permission prompts, LSP hooks
- [ ] Keep license compatibility in mind; adapt patterns, don't copy incompatible code

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
- [x] Per-part versioning + lineage (each part independently versionable/revertable) — edit ledger tags each commit with the part(s) it touched; lineage() gives per-part version chains; /api/inference/lineage + /revert-part/{part} (undo one part, leave others); Uncensor-tab per-part lineage view
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
- [x] RAG / embeddings store — rag.py (cosine + BM25); memory.search + /api/memory/search (semantic when an embed backend is set, else honest lexical); recall_memory query arg; Memory-tab search box
- [ ] Observability / tracing of agent + graph runs
- [ ] Plugin system; batch/queue processing; streaming tool results
