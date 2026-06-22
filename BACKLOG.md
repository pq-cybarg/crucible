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
- [ ] Insertion auto-tuner (search layers × coefficient × direction-source for a coherent+effective window)
- [x] Restoration via remove-the-suppressor (target-prompt suppressor direction, proven removal)

## crucible CLI — full Claude-Code parity
- [x] Functional CLI (agent loop, remote-aware --endpoint, slash commands)
- [ ] Persistent sessions
- [ ] Settings file (~/.crucible/settings.json)
- [ ] MCP server support
- [ ] Full slash-command set + status line + config

## Remote Windows orchestration
- [x] Remote-aware endpoints (--endpoint / CRUCIBLE_ENDPOINT)
- [ ] Drive the real 1.5 TB-class model on the Windows node end-to-end (in-place edit, diagnose, serve over the network)

## Benchmarks
- [x] lm-eval integration (canonical suite, capability retention)
- [ ] loglikelihood MC tasks via the completions backend (MMLU/ARC/HellaSwag)
