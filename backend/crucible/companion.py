from __future__ import annotations
# The real-time COMPANION DRIVE LOOP — the beating heart that turns the AI's emotional STATE into a smooth,
# continuous stream of face frames, decoupled from the slow reply cycle. Reactions/emotions arrive
# asynchronously (from the co-watch stream, chat, tool events) and set a TARGET mood; the driver eases the
# current mood toward it every tick (layered crossfade, never a jump-cut), layers idle life (saccadic gaze,
# blink, micro-expressions) and TALK lip-sync on top, and emits a `rigmap.rig_frame` — the engine-agnostic
# frame every consumer speaks (TUI pixel face, web SVG/Live2D, external VTube Studio rig).
#
# Deterministic by design: an internal tick counter (not wall-clock) drives easing/breath/blink so a given
# seed + input sequence yields the same frames — unit-testable. Thread-safe: `react`/`set_mood`/`set_talking`
# may be called from any thread while a render loop steps the driver.
import math
import threading
from typing import Optional

from crucible.animation import IdleAnimator
from crucible.expression import expression_for
from crucible.rigmap import rig_frame


def _clean_weights(weights: dict) -> dict:
    out = {str(k): float(v) for k, v in (weights or {}).items() if v and v > 0}
    return out or {"neutral": 1.0}


class CompanionDriver:
    """Holds the companion's live emotional state and produces smoothed face frames. Call `step()` once per
    tick from a render loop (or use `run`/`frames`); update the mood from any thread via `set_mood`/`react`.
    """

    def __init__(self, seed: int = 7, ease: float = 0.25, blink_hold: int = 3,
                 breath_period: int = 90, breath_depth: float = 0.05):
        self._idle = IdleAnimator(seed=seed)
        self._ease = ease
        self._blink_hold = blink_hold
        self._breath_period = max(1, breath_period)
        self._breath_depth = breath_depth
        self._lock = threading.Lock()
        self._target = {"neutral": 1.0}     # where the mood is heading (set by reactions/emotions)
        self._weights = {"neutral": 1.0}    # the current, eased mood
        self._talking = False
        self._speech = 0.0                  # explicit lip-sync amplitude in [0,1] (overrides auto-flap)
        self._t = 0
        self._held = 0

    # --- inputs (thread-safe) ---------------------------------------------------------------------
    def set_mood(self, weights: dict) -> None:
        """Set the TARGET mood as expression weights (blendshape-style). The driver crossfades to it."""
        with self._lock:
            self._target = _clean_weights(weights)

    def react(self, reaction: str) -> None:
        """A reaction word (co-watch / chat vocabulary) → its expression preset becomes the target mood."""
        self.set_mood({expression_for(reaction).name: 1.0})

    def set_emotion(self, name: str) -> None:
        self.set_mood({name: 1.0})

    def set_talking(self, on: bool) -> None:
        with self._lock:
            self._talking = bool(on)
            if not on:
                self._speech = 0.0

    def set_speech_level(self, level: float) -> None:
        """Drive the mouth directly from live TTS/audio amplitude (0 closed … 1 wide) — real lip-sync."""
        with self._lock:
            self._speech = max(0.0, min(1.0, float(level)))
            if self._speech > 0:
                self._talking = True

    def mood(self) -> dict:
        with self._lock:
            return dict(self._weights)

    # --- the loop ---------------------------------------------------------------------------------
    def _mouth_open(self) -> float:
        if self._speech > 0.0:
            return self._speech                      # explicit amplitude wins (true lip-sync)
        if self._talking:
            return 0.55 if self._t % 2 == 0 else 0.08  # auto flap when talking without amplitude
        return 0.0

    def step(self) -> dict:
        """Advance one tick and return the current `rig_frame` (params + all engine mappings)."""
        self._t += 1
        with self._lock:
            target = dict(self._target)
            talking, speech = self._talking, self._speech
        # ease the current mood toward the target (smooth, layered crossfade — not a jump-cut)
        keys = set(self._weights) | set(target)
        nw: dict = {}
        for k in keys:
            cur = self._weights.get(k, 0.0)
            tgt = target.get(k, 0.0)
            v = cur + (tgt - cur) * self._ease
            if v > 1e-3:
                nw[k] = round(v, 5)
        self._weights = nw or {"neutral": 1.0}

        idle = self._idle.step()
        if idle.blink:
            self._held = self._blink_hold
        blink = 1.0 if self._held > 0 else 0.0
        self._held = max(0, self._held - 1)

        weights = dict(self._weights)
        for name, w in idle.micro.items():           # faint micro-expression flicker layered on the mood
            weights.setdefault(name, w)

        extra: dict = {}
        mouth = self._mouth_open() if (talking or speech > 0) else self._mouth_open()
        if mouth > 0:
            extra["mouth_open"] = mouth              # lip-sync / talk lifts the mouth open param
        breath = math.sin(2 * math.pi * (self._t % self._breath_period) / self._breath_period)
        extra["head_tilt"] = breath * self._breath_depth   # subtle idle breath/sway

        return rig_frame(weights, gaze=idle.gaze, extra=extra, blink=blink)

    def frames(self, count: int):
        """A pure generator of `count` frames (timing is the caller's — sleep between yields at your fps)."""
        for _ in range(count):
            yield self.step()

    async def run(self, send, fps: int = 20, count: Optional[int] = None) -> int:
        """Step at `fps`, awaiting `send(frame)` for each. Runs until `count` frames (or forever if None)."""
        import asyncio
        pushed = 0
        while count is None or pushed < count:
            await send(self.step())
            pushed += 1
            await asyncio.sleep(1.0 / max(1, fps))
        return pushed
