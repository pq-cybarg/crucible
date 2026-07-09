from __future__ import annotations
# Idle LIFE for the companion face — the small, involuntary motion that makes an avatar read as alive
# rather than a frozen sprite, layered ON TOP of whatever emotion it's expressing. Three signals, all
# driven off a low frame-rate tick so they're cheap to redraw in the TUI:
#
#   • SACCADES — the eyes don't hold dead-centre; they flick to a new fixation point every second or two
#     (ballistic jump, then hold), with an occasional larger glance. Emitted as a GAZE axis (dx,dy) in
#     [-1,1] that `avatar.render_sprites(gaze=…)` turns into a few-pixel pupil offset — MIXABLE with any
#     expression (a happy face can look away).
#   • BLINKS — a natural, slightly irregular cadence (not a metronome).
#   • MICRO-EXPRESSIONS — a faint, brief accent mood (a flicker of curious/smug/…) overlaid at low weight
#     on the current expression blend, so the face is subtly, continuously alive.
#
# Deterministic: seeded RNG + an explicit tick counter → the same sequence every run (testable, and it
# won't use wall-clock randomness). Advance one `step()` per face tick.
import random
from dataclasses import dataclass, field

# subtle accents to flicker as micro-expressions (kept gentle so they never override the real mood)
MICRO_ACCENTS = ("curious", "happy", "smug", "surprised", "bored")


@dataclass
class IdleState:
    gaze: tuple = (0.0, 0.0)     # current eased look-direction in [-1,1]
    blink: bool = False          # blink this tick
    micro: dict = field(default_factory=dict)   # {expression: weight} accent to overlay (may be empty)


class IdleAnimator:
    """Stateful, seeded generator of idle face motion. Call `step()` once per face tick; it returns an
    `IdleState` (gaze / blink / micro-expression) to layer on top of the driven expression."""

    def __init__(self, seed: int = 0, blink_every: tuple = (12, 30), saccade_every: tuple = (4, 12),
                 micro_every: tuple = (16, 44), ease: float = 0.55):
        self._r = random.Random(seed)
        self._blink_every = blink_every
        self._saccade_every = saccade_every
        self._micro_every = micro_every
        self._ease = ease
        self._t = 0
        self._gaze = (0.0, 0.0)
        self._target = (0.0, 0.0)
        self._next_saccade = self._r.randint(*saccade_every)
        self._next_blink = self._r.randint(*blink_every)
        self._next_micro = self._r.randint(*micro_every)
        self._micro: dict = {}
        self._micro_until = 0

    def _new_fixation(self) -> tuple:
        # mostly small glances near centre; occasionally a wider look. Bias slightly toward centre so the
        # eyes keep returning to the viewer rather than drifting away.
        big = self._r.random() < 0.25
        span = 0.9 if big else 0.4
        return (round(self._r.uniform(-span, span), 3), round(self._r.uniform(-span * 0.6, span * 0.6), 3))

    def step(self) -> IdleState:
        self._t += 1

        if self._t >= self._next_saccade:                       # pick a new fixation point (ballistic)
            self._target = self._new_fixation()
            self._next_saccade = self._t + self._r.randint(*self._saccade_every)
        # ease toward the target: a quick move then a hold (a saccade, not a slow pan)
        gx = self._gaze[0] + (self._target[0] - self._gaze[0]) * self._ease
        gy = self._gaze[1] + (self._target[1] - self._gaze[1]) * self._ease
        self._gaze = (round(gx, 4), round(gy, 4))

        blink = self._t >= self._next_blink                     # slightly irregular blink cadence
        if blink:
            self._next_blink = self._t + self._r.randint(*self._blink_every)

        if self._t >= self._next_micro:                         # start a brief micro-expression flicker
            accent = self._r.choice(MICRO_ACCENTS)
            self._micro = {accent: round(self._r.uniform(0.08, 0.18), 3)}
            self._micro_until = self._t + self._r.randint(2, 5)
            self._next_micro = self._micro_until + self._r.randint(*self._micro_every)
        if self._t >= self._micro_until:
            self._micro = {}

        return IdleState(gaze=self._gaze, blink=blink, micro=dict(self._micro))
