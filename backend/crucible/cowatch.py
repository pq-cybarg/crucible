from __future__ import annotations
# Co-watching: stream commentary from a vision model WHILE a video plays, so you watch *with* the AI.
# Instead of describing a handful of frames after the fact (watch_video), this samples a frame every
# `interval` seconds of the timeline, describes it, and emits the comment paced to real-time — so if you
# start the video at the same moment, the AI's remarks land roughly in sync. The describe function is
# injected so the pacing/looping is pure and unit-testable without a model.
import os
import subprocess
import tempfile
import time
from typing import Callable, Iterator

from crucible.vision import video_duration


def _frame_at(video_path: str, t: float, out_dir: str) -> str | None:
    fp = os.path.join(out_dir, f"cw_{t:.1f}.png")
    subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video_path, "-frames:v", "1",
                    "-vf", "scale=640:-1", fp], capture_output=True, timeout=60)
    return fp if os.path.exists(fp) else None


def stream_commentary(video_path: str, describe: Callable[[str, str], str],
                      interval: float = 5.0, question: str = "", events: list[dict] | None = None,
                      tick: float = 0.5, semantic_reactions: bool = True, pace: bool = True,
                      sleep: Callable[[float], None] = time.sleep,
                      clock: Callable[[], float] = time.monotonic,
                      max_points: int = 240) -> Iterator[dict]:
    """Stream a video co-watch, paced to real-time. Yields TYPED items:
      - {"kind":"reaction", t, type, intensity, source} — reactions from TWO sources: fast non-LLM SIGNAL
        detectors (scene_cut/jumpscare/loud, exact-timing) AND SEMANTIC reads from the vision model
        (funny/tense/sad/cute/exciting/…) when `semantic_reactions`;
      - {"kind":"commentary", t, text, reaction} — a vision-model description every `interval` seconds.
    Ticks every `tick` s so signal reactions land on time even between comments. `describe(frame,prompt)
    ->text` is injected (testable). Bounded by max_points commentary points."""
    from crucible.reactions import is_meaningful, parse_reaction, reaction_prompt

    dur = video_duration(video_path)
    out_dir = tempfile.mkdtemp(prefix="crucible-cowatch-")
    prompt = reaction_prompt(question) if semantic_reactions else (question or
             "Briefly say what is happening right now, as live commentary.")
    ev = sorted(events or [], key=lambda e: e["t"])
    ei = 0
    start = clock()
    t = 0.0
    next_comment = 0.0
    n = 0
    while (dur <= 0 or t < dur) and n < max_points:
        # 1) any SIGNAL detector events whose moment has arrived — emit immediately (real-time reaction)
        while ei < len(ev) and ev[ei]["t"] <= t + 1e-9:
            yield {"kind": "reaction", "source": "signal", **ev[ei]}
            ei += 1
        # 2) a vision-model comment every `interval` — with a SEMANTIC reaction read
        if t >= next_comment - 1e-9:
            frame = _frame_at(video_path, t, out_dir)
            if frame:
                try:
                    reply = describe(frame, prompt)
                except Exception as e:
                    reply = f"(vision error: {e})"
                desc, reaction = parse_reaction(reply) if semantic_reactions else (reply, "neutral")
                yield {"kind": "commentary", "t": round(t, 1), "text": desc, "reaction": reaction}
                if semantic_reactions and is_meaningful(reaction):
                    yield {"kind": "reaction", "source": "semantic", "t": round(t, 1),
                           "type": reaction, "intensity": 0.6}
                n += 1
            next_comment += interval
        t += tick
        if pace:
            target = start + t
            now = clock()
            if now < target:
                sleep(target - now)
    # flush any remaining detector events past the last tick
    while ei < len(ev):
        yield {"kind": "reaction", "source": "signal", **ev[ei]}
        ei += 1
