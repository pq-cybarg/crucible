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
                      interval: float = 5.0, question: str = "", pace: bool = True,
                      sleep: Callable[[float], None] = time.sleep,
                      clock: Callable[[], float] = time.monotonic,
                      max_points: int = 240) -> Iterator[dict]:
    """Yield {t, text} commentary for a video, one frame every `interval` seconds. `describe(frame, prompt)
    -> text` is the injected vision call. With `pace`, emissions are spaced to wall-clock so they track
    the video timeline (start your playback at the same time to watch along). Bounded by max_points."""
    dur = video_duration(video_path)
    out_dir = tempfile.mkdtemp(prefix="crucible-cowatch-")
    prompt = question or "Briefly say what is happening right now, as live commentary."
    start = clock()
    t = 0.0
    n = 0
    while (dur <= 0 or t < dur) and n < max_points:
        frame = _frame_at(video_path, t, out_dir)
        if frame:
            try:
                text = describe(frame, prompt)
            except Exception as e:
                text = f"(vision error: {e})"
            yield {"t": round(t, 1), "text": text}
            n += 1
        t += interval
        if pace:
            target = start + t
            now = clock()
            if now < target:
                sleep(target - now)
        if dur > 0 and t >= dur:
            break
