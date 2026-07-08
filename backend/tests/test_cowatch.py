"""Real-time co-watch: paced commentary stream from a vision model while a video plays. Tests the
pure pacing/looping with an injected describe fn + a fake clock — no model, no real sleeping."""
import shutil
import subprocess

import pytest

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")


def _make_video(path: str, dur: int = 6) -> None:
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=5:duration={dur}",
                    str(path)], capture_output=True, timeout=60)


def test_cowatch_streams_paced_commentary(tmp_path):
    from crucible.cowatch import stream_commentary
    vid = tmp_path / "clip.mp4"
    _make_video(str(vid), dur=6)

    seen_frames = []
    def describe(frame, prompt):
        seen_frames.append(frame)
        return f"frame#{len(seen_frames)}"

    slept = []
    fake_now = [0.0]
    def sleep(s): slept.append(s); fake_now[0] += s      # fake sleep advances the fake clock
    def clock(): return fake_now[0]

    points = list(stream_commentary(str(vid), describe, interval=2.0, question="what?",
                                    pace=True, sleep=sleep, clock=clock))
    # a 6s clip at 2s interval → commentary at t=0,2,4
    assert [p["t"] for p in points] == [0.0, 2.0, 4.0]
    assert all(p["text"].startswith("frame#") for p in points)
    assert len(seen_frames) == 3
    # it paced (slept) between points to track the timeline
    assert len(slept) >= 2 and all(s >= 0 for s in slept)


def test_cowatch_bounded_by_max_points(tmp_path):
    from crucible.cowatch import stream_commentary
    vid = tmp_path / "clip.mp4"
    _make_video(str(vid), dur=6)
    points = list(stream_commentary(str(vid), lambda f, p: "x", interval=1.0, pace=False, max_points=2))
    assert len(points) == 2      # capped
