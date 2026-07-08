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

    items = list(stream_commentary(str(vid), describe, interval=2.0, question="what?",
                                   pace=True, sleep=sleep, clock=clock))
    comments = [i for i in items if i["kind"] == "commentary"]
    # a 6s clip at 2s interval → commentary at t=0,2,4
    assert [c["t"] for c in comments] == [0.0, 2.0, 4.0]
    assert all(c["text"].startswith("frame#") for c in comments)
    assert len(seen_frames) == 3
    # it paced (slept) between points to track the timeline
    assert len(slept) >= 2 and all(s >= 0 for s in slept)


def test_cowatch_interleaves_reactions_at_their_moment(tmp_path):
    from crucible.cowatch import stream_commentary
    vid = tmp_path / "clip.mp4"
    _make_video(str(vid), dur=6)
    events = [{"t": 1.0, "type": "jumpscare", "intensity": 0.9},
              {"t": 3.5, "type": "scene_cut", "intensity": 0.4}]
    items = list(stream_commentary(str(vid), lambda f, p: "c", interval=2.0, events=events,
                                   tick=0.5, pace=False))
    reacts = [i for i in items if i["kind"] == "reaction"]
    assert [(r["t"], r["type"]) for r in reacts] == [(1.0, "jumpscare"), (3.5, "scene_cut")]
    # the jumpscare (t=1.0) is emitted BEFORE the t=2 commentary — i.e. at its moment, not batched
    order = [(i["kind"], i["t"]) for i in items]
    assert order.index(("reaction", 1.0)) < order.index(("commentary", 2.0))


def test_cowatch_bounded_by_max_points(tmp_path):
    from crucible.cowatch import stream_commentary
    vid = tmp_path / "clip.mp4"
    _make_video(str(vid), dur=6)
    comments = [i for i in stream_commentary(str(vid), lambda f, p: "x", interval=1.0, pace=False, max_points=2)
                if i["kind"] == "commentary"]
    assert len(comments) == 2      # capped
