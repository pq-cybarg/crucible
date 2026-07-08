"""Fast non-LLM event detection (scene cuts / jumpscares) — the real-time reaction signal."""
import shutil
import subprocess

import pytest

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")


def _clip(path: str, color: str, dur: float) -> None:
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x240:d={dur}:r=10",
                    str(path)], capture_output=True, timeout=60)


def test_scene_cut_is_detected(tmp_path):
    from crucible.detect import scene_cuts
    # concat black → white → black: two hard cuts the scene detector should catch
    a, b, c = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "w.mp4"
    _clip(str(a), "black", 1.5); _clip(str(c), "white", 1.5); _clip(str(b), "black", 1.5)
    lst = tmp_path / "l.txt"
    lst.write_text(f"file '{a}'\nfile '{c}'\nfile '{b}'\n")
    joined = tmp_path / "joined.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(joined)], capture_output=True, timeout=60)

    cuts = scene_cuts(str(joined), threshold=0.3)
    assert len(cuts) >= 1                                  # at least one hard cut found
    assert all(c["type"] in ("scene_cut", "jumpscare") for c in cuts)
    assert all("t" in c and "intensity" in c for c in cuts)
    # a cut lands near a boundary (~1.5s or ~3.0s), not at t=0
    assert any(c["t"] > 1.0 for c in cuts)


def test_detect_events_merges_and_sorts(tmp_path):
    from crucible.detect import detect_events
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _clip(str(a), "black", 1.0); _clip(str(b), "white", 1.0)
    lst = tmp_path / "l.txt"; lst.write_text(f"file '{a}'\nfile '{b}'\n")
    joined = tmp_path / "j.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)],
                   capture_output=True, timeout=60)
    ev = detect_events(str(joined), want_audio=False)     # silent clips → scene cuts only
    assert ev == sorted(ev, key=lambda e: e["t"])         # sorted by time
