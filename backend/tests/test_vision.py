"""Vision/video tools — delegate to a vision model so text-only agents can 'see'. Tests never load a
real (big) model: they mock the vision HTTP call and use ffmpeg only for frame extraction."""
import shutil
import subprocess

import pytest

_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _make_video(path: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=5:duration=2",
                    str(path)], capture_output=True, timeout=60)


def test_see_image_requires_a_configured_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path))     # empty prefs → no vision model
    from crucible.tools.vision import SeeImage
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n")
    r = SeeImage(root=str(tmp_path)).run(path="x.png")
    assert r.ok is False and "vision model" in r.error


def test_see_image_calls_vision_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path))
    from crucible.prefs import PreferencesStore
    PreferencesStore(tmp_path / "preferences.json").save(
        {"vision_model": "moondream", "resource_limits": {"keep_alive": "0", "num_ctx": 2048}})
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\nfake")

    sent = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "a red square"}}

    def fake_post(url, json, timeout):
        sent["url"] = url
        sent["body"] = json
        return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "post", fake_post)
    from crucible.tools.vision import SeeImage
    r = SeeImage(root=str(tmp_path)).run(path="shot.png", question="what shape?")
    assert r.ok and r.output == "a red square"
    assert sent["url"].endswith("/api/chat")
    assert sent["body"]["model"] == "moondream"
    assert sent["body"]["keep_alive"] == "0"                   # unloads after — no lingering RAM
    assert sent["body"]["options"]["num_ctx"] == 2048          # ctx cap applied
    assert sent["body"]["messages"][0]["images"]               # image was attached (base64)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
def test_extract_frames_from_a_real_video(tmp_path):
    from crucible.vision import extract_frames, video_duration
    vid = tmp_path / "clip.mp4"
    _make_video(str(vid))
    assert video_duration(str(vid)) > 1.0
    frames = extract_frames(str(vid), n=4)
    assert len(frames) == 4 and all(f.endswith(".png") for f in frames)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
def test_watch_video_samples_frames_and_describes(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path))
    from crucible.prefs import PreferencesStore
    PreferencesStore(tmp_path / "preferences.json").save({"vision_model": "moondream"})
    vid = tmp_path / "clip.mp4"
    _make_video(str(vid))

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "a test pattern moves"}}

    import httpx
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: _Resp())
    from crucible.tools.vision import WatchVideo
    r = WatchVideo(root=str(tmp_path)).run(path="clip.mp4", frames=3)
    assert r.ok and "3 frames sampled" in r.output and "test pattern" in r.output
