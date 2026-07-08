from __future__ import annotations
# FAST, non-LLM event detection over a video — the "tiny detector subagent" that watches densely for
# salient moments so the AI can REACT at the exact instant (a scene cut / jumpscare) without waiting on
# a slow vision-model description. Pure ffmpeg signal analysis: cheap, runs in a pass or two.
#
# Emits typed, timestamped events: {t, type, intensity}. type ∈ {scene_cut, jumpscare, loud}. These feed
# the co-watch stream as real-time `reaction`s and, downstream, hooks like a VTuber rig.
import re
import subprocess

_PTS = re.compile(r"pts_time:([0-9.]+)")
_SCENE = re.compile(r"scene_score=([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end: ([0-9.]+)")


def scene_cuts(video_path: str, threshold: float = 0.35, timeout: float = 300.0) -> list[dict]:
    """Timestamps where the picture changes abruptly (scene cut), via ffmpeg's scene score. A very high
    score is flagged 'jumpscare' (hard cut), else 'scene_cut'. One decode pass, no model."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-i", video_path, "-filter:v",
             f"select='gt(scene,{threshold})',showinfo,metadata=print", "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=timeout).stderr
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    events: list[dict] = []
    pending_t = None
    for line in out.splitlines():
        mt = _PTS.search(line)
        if mt:
            pending_t = float(mt.group(1))
        ms = _SCENE.search(line)
        if ms and pending_t is not None:
            score = float(ms.group(1))
            events.append({"t": round(pending_t, 2),
                           "type": "jumpscare" if score >= 0.75 else "scene_cut",
                           "intensity": round(score, 3)})
            pending_t = None
    return events


def loud_moments(video_path: str, noise_db: int = -18, min_gap: float = 0.4,
                 timeout: float = 300.0) -> list[dict]:
    """Timestamps where sound jumps loud after quiet (a bang/scream — often a jumpscare cue). Uses
    ffmpeg silencedetect: the END of a silence = onset of loudness. No model."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-i", video_path, "-af",
             f"silencedetect=noise={noise_db}dB:d=0.3", "-f", "null", "-"],
            capture_output=True, text=True, timeout=timeout).stderr
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    events: list[dict] = []
    last = -1e9
    for line in out.splitlines():
        m = _SILENCE_END.search(line)
        if m:
            t = float(m.group(1))
            if t - last >= min_gap:
                events.append({"t": round(t, 2), "type": "loud", "intensity": 1.0})
                last = t
    return events


def detect_events(video_path: str, want_audio: bool = True) -> list[dict]:
    """All fast detector events, merged + sorted by time. A scene cut coinciding (±0.4s) with a loud
    onset is upgraded to 'jumpscare' (sudden picture + sound is the classic cue)."""
    scenes = scene_cuts(video_path)
    louds = loud_moments(video_path) if want_audio else []
    for s in scenes:
        if s["type"] != "jumpscare" and any(abs(s["t"] - l["t"]) <= 0.4 for l in louds):
            s["type"] = "jumpscare"
    return sorted(scenes + louds, key=lambda e: e["t"])
