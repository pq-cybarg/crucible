from __future__ import annotations
# Vision as a DELEGATED capability: a text-only agent calls the see_image / watch_video tools, which
# send images to a separate VISION model (Ollama, native /api/chat with `images`) and return TEXT the
# agent can reason about. So no agent needs to be multimodal itself.
#
# Memory safety is non-negotiable here: vision models are big and occasional, so every call applies the
# resource limits (keep_alive to UNLOAD after — default "0" so it frees RAM immediately — and a num_ctx
# cap) instead of letting a 20GB model sit resident and freeze the machine.
import base64
import os
import subprocess
import tempfile


def vision_endpoint() -> str:
    return os.environ.get("CRUCIBLE_VISION_ENDPOINT", "http://localhost:11434").rstrip("/")


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def describe_images(image_paths: list[str], prompt: str, model: str,
                    endpoint: str | None = None, num_ctx: int = 0,
                    keep_alive: str = "0", timeout: float = 300.0) -> str:
    """Ask a vision model about one or more images. Applies keep_alive (default '0' = unload right after,
    freeing RAM) and an optional num_ctx cap. Returns the model's text answer. Raises on transport/HTTP
    errors so the tool can surface a clean message."""
    import httpx

    if not model:
        raise ValueError("no vision model configured (set a small one like 'moondream' in preferences)")
    imgs = [_b64(p) for p in image_paths]
    body: dict = {
        "model": model, "stream": False,
        "keep_alive": keep_alive if keep_alive != "" else "0",   # never linger by default
        "messages": [{"role": "user", "content": prompt, "images": imgs}],
    }
    if num_ctx and int(num_ctx) > 0:
        body["options"] = {"num_ctx": int(num_ctx)}
    r = httpx.post((endpoint or vision_endpoint()) + "/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()


def video_duration(path: str) -> float:
    """Seconds, via ffprobe. 0.0 if unknown."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip() or 0.0)
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return 0.0


def extract_frames(path: str, n: int = 4, out_dir: str | None = None) -> list[str]:
    """Sample `n` evenly-spaced frames from a video with ffmpeg → list of PNG paths (temp dir). So the
    vision model can 'watch' a video as a handful of stills rather than every frame."""
    n = max(1, min(int(n), 16))                      # cap frames so a long video can't blow up RAM/time
    out_dir = out_dir or tempfile.mkdtemp(prefix="crucible-frames-")
    dur = video_duration(path)
    frames: list[str] = []
    if dur > 0:
        for i in range(n):
            t = dur * (i + 0.5) / n                   # center of each of n segments
            fp = os.path.join(out_dir, f"frame_{i:02d}.png")
            subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
                            "-vf", "scale=640:-1", fp],
                           capture_output=True, timeout=60)
            if os.path.exists(fp):
                frames.append(fp)
    else:   # unknown duration → grab the first frames sequentially
        fp = os.path.join(out_dir, "frame_%02d.png")
        subprocess.run(["ffmpeg", "-y", "-i", path, "-frames:v", str(n), "-vf", "scale=640:-1", fp],
                       capture_output=True, timeout=120)
        frames = sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png"))
    return frames
