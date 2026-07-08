from __future__ import annotations
# see_image / watch_video: let ANY agent (even a text-only one) look at images and videos by delegating
# to a configured VISION model. Both read the vision model + resource limits from preferences and always
# unload the model after (keep_alive) so a big model can't linger and freeze the machine.
import os

from crucible.tools.base import ToolResult


def _vision_config() -> dict:
    """Read {vision_model, num_ctx, keep_alive} from the persisted preferences (UI-configurable)."""
    try:
        from crucible.config import get_settings
        from crucible.prefs import PreferencesStore
        prefs = PreferencesStore(get_settings().data_dir / "preferences.json").get()
        rl = prefs.get("resource_limits", {})
        # NOTE: do NOT apply the general num_ctx cap to vision — an image needs a large context for its
        # tokens, and a small cap (e.g. 2048) makes the vision model 400 or return nothing. Memory
        # safety for vision comes from keep_alive (unload right after), not from starving context.
        return {"model": prefs.get("vision_model", ""),
                "num_ctx": 0,
                "keep_alive": str(rl.get("keep_alive", "") or "0")}
    except Exception:
        return {"model": "", "num_ctx": 0, "keep_alive": "0"}


class SeeImage:
    name = "see_image"
    description = ("Look at an image file and answer a question about it (delegates to a vision model, "
                   "so a text-only agent can 'see'). Args: path (image file), question (optional).")
    parameters = {"type": "object",
                  "properties": {"path": {"type": "string"}, "question": {"type": "string"}},
                  "required": ["path"]}

    def __init__(self, root=None):
        self.root = str(root) if root else "."

    def _resolve(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    def run(self, path: str = "", question: str = "") -> ToolResult:
        cfg = _vision_config()
        if not cfg["model"]:
            return ToolResult(ok=False, output="", error="no vision model set — pick a SMALL one (e.g. "
                              "'moondream' or 'llava:7b') in Preferences → vision model")
        full = self._resolve(path)
        if not os.path.exists(full):
            return ToolResult(ok=False, output="", error=f"image not found: {full}")
        from crucible.vision import describe_images
        try:
            out = describe_images([full], question or "Describe this image in detail.", cfg["model"],
                                  num_ctx=cfg["num_ctx"], keep_alive=cfg["keep_alive"])
            return ToolResult(ok=True, output=out or "(no description returned)")
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"vision failed: {e}")


class WatchVideo:
    name = "watch_video"
    description = ("Watch a video by sampling frames and describing them (delegates to a vision model). "
                   "A local file path OR a video URL (YouTube etc. — downloaded low-res first). Args: "
                   "path (file or URL), question (optional), frames (default 6, max 16), "
                   "max_seconds (optional: only the opening window, for long/URL videos).")
    parameters = {"type": "object",
                  "properties": {"path": {"type": "string"}, "question": {"type": "string"},
                                 "frames": {"type": "number"}, "max_seconds": {"type": "number"}},
                  "required": ["path"]}

    def __init__(self, root=None):
        self.root = str(root) if root else "."

    def run(self, path: str = "", question: str = "", frames: int = 6, max_seconds: float = 0.0) -> ToolResult:
        cfg = _vision_config()
        if not cfg["model"]:
            return ToolResult(ok=False, output="", error="no vision model set — pick a SMALL one in "
                              "Preferences → vision model")
        from crucible.vision import describe_frames, download_video, extract_frames, is_url
        note = ""
        try:
            if is_url(path):
                full = download_video(path, max_height=360, max_seconds=max_seconds)
                note = f"downloaded {path} → {os.path.basename(full)}\n"
            else:
                full = path if os.path.isabs(path) else os.path.join(self.root, path)
            if not os.path.exists(full):
                return ToolResult(ok=False, output="", error=f"video not found: {full}")
            imgs = extract_frames(full, int(frames))
            if not imgs:
                return ToolResult(ok=False, output="", error="could not extract frames (is ffmpeg installed?)")
            # describe one frame at a time (small vision models overflow on multiple images), then combine
            q = question or "Describe what is shown."
            out = describe_frames(imgs, q, cfg["model"])
            return ToolResult(ok=True, output=f"{note}[{len(imgs)} frames sampled, in order]\n{out}")
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"watch_video failed: {e}")
