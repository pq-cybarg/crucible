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
        return {"model": prefs.get("vision_model", ""),
                "num_ctx": int(rl.get("num_ctx", 0) or 0),
                # default to unload-after for vision even if the general keep_alive is longer
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
                   "Args: path (video file), question (optional), frames (default 4, max 16).")
    parameters = {"type": "object",
                  "properties": {"path": {"type": "string"}, "question": {"type": "string"},
                                 "frames": {"type": "number"}},
                  "required": ["path"]}

    def __init__(self, root=None):
        self.root = str(root) if root else "."

    def run(self, path: str = "", question: str = "", frames: int = 4) -> ToolResult:
        cfg = _vision_config()
        if not cfg["model"]:
            return ToolResult(ok=False, output="", error="no vision model set — pick a SMALL one in "
                              "Preferences → vision model")
        full = path if os.path.isabs(path) else os.path.join(self.root, path)
        if not os.path.exists(full):
            return ToolResult(ok=False, output="", error=f"video not found: {full}")
        from crucible.vision import describe_images, extract_frames
        try:
            imgs = extract_frames(full, int(frames))
            if not imgs:
                return ToolResult(ok=False, output="", error="could not extract frames (is ffmpeg installed?)")
            prompt = (f"These are {len(imgs)} still frames sampled in order from a video. "
                      + (question or "Describe what happens across the video."))
            out = describe_images(imgs, prompt, cfg["model"], num_ctx=cfg["num_ctx"], keep_alive=cfg["keep_alive"])
            return ToolResult(ok=True, output=f"[{len(imgs)} frames sampled]\n{out}")
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"watch_video failed: {e}")
