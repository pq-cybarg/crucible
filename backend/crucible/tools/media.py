from __future__ import annotations
from crucible.media import (comfyui_txt2img, media_endpoint, openai_image_request)
from crucible.tools.base import ToolResult

# Agent-facing media tools — so the forge (and /api/tools consumers) can generate images and
# transcribe/synthesize audio through whatever backend is configured (ComfyUI or an
# OpenAI-compatible media server). Network paths; the request builders are unit-tested.


class GenerateImage:
    name = "generate_image"
    description = "Generate an image from a text prompt via the configured image backend (ComfyUI or OpenAI-compatible /v1/images)."
    parameters = {"type": "object", "properties": {
        "prompt": {"type": "string"}, "size": {"type": "string", "description": "e.g. 512x512"}},
        "required": ["prompt"]}

    def __init__(self, root=None, timeout: float = 180.0):
        self.timeout = timeout

    def run(self, prompt: str, size: str = "512x512") -> ToolResult:
        endpoint = media_endpoint("image")
        if not endpoint:
            return ToolResult(ok=False, output="",
                              error="no image backend — set CRUCIBLE_IMAGE_ENDPOINT (ComfyUI :8188 or an OpenAI-images server)")
        try:
            import httpx
            if "8188" in endpoint or endpoint.endswith("/prompt"):    # ComfyUI
                url = endpoint if endpoint.endswith("/prompt") else endpoint + "/prompt"
                r = httpx.post(url, json=comfyui_txt2img(prompt), timeout=self.timeout)
                r.raise_for_status()
                return ToolResult(ok=True, output=f"queued ComfyUI job: {r.text[:200]}")
            url = endpoint + "/v1/images/generations"
            r = httpx.post(url, json=openai_image_request(prompt, size), timeout=self.timeout)
            r.raise_for_status()
            body = r.json()
            first = (body.get("data") or [{}])[0]
            return ToolResult(ok=True, output=first.get("url") or first.get("b64_json", "")[:120] or r.text[:200])
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"image generation failed: {e}")


class Transcribe:
    name = "transcribe_audio"
    description = "Transcribe an audio file (path or URL) to text via the configured STT backend."
    parameters = {"type": "object", "properties": {"audio": {"type": "string"}}, "required": ["audio"]}

    def __init__(self, root=None, timeout: float = 180.0):
        self.timeout = timeout

    def run(self, audio: str) -> ToolResult:
        endpoint = media_endpoint("stt")
        if not endpoint:
            return ToolResult(ok=False, output="", error="no STT backend — set CRUCIBLE_STT_ENDPOINT")
        try:
            import httpx
            r = httpx.post(endpoint + "/v1/audio/transcriptions",
                           json={"file": audio, "model": "whisper"}, timeout=self.timeout)
            r.raise_for_status()
            body = r.json()
            return ToolResult(ok=True, output=body.get("text", r.text[:400]))
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"transcription failed: {e}")
