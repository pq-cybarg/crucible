from __future__ import annotations
# Multimodal I/O. Crucible's provider is OpenAI-compatible, so it can broker the sibling
# modalities too: images (ComfyUI or OpenAI /v1/images), speech-to-text, text-to-speech, and
# embeddings — each routed to a configured backend. These build the requests (pure, tested);
# the actual proxying is a network path. Backends are set by env so any local/remote service
# plugs in.
import os

_ENV = {"image": "CRUCIBLE_IMAGE_ENDPOINT", "stt": "CRUCIBLE_STT_ENDPOINT",
        "tts": "CRUCIBLE_TTS_ENDPOINT", "embed": "CRUCIBLE_EMBED_ENDPOINT"}


def media_endpoint(kind: str) -> str:
    """Resolve a media backend endpoint (image/stt/tts/embed) from the environment."""
    return os.environ.get(_ENV.get(kind, ""), "").rstrip("/")


def comfyui_txt2img(prompt: str, ckpt: str = "model.safetensors", steps: int = 20,
                    width: int = 512, height: int = 512, seed: int = 0,
                    negative: str = "") -> dict:
    """A minimal, valid ComfyUI /prompt workflow for text-to-image. Returns the POST body."""
    graph = {
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 7.0,
              "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
              "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "crucible", "images": ["8", 0]}},
    }
    return {"prompt": graph}


def openai_image_request(prompt: str, size: str = "512x512", n: int = 1) -> dict:
    return {"prompt": prompt, "size": size, "n": max(1, int(n))}


def embeddings_request(inputs, model: str = "local") -> dict:
    return {"model": model, "input": inputs if isinstance(inputs, list) else [inputs]}


def is_multimodal_content(messages: list) -> bool:
    """True if any message carries non-text content parts (image_url / input_audio / video),
    i.e. the request needs a multimodal backing model rather than a text-only one."""
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") not in ("text", None):
                    return True
    return False


def content_modalities(messages: list) -> list[str]:
    """The distinct content modalities present across the messages (text/image/audio/video)."""
    mods: set[str] = set()
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            mods.add("text")
        elif isinstance(c, list):
            for part in c:
                if not isinstance(part, dict):
                    continue
                t = part.get("type", "text")
                if t in ("text",):
                    mods.add("text")
                elif "image" in t:
                    mods.add("image")
                elif "audio" in t or "input_audio" in t:
                    mods.add("audio")
                elif "video" in t:
                    mods.add("video")
    return sorted(mods)
