from __future__ import annotations
# Local text→image generation for avatar parts (and general image gen) via diffusers on Apple Metal
# (MPS). Small anime SD 1.5 by default so it's fast + light (~2 GB, far under the LLMs that freeze the
# machine). Memory-conscious: lazy-load, attention/vae slicing, and UNLOAD after use to free RAM. The
# safety checker is OFF (this is an uncensoring workbench) so it never blurs/blocks legit anime output.
import os
from typing import Optional

# OPSEC: no phoning home. Disable HuggingFace/transformers/diffusers telemetry BEFORE they import, and
# don't send an implicit auth token. Models are cached LOCALLY and, once cached, loaded offline (no HF
# round-trip at all). Set before any hf import so it takes effect.
for _k in ("HF_HUB_DISABLE_TELEMETRY", "DISABLE_TELEMETRY", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
           "TRANSFORMERS_NO_ADVISORY_WARNINGS"):
    os.environ.setdefault(_k, "1")

_PIPE = None
_MODEL: Optional[str] = None

# anime SD 1.5 (~2 GB, light + safe on the RAM budget). admruul/anything-v3.0 is a NON-GATED,
# diffusers-format mirror of Anything-v3.0. Override with CRUCIBLE_SD_MODEL for other checkpoints.
DEFAULT_MODEL = os.environ.get("CRUCIBLE_SD_MODEL", "admruul/anything-v3.0")


def _device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipe(model: Optional[str] = None):
    """Lazy-load the diffusers text2img pipeline (downloads the model on first use). Cached across calls."""
    global _PIPE, _MODEL
    import torch
    from diffusers import StableDiffusionPipeline

    m = model or DEFAULT_MODEL
    if _PIPE is not None and _MODEL == m:
        return _PIPE
    dev = _device()
    # fp16 is only stable on CUDA; on Apple MPS it produces NaN/black images, so use fp32 there (a bit
    # slower, but correct). Memory is still fine (~4 GB for SD 1.5 fp32).
    dtype = torch.float16 if dev == "cuda" else torch.float32
    # LOCAL-FIRST: try the cached copy with no network at all; only if it isn't cached do we download
    # once (then it's local forever). Keeps normal operation fully offline.
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            m, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False, local_files_only=True)
    except Exception:
        pipe = StableDiffusionPipeline.from_pretrained(
            m, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()                  # keep peak memory low
        pipe.enable_vae_slicing()
    except Exception:
        pass
    _PIPE, _MODEL = pipe, m
    return pipe


def generate(prompt: str, negative: str = "", size: tuple = (384, 384), steps: int = 24,
             guidance: float = 7.0, seed: Optional[int] = None):
    """Generate one image → PIL. Small default size for avatar parts (fast + low memory)."""
    import torch

    pipe = load_pipe()
    w, h = int(size[0]) - int(size[0]) % 8, int(size[1]) - int(size[1]) % 8   # SD needs multiples of 8
    gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
    return pipe(prompt, negative_prompt=(negative or None), width=w, height=h,
                num_inference_steps=int(steps), guidance_scale=float(guidance), generator=gen).images[0]


def unload() -> None:
    """Drop the pipeline and free memory (called after generating so the model doesn't linger)."""
    global _PIPE, _MODEL
    _PIPE, _MODEL = None, None
    import gc
    import torch
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def available() -> bool:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
