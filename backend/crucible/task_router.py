from __future__ import annotations
# Task-aware model routing (mixture-of-models). Classify the request, then route it to the
# model best suited to it — a coding model for code, a reasoning model for analysis, a small
# fast model for chat — or honor the user's chosen quality LEVEL (fast / balanced / max).
# This turns the registry into a mixture of experts: the right model per task instead of one
# model for everything. Pure heuristics here (deterministic + unit-tested); a model-based
# classifier can replace classify_task later without changing the routing.
from typing import Callable, Optional

CATEGORIES = ["code", "math", "reasoning", "creative", "extract", "tools", "vision", "chat"]

_SIGNALS: dict[str, list[str]] = {
    "code": ["def ", "class ", "import ", "```", "function", "bug", "refactor", "compile",
             "stack trace", "python", "javascript", "typescript", "rust", "regex", "api"],
    "math": ["calculate", "solve", "equation", "derivative", "integral", "probability",
             "theorem", "prove", "sum of", "factorial", "matrix", "="],
    "reasoning": ["why", "explain", "reason", "step by step", "analyze", "compare",
                  "trade-off", "pros and cons", "implications", "root cause"],
    "creative": ["write a story", "poem", "creative", "imagine", "fiction", "song",
                 "screenplay", "brainstorm", "lyrics"],
    "extract": ["extract", "summarize", "summary", "list", "json", "parse", "classify",
                "label", "table", "translate"],
    "tools": ["search the web", "fetch", "run this", "execute", "read the file",
              "list the", "browse", "grep", "shell"],
    "vision": ["image", "picture", "photo", "screenshot", "diagram", "chart", "this png"],
}

LEVELS = {"fast": 0, "balanced": 1, "max": 2}


def classify_task(prompt: str) -> dict:
    """Classify a prompt into a task category with a confidence. Defaults to 'chat'."""
    text = (prompt or "").lower()
    scores = {cat: sum(text.count(s) for s in sigs) for cat, sigs in _SIGNALS.items()}
    best = max(scores, key=lambda c: scores[c]) if scores else "chat"
    total = sum(scores.values())
    if total == 0:
        return {"type": "chat", "confidence": 0.0, "scores": scores}
    return {"type": best, "confidence": round(scores[best] / total, 3), "scores": scores}


def infer_tags(name: str, quant: str = "") -> tuple[list[str], int]:
    """Best-effort capability tags + a quality tier (0 fast .. 2 max) from a model's name and
    quant — so routing works even without hand-labeled models."""
    n = (name or "").lower()
    q = (quant or "").lower()
    tags: list[str] = []
    if any(k in n for k in ("coder", "code", "deepseek-coder", "starcoder", "codellama")):
        tags.append("code")
    if any(k in n for k in ("math", "wizardmath", "mathstral")):
        tags.append("math")
    if any(k in n for k in ("reason", "r1", "o1", "thinking", "qwq")):
        tags.append("reasoning")
    if any(k in n for k in ("vision", "llava", "-vl", "vl-", "pixtral")):
        tags.append("vision")
    if not tags:
        tags.append("chat")
    # tier from parameter-size hints + quant
    tier = 1
    if any(k in n for k in ("70b", "72b", "34b", "32b", "27b", "20b", "8x7b", "mixtral")):
        tier = 2
    elif any(k in n for k in ("0.5b", "1b", "1.1b", "1.5b", "2b", "3b", "tiny", "mini")):
        tier = 0
    if any(k in q for k in ("q2", "q3", "iq1", "iq2", "1.58")):
        tier = max(0, tier - 1)     # heavy quant knocks it down a tier
    return tags, tier


def route_by_task(task_type: str, models: list[dict], user_level: str = "balanced",
                  is_available: Optional[Callable[[str], bool]] = None) -> Optional[str]:
    """Pick the model best matching the task, biased toward the user's quality level.
    models: [{id, tags:[...], tier:int}]. Returns a model id, or None if none available."""
    avail = is_available or (lambda _i: True)
    cand = [m for m in models if avail(m["id"])]
    if not cand:
        return None
    level = LEVELS.get(user_level, 1)

    def score(m: dict) -> tuple:
        tag_match = 1 if task_type in (m.get("tags") or []) else 0
        tier = int(m.get("tier", 1))
        return (tag_match, -abs(tier - level), tier)   # tag first, then closeness to level

    return max(cand, key=score)["id"]


def route(prompt: str, models: list[dict], user_level: str = "balanced",
          is_available: Optional[Callable[[str], bool]] = None) -> dict:
    """End-to-end: classify the prompt and route it. Returns the decision + why."""
    task = classify_task(prompt)
    chosen = route_by_task(task["type"], models, user_level, is_available)
    return {"task": task["type"], "confidence": task["confidence"],
            "level": user_level, "chosen": chosen,
            "why": (f"{task['type']} task -> '{chosen}'" if chosen else "no model available")}
