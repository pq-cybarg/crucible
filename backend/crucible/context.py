from __future__ import annotations
# Context management. A long agent conversation eventually overflows the model's context window.
# Compaction summarizes the OLD turns into a compact synopsis and keeps the recent turns verbatim,
# plus any leading system prompt — so the thread continues without losing the plot. Token counting
# here is an HONEST HEURISTIC (chars/4, labeled as such — not a real tokenizer; it's for budgeting
# and the UI meter). The summarizer is injected, so the compaction policy is pure and unit-tested.
from typing import Callable

# Rough English average ~4 chars/token. A heuristic for budgeting, NOT a real tokenizer.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text_or_messages) -> int:
    """Heuristic token estimate (chars/4). NOT a real tokenizer — for budgeting + the UI meter."""
    if isinstance(text_or_messages, str):
        return max(0, len(text_or_messages) // _CHARS_PER_TOKEN)
    total = 0
    for m in text_or_messages or []:
        total += len(str(m.get("content", "") or "")) // _CHARS_PER_TOKEN
    return total


def _split(messages: list[dict], keep_recent: int):
    """Leading system prompt(s), the old middle to summarize, and the last keep_recent turns."""
    system = [m for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") != "system"]
    # keep_recent >= len(convo) -> keep everything (nothing old); keep_recent <= 0 -> keep nothing
    recent = convo[-keep_recent:] if keep_recent > 0 else []
    old = convo[: len(convo) - len(recent)]
    return system, old, recent


def needs_compaction(messages: list[dict], max_tokens: int, keep_recent: int = 6) -> bool:
    """True when the estimated size is over budget AND there are old turns to summarize."""
    non_system = [m for m in messages if m.get("role") != "system"]
    return estimate_tokens(messages) > max_tokens and len(non_system) > keep_recent


def render_transcript(messages: list[dict]) -> str:
    """Flatten turns into a plain transcript the summarizer can read."""
    return "\n".join(f"{m.get('role', '?')}: {m.get('content', '') or ''}" for m in messages)


def _passthrough(messages: list[dict]) -> dict:
    tok = estimate_tokens(messages)
    return {"messages": messages, "summary": None, "compacted": False,
            "stats": {"before_tokens": tok, "after_tokens": tok, "summarized_turns": 0,
                      "token_estimate": "heuristic (chars/4), not a tokenizer"}}


def compact(messages: list[dict], summarizer: Callable[[str], str],
            max_tokens: int = 4000, keep_recent: int = 6) -> dict:
    """Summarize the old middle turns via summarizer(text)->str; keep the system prompt + the last
    keep_recent turns verbatim. Returns {messages, summary, compacted, stats}. No old turns ->
    a no-op passthrough. The summary is inserted as a system message so it steers the next turn."""
    system, old, recent = _split(messages, keep_recent)
    if not old:
        return _passthrough(messages)
    summary_text = str(summarizer(render_transcript(old)) or "").strip()
    if not summary_text:
        return _passthrough(messages)
    summary_msg = {"role": "system",
                   "content": "Summary of earlier conversation (older turns were compacted to save "
                              "context; key facts, decisions, and open threads below):\n" + summary_text}
    new_messages = system + [summary_msg] + recent
    return {"messages": new_messages, "summary": summary_text, "compacted": True,
            "stats": {"before_tokens": estimate_tokens(messages),
                      "after_tokens": estimate_tokens(new_messages),
                      "summarized_turns": len(old),
                      "token_estimate": "heuristic (chars/4), not a tokenizer"}}


def maybe_compact(messages: list[dict], summarizer: Callable[[str], str],
                  max_tokens: int = 4000, keep_recent: int = 6) -> dict:
    """Compact only if over the token budget; otherwise pass the messages through untouched."""
    if not needs_compaction(messages, max_tokens, keep_recent):
        return _passthrough(messages)
    return compact(messages, summarizer, max_tokens, keep_recent)


SUMMARIZE_INSTRUCTION = (
    "Summarize the conversation so far so it can CONTINUE without losing the plot. Preserve: the "
    "user's goal and constraints, decisions made, facts established, files/values referenced, and "
    "any open threads or next steps. Be concise and factual — no preamble. Transcript:\n\n"
)
