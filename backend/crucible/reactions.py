from __future__ import annotations
# The reaction vocabulary — reactions are far richer than jumpscares. Two sources feed the co-watch
# reaction stream:
#   SIGNAL reactions  (crucible.detect): cheap, non-LLM, exact-timing — scene_cut / jumpscare / loud.
#   SEMANTIC reactions (this module):    the vision model's read of a scene's emotional/content nature —
#                                        funny, tense, sad, cute, exciting, beautiful, action, …
# The vocabulary is a starting set, not a cage: `parse_reaction` also accepts any single word the model
# offers, so the model isn't limited to a fixed list. Downstream (VTuber rig, overlays) key off `type`.
import re

# emoji is a display hint only; the set is extensible — add freely, or let the model coin new ones.
REACTIONS: dict[str, str] = {
    "funny": "😂", "tense": "😬", "scary": "😱", "sad": "😢", "exciting": "🤩", "cute": "🥰",
    "beautiful": "😍", "gross": "🤢", "shocking": "😲", "surprising": "😮", "wholesome": "🥹",
    "awkward": "😅", "epic": "🔥", "calm": "😌", "romantic": "❤️", "action": "💥", "boring": "🥱",
    "confusing": "🤔", "sus": "🤨", "cringe": "😖", "wtf": "🫥", "dialogue": "💬", "neutral": "",
}

_WORD = re.compile(r"[a-z]+")

# words that carry no reaction — don't emit an event for these
_SKIP = {"neutral", "none", "nothing", "normal", "unknown", "na"}


def reaction_prompt(question: str = "") -> str:
    """Ask the vision model for a one-sentence description AND a one-word reaction. Kept open — the model
    may use a word outside the suggested set if it fits better."""
    focus = f" {question.strip()}" if question else ""
    return ("Look at this video frame.%s In ONE sentence, say what is happening. Then on a new line write "
            "'Reaction: <one word for the emotional tone — e.g. funny, tense, scary, sad, exciting, cute, "
            "beautiful, action, calm, surprising — or your own word>'." % focus)


def parse_reaction(text: str) -> tuple[str, str]:
    """Split a model reply into (description, reaction_word). Finds a 'Reaction: <word>' line; falls back
    to matching any known reaction word in the text. Returns ('', 'neutral') reaction when none is found."""
    desc = text.strip()
    reaction = ""
    m = re.search(r"reaction\s*[:\-]\s*([A-Za-z]+)", text, re.IGNORECASE)
    if m:
        reaction = m.group(1).lower()
        desc = text[: m.start()].strip() or desc
    else:
        for w in _WORD.findall(text.lower()):
            if w in REACTIONS and w not in _SKIP:
                reaction = w
                break
    reaction = reaction or "neutral"
    return desc, reaction


def is_meaningful(reaction: str) -> bool:
    return bool(reaction) and reaction.lower() not in _SKIP


def icon_for(reaction: str) -> str:
    return REACTIONS.get(reaction.lower(), "❗")
