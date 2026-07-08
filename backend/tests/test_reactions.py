"""Reaction vocabulary + semantic-reaction parsing — reactions are richer than jumpscares."""
from crucible.reactions import REACTIONS, is_meaningful, parse_reaction


def test_parse_reaction_line():
    desc, r = parse_reaction("A dog trips over a hose and tumbles.\nReaction: funny")
    assert r == "funny" and "dog trips" in desc and "Reaction" not in desc


def test_parse_reaction_falls_back_to_known_word():
    _, r = parse_reaction("This is a very tense standoff between two gunmen.")
    assert r == "tense"


def test_parse_reaction_accepts_novel_word():
    # the model isn't limited to the fixed set — a Reaction: line wins even for an unlisted word
    _, r = parse_reaction("Someone eats a lemon.\nReaction: sour")
    assert r == "sour"


def test_neutral_when_no_reaction():
    _, r = parse_reaction("A person stands in a room.")
    assert r == "neutral" and not is_meaningful(r)
    assert is_meaningful("funny") and "funny" in REACTIONS


def test_stream_emits_semantic_reactions(tmp_path):
    import shutil
    if not shutil.which("ffmpeg"):
        return
    import subprocess
    from crucible.cowatch import stream_commentary
    vid = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=5:duration=4",
                    str(vid)], capture_output=True, timeout=60)

    # a model that returns a description + a funny reaction
    def describe(frame, prompt):
        return "A cat does something silly.\nReaction: funny"

    items = list(stream_commentary(str(vid), describe, interval=2.0, semantic_reactions=True, pace=False))
    comments = [i for i in items if i["kind"] == "commentary"]
    sem = [i for i in items if i["kind"] == "reaction" and i.get("source") == "semantic"]
    assert comments and comments[0]["reaction"] == "funny" and "cat" in comments[0]["text"]
    assert sem and sem[0]["type"] == "funny"        # a semantic reaction event was emitted
