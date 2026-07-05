"""Context compaction: summarize old turns, keep the system prompt + recent turns verbatim.
Pure policy — the summarizer is injected, so behavior is exercised without a live model."""
from crucible.context import (compact, estimate_tokens, maybe_compact, needs_compaction,
                              render_transcript)


def _convo(n_pairs: int, sys: bool = True) -> list[dict]:
    msgs = [{"role": "system", "content": "you are helpful"}] if sys else []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"question {i} " + "x" * 40})
        msgs.append({"role": "assistant", "content": f"answer {i} " + "y" * 40})
    return msgs


def _fake_summarizer(text: str) -> str:
    return f"[summary of {text.count(chr(10)) + 1} lines]"


def test_estimate_tokens_heuristic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 40) == 10          # chars/4
    assert estimate_tokens([{"content": "a" * 40}, {"content": "b" * 40}]) == 20


def test_needs_compaction_only_when_over_budget_with_old_turns():
    msgs = _convo(10)
    assert needs_compaction(msgs, max_tokens=1, keep_recent=2) is True
    assert needs_compaction(msgs, max_tokens=100000, keep_recent=2) is False
    # not enough turns to bother
    assert needs_compaction(_convo(1), max_tokens=1, keep_recent=6) is False


def test_compact_keeps_system_and_recent_and_summarizes_old():
    msgs = _convo(10)                       # 1 system + 20 turns
    out = compact(msgs, _fake_summarizer, keep_recent=4)
    assert out["compacted"] is True
    new = out["messages"]
    # system prompt preserved, a summary system message inserted, last 4 turns verbatim
    assert new[0]["content"] == "you are helpful"
    assert new[1]["role"] == "system" and "Summary of earlier conversation" in new[1]["content"]
    assert new[-4:] == msgs[-4:]
    # fewer messages than before, and stats report the reduction
    assert len(new) < len(msgs)
    assert out["stats"]["summarized_turns"] == 16      # 20 - 4 recent
    assert out["stats"]["after_tokens"] < out["stats"]["before_tokens"]


def test_compact_noop_when_no_old_turns():
    msgs = _convo(2)                        # 4 turns, keep_recent 6 -> nothing old
    out = compact(msgs, _fake_summarizer, keep_recent=6)
    assert out["compacted"] is False and out["messages"] == msgs


def test_compact_noop_when_summarizer_returns_empty():
    out = compact(_convo(10), lambda t: "   ", keep_recent=2)
    assert out["compacted"] is False


def test_maybe_compact_passthrough_under_budget():
    msgs = _convo(3)
    out = maybe_compact(msgs, _fake_summarizer, max_tokens=100000)
    assert out["compacted"] is False and out["messages"] == msgs
    assert "heuristic" in out["stats"]["token_estimate"]


def test_maybe_compact_triggers_over_budget():
    msgs = _convo(20)
    out = maybe_compact(msgs, _fake_summarizer, max_tokens=1, keep_recent=4)
    assert out["compacted"] is True and len(out["messages"]) < len(msgs)


def test_render_transcript_roundtrips_roles():
    t = render_transcript([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    assert t == "user: hi\nassistant: yo"
