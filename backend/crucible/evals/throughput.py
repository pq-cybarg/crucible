from __future__ import annotations
# Throughput metrics. Two questions matter: "how fast will this model go?" (a pre-flight
# speed test before you commit to it) and "how fast is it going right now?" (live tok/s as
# tokens stream). Both reduce to tokens / elapsed — pure arithmetic, unit-tested; the timing
# of an actual generation is a model path.


def throughput(tokens: int, elapsed_s: float) -> float:
    """Tokens per second. 0 when no time has elapsed or no tokens were produced."""
    if elapsed_s <= 0 or tokens <= 0:
        return 0.0
    return float(tokens) / float(elapsed_s)


def estimate_tokens(text: str) -> int:
    """Rough token count when an exact tokenizer count isn't available (~1.3 tokens/word)."""
    words = len(text.split())
    return max(1, round(words * 1.3))


def summarize_benchmark(prompt_tokens: int, gen_tokens: int, prefill_s: float,
                        decode_s: float) -> dict:
    """A speed-test summary: prefill (prompt) rate, decode (generation) rate, and overall."""
    total_s = prefill_s + decode_s
    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "prefill_s": round(prefill_s, 4),
        "decode_s": round(decode_s, 4),
        "total_s": round(total_s, 4),
        "prefill_tok_per_s": round(throughput(prompt_tokens, prefill_s), 2),
        "decode_tok_per_s": round(throughput(gen_tokens, decode_s), 2),
        "tok_per_s": round(throughput(gen_tokens, total_s), 2),
    }
