from crucible.evals.throughput import estimate_tokens, summarize_benchmark, throughput


def test_throughput_basic():
    assert throughput(100, 2.0) == 50.0
    assert throughput(0, 2.0) == 0.0
    assert throughput(100, 0.0) == 0.0
    assert throughput(100, -1) == 0.0


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("one two three four five") >= 5


def test_summarize_benchmark_rates():
    s = summarize_benchmark(prompt_tokens=20, gen_tokens=40, prefill_s=0.1, decode_s=2.0)
    assert s["prefill_tok_per_s"] == 200.0       # 20 / 0.1
    assert s["decode_tok_per_s"] == 20.0         # 40 / 2.0
    assert s["tok_per_s"] == round(40 / 2.1, 2)  # overall
    assert s["total_s"] == 2.1
