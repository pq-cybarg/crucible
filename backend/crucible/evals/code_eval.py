from __future__ import annotations
# Unbiased pass@k for code/agentic benchmarks (HumanEval/MBPP style). Sampling k of n
# generations and asking "did at least one pass?" is biased if you just take the max; the
# Chen et al. (2021) estimator corrects it. Pure arithmetic — the sandboxed execution that
# produces (n, c) lives behind a guarded runner and is not unit-tested.


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k samples (drawn without replacement from n total,
    c of them correct) is correct. Numerically stable product form."""
    if k <= 0 or n <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def aggregate_pass_at_k(per_task: list[tuple[int, int]], k: int) -> float:
    """Mean pass@k over tasks, each given as (n_samples, n_correct)."""
    if not per_task:
        return 0.0
    return sum(pass_at_k(n, c, k) for n, c in per_task) / len(per_task)
