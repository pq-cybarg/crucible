"""Idle life for the companion face — deterministic saccades, natural blinks, micro-expression flicker."""
from crucible.animation import IdleAnimator, MICRO_ACCENTS


def _run(seed=7, n=200):
    a = IdleAnimator(seed=seed)
    return [a.step() for _ in range(n)]


def test_is_deterministic_for_a_seed():
    a = [(s.gaze, s.blink, tuple(sorted(s.micro.items()))) for s in _run(seed=3)]
    b = [(s.gaze, s.blink, tuple(sorted(s.micro.items()))) for s in _run(seed=3)]
    assert a == b                                            # same seed → identical sequence (testable)


def test_different_seeds_diverge():
    a = [s.gaze for s in _run(seed=1)]
    b = [s.gaze for s in _run(seed=2)]
    assert a != b


def test_gaze_stays_in_range_and_actually_moves():
    states = _run()
    xs = [s.gaze[0] for s in states]
    ys = [s.gaze[1] for s in states]
    assert all(-1.0 <= x <= 1.0 for x in xs) and all(-1.0 <= y <= 1.0 for y in ys)
    assert max(xs) - min(xs) > 0.2                          # the eyes rove (saccades happen)
    assert any(abs(x) > 0.05 for x in xs)                   # not stuck dead-centre


def test_blinks_occur_at_a_natural_irregular_cadence():
    states = _run(n=300)
    blink_ticks = [i for i, s in enumerate(states) if s.blink]
    assert len(blink_ticks) >= 8                            # blinks happen repeatedly over ~75s
    gaps = [j - i for i, j in zip(blink_ticks, blink_ticks[1:])]
    assert len(set(gaps)) > 1                               # not a fixed metronome — varied gaps


def test_micro_expressions_flicker_briefly_and_are_gentle():
    states = _run(n=300)
    micro = [s.micro for s in states if s.micro]
    assert micro                                            # some ticks carry a micro-expression
    for m in micro:
        (name, weight), = m.items()
        assert name in MICRO_ACCENTS
        assert 0.0 < weight <= 0.2                          # faint — never overrides the real mood
