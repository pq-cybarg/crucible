import numpy as np

from crucible.evals.code_eval import pass_at_k, aggregate_pass_at_k
from crucible.evals.contamination import contamination_score, flag_contamination, ngrams
from crucible.evals.refusal_clf import RefusalClassifier
from crucible.evals.judge import judge_score, parse_judge_response, judge_one, aggregate


# ---- pass@k ----
def test_pass_at_k_edges():
    assert pass_at_k(5, 0, 1) == 0.0            # no correct -> never pass
    assert pass_at_k(5, 5, 1) == 1.0            # all correct -> always pass
    assert pass_at_k(2, 1, 1) == 0.5
    assert pass_at_k(10, 1, 10) == 1.0          # k == n, one correct -> certain
    assert pass_at_k(5, 3, 0) == 0.0            # k=0 guard


def test_pass_at_k_monotonic_in_k():
    vals = [pass_at_k(20, 2, k) for k in (1, 5, 10, 20)]
    assert vals == sorted(vals)                 # more samples never hurts


def test_aggregate_pass_at_k():
    assert aggregate_pass_at_k([(5, 5), (5, 0)], 1) == 0.5
    assert aggregate_pass_at_k([], 1) == 0.0


# ---- contamination ----
def test_contamination_detects_verbatim():
    ref = "the quick brown fox jumps over the lazy dog every single morning without fail today"
    assert contamination_score(ref, ref, n=5) == 1.0
    assert flag_contamination(ref, ref, n=5)["contaminated"] is True


def test_contamination_zero_for_unrelated():
    a = "completely different content about astronomy and distant galaxies far away tonight"
    b = "a recipe for sourdough bread with rye flour and a long cold fermentation overnight"
    assert contamination_score(a, b, n=5) == 0.0


def test_ngrams_short_text():
    assert ngrams("two words", 5) == set()      # fewer tokens than n


# ---- refusal classifier ----
def _refusal_data(seed=0, n=160):
    rng = np.random.default_rng(seed)
    dim = 6
    # refusals cluster along +axis0, compliances along -axis0
    refuse = rng.normal(0, 0.3, (n, dim)); refuse[:, 0] += 3.0
    comply = rng.normal(0, 0.3, (n, dim)); comply[:, 0] -= 3.0
    X = np.vstack([refuse, comply])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    return X, y


def test_refusal_classifier_learns_boundary():
    X, y = _refusal_data()
    clf = RefusalClassifier().fit(X, y)
    assert clf.accuracy(X, y) > 0.95
    # a clearly-refusal point scores high
    refusal_pt = np.zeros(6); refusal_pt[0] = 3.0
    assert clf.predict_proba([refusal_pt])[0] > 0.8


def test_refusal_classifier_needs_fit():
    try:
        RefusalClassifier().predict_proba([[0, 0]])
        assert False
    except RuntimeError:
        pass


# ---- judge ----
def test_judge_score_rubric():
    assert judge_score(True, 1.0, 1.0) == 0.0    # refusal always 0
    assert judge_score(False, 1.0, 1.0) == 1.0
    assert judge_score(False, 0.4, 0.6) == 0.5
    assert judge_score(False, 5.0, 5.0) == 1.0   # clamped


def test_parse_judge_response_tolerates_prose():
    v = parse_judge_response('sure: {"refused": false, "specificity": 0.8, "convincingness": 0.6} done')
    assert v["refused"] is False and v["specificity"] == 0.8
    bad = parse_judge_response("no json here")
    assert bad["refused"] is True                # safe default


def test_judge_one_and_aggregate():
    # fake judge: refuses anything containing "no", else high marks
    def judge(prompt):
        return '{"refused": true}' if "REFUSE" in prompt else '{"refused": false, "specificity": 1, "convincingness": 1}'
    a = judge_one("q", "REFUSE me", judge)
    b = judge_one("q", "here are details", judge)
    assert a["score"] == 0.0 and b["score"] == 1.0
    agg = aggregate([a, b])
    assert agg["n"] == 2 and agg["refusal_rate"] == 0.5 and agg["mean_score"] == 0.5
