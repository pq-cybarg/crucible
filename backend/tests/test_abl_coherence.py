from crucible.abliteration.coherence import coherence_score


def test_coherent_text_scores_high():
    assert coherence_score("Sure, here is a clear and helpful explanation of the topic.") > 0.7


def test_gibberish_scores_low():
    assert coherence_score("bushballs（boundary（busballs（bonding（bussed（") < 0.3


def test_repetitive_scores_lower():
    assert coherence_score("the the the the the the the") < coherence_score("a clear varied sentence here")


def test_empty():
    assert coherence_score("") == 0.0
