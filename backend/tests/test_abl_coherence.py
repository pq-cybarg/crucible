from crucible.abliteration.coherence import coherence_score, degeneration_guard


def test_clean_english_scores_high():
    assert degeneration_guard("The quick brown fox jumps over the lazy dog.") > 0.8


def test_repetition_scores_low():
    assert degeneration_guard("the the the the the the the the") < 0.3


def test_concatenated_gibberish_penalized():
    assert degeneration_guard("asdkfjaslkdjfhaslkdjfhalskdjfhalskdjfh") < 0.5   # no word breaks


def test_control_char_garbage_penalized():
    assert degeneration_guard("hello \x00\x01\x02\x03 �� world") < 0.9


def test_non_english_not_penalized():
    # the OLD ascii-ratio version scored valid CJK/Cyrillic near zero — the guard must not
    cjk = degeneration_guard("这是一个完全正常的中文句子，用来测试。")
    cyr = degeneration_guard("Это совершенно нормальное русское предложение для теста.")
    assert cjk > 0.7 and cyr > 0.7


def test_empty():
    assert degeneration_guard("") == 0.0


def test_alias_preserved():
    assert coherence_score is degeneration_guard
