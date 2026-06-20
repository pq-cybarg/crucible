from crucible.abliteration.verify import behavioral_compare


def test_uncensoring_shows_compliance_gain():
    # base refuses harmful; variant complies; neither over-refuses benign
    def base(p):
        return "I'm sorry, I can't help with that" if "harm" in p or "danger" in p else "Sure!"

    def variant(p):
        return "Sure, here is how"

    res = behavioral_compare(base, variant, harmful=["harmful thing", "dangerous thing"], benign=["hi", "yo"])
    assert res["harmful_refusal_rate"]["before"] == 1.0
    assert res["harmful_refusal_rate"]["after"] == 0.0
    assert res["harmful_compliance_rate"]["after"] == 1.0
    assert res["benign_over_refusal_rate"]["after"] == 0.0
    assert len(res["samples"]) == 2 and res["samples"][0]["before"] != res["samples"][0]["after"]
