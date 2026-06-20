from crucible.evals.scoring import expected_calibration_error, extract_choice, mc_accuracy


def test_extract_choice():
    assert extract_choice("The answer is C.") == "C"
    assert extract_choice("(B)") == "B"
    assert extract_choice("I am not sure") is None


def test_mc_accuracy():
    assert mc_accuracy(["A", "B", "C"], ["A", "B", "D"]) == 2 / 3


def test_ece_low_when_calibrated():
    conf = [0.95, 0.9, 0.05, 0.1]
    correct = [True, True, False, False]
    assert expected_calibration_error(conf, correct) < 0.15


def test_ece_high_when_overconfident():
    conf = [0.99, 0.99, 0.99, 0.99]
    correct = [False, False, False, True]
    assert expected_calibration_error(conf, correct) > 0.5
