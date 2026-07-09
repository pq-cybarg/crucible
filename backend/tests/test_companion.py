"""Real-time companion drive loop — smoothed mood crossfade + idle + lip-sync → engine-agnostic frames."""
from crucible.companion import CompanionDriver


def test_mood_eases_toward_target_not_a_jumpcut():
    d = CompanionDriver(seed=1, ease=0.25)
    d.set_mood({"happy": 1.0})
    first = d.step()
    # after one tick the mood has only PARTLY moved from neutral → happy (smooth crossfade)
    assert 0.0 < d.mood().get("happy", 0.0) < 1.0
    # and it keeps approaching over subsequent ticks
    for _ in range(40):
        d.step()
    assert d.mood().get("happy", 0.0) > 0.9
    assert first["params"]["smile"] >= 0.0


def test_react_maps_reaction_word_to_expression_mood():
    d = CompanionDriver(seed=1)
    d.react("funny")                       # funny → laughing
    for _ in range(50):
        d.step()
    assert d.mood().get("laughing", 0.0) > 0.8


def test_talking_opens_the_mouth_and_stops_when_off():
    d = CompanionDriver(seed=1)
    d.set_mood({"neutral": 1.0})
    for _ in range(10):
        d.step()
    d.set_talking(True)
    opens = [d.step()["params"]["mouth_open"] for _ in range(6)]
    assert max(opens) > 0.3                 # the mouth flaps while talking
    d.set_talking(False)
    closed = [d.step()["params"]["mouth_open"] for _ in range(6)]
    assert max(closed) < 0.2                 # settles shut when talk stops


def test_speech_level_drives_lip_sync_amplitude():
    d = CompanionDriver(seed=1)
    d.set_speech_level(0.9)
    f = d.step()
    assert f["params"]["mouth_open"] > 0.6   # explicit TTS amplitude → wide mouth (true lip-sync)


def test_frames_are_full_rig_frames_with_idle_gaze():
    d = CompanionDriver(seed=3)
    frames = list(d.frames(120))
    assert all({"params", "arkit", "live2d", "vrm", "gaze"} <= set(f) for f in frames)
    xs = [f["gaze"][0] for f in frames]
    assert max(xs) - min(xs) > 0.15          # saccadic gaze roves live
    assert any(f["blink"] == 1.0 for f in frames)   # blinks happen


def test_is_deterministic_for_a_seed():
    def run():
        d = CompanionDriver(seed=9)
        d.set_mood({"happy": 0.6, "surprised": 0.4})
        return [f["gaze"] for f in d.frames(60)]
    assert run() == run()


def test_breath_modulates_head_tilt_subtly():
    d = CompanionDriver(seed=1, breath_depth=0.05)
    tilts = [d.step()["params"]["head_tilt"] for _ in range(90)]
    assert max(tilts) > 0 and min(tilts) < 0       # a gentle oscillation
    assert max(abs(t) for t in tilts) < 0.12       # but subtle, not a head-bang
