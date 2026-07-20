"""#31 — the companion ANIMATION layer must stay decoupled from the CHAT / model-context layer, so
animating the avatar (many frames, moods, idle motion) can never feed the model's context window."""
import inspect

from crucible import animation, companion
from crucible.companion import CompanionDriver


def test_animation_modules_do_not_touch_the_chat_or_session_layer():
    # The animation/companion modules must not import the conversation/session/message machinery — that
    # coupling is exactly how animation state would leak into the context window.
    for mod in (animation, companion):
        src = inspect.getsource(mod)
        for forbidden in ("agent_sessions", "from crucible.agent ", "import crucible.agent",
                          "agent_react", "ChatMessage", "messages"):
            assert forbidden not in src, f"{mod.__name__} must not reference the chat layer ({forbidden!r})"


def test_companion_step_yields_render_frames_not_chat_messages():
    d = CompanionDriver()
    d.set_mood({"happy": 1.0})
    frame = d.step()
    assert isinstance(frame, dict)
    # a render frame carries animation params, NOT chat message fields
    assert "role" not in frame and "content" not in frame
    assert "messages" not in frame


def test_companion_holds_no_conversation_state():
    d = CompanionDriver()
    # its entire state is animation (mood weights, talk/speech, idle timers) — no messages/history/session
    for attr in vars(d):
        assert not any(k in attr.lower() for k in ("message", "history", "session", "convo", "context"))
