"""VTube Studio bridge — pure request builders + the auth/create/stream loop over a fake websocket."""
import asyncio
import json

from crucible.rigmap import VTS_PARAMS, rig_frame
from crucible.vtsbridge import PARAM_SPECS, VTSBridge, idle_frames


class FakeWS:
    """A stand-in VTube Studio socket: answers requests by messageType, records everything sent."""

    def __init__(self, grant_token="TKN", authenticate=True, first_auth_fails=False):
        self.grant_token = grant_token
        self.authenticate = authenticate
        self.first_auth_fails = first_auth_fails
        self._auth_calls = 0
        self.sent: list[dict] = []
        self._out: list[str] = []

    async def send(self, raw):
        msg = json.loads(raw)
        self.sent.append(msg)
        mt = msg["messageType"]
        if mt == "AuthenticationTokenRequest":
            self._out.append(json.dumps({"messageType": "AuthenticationTokenResponse",
                                         "data": {"authenticationToken": self.grant_token}}))
        elif mt == "AuthenticationRequest":
            self._auth_calls += 1
            ok = self.authenticate and not (self.first_auth_fails and self._auth_calls == 1)
            self._out.append(json.dumps({"messageType": "AuthenticationResponse",
                                         "data": {"authenticated": ok}}))
        elif mt == "ParameterCreationRequest":
            self._out.append(json.dumps({"messageType": "ParameterCreationResponse", "data": {}}))
        # InjectParameterDataRequest expects no reply (send-only)

    async def recv(self):
        return self._out.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def of_type(self, mt):
        return [m for m in self.sent if m["messageType"] == mt]


def _connect_factory(ws):
    def connect(url):
        ws.url = url
        return ws
    return connect


# --- pure builders -----------------------------------------------------------------------------------

def test_envelopes_are_well_formed_vts_messages():
    b = VTSBridge()
    tok = b.token_request()
    assert tok["apiName"] == "VTubeStudioPublicAPI" and tok["messageType"] == "AuthenticationTokenRequest"
    assert tok["data"]["pluginName"] == "Crucible Companion"
    auth = b.auth_request("ABC")
    assert auth["messageType"] == "AuthenticationRequest" and auth["data"]["authenticationToken"] == "ABC"


def test_param_creation_covers_every_custom_param_with_ranges():
    reqs = VTSBridge().param_creation_requests()
    names = [r["data"]["parameterName"] for r in reqs]
    assert names == list(VTS_PARAMS)                              # one creation per custom param
    for r in reqs:
        d = r["data"]
        lo, hi, default, _ = PARAM_SPECS[d["parameterName"]]
        assert d["min"] == lo and d["max"] == hi and d["defaultValue"] == default
        assert lo <= default <= hi


def test_inject_request_carries_the_face_frame():
    frame = rig_frame({"happy": 1.0}, gaze=(0.5, 0.0))
    msg = VTSBridge().inject_request(frame["params"], gaze=frame["gaze"])
    assert msg["messageType"] == "InjectParameterDataRequest"
    ids = {pv["id"] for pv in msg["data"]["parameterValues"]}
    assert ids == set(VTS_PARAMS)
    eyex = next(pv for pv in msg["data"]["parameterValues"] if pv["id"] == "CrucibleEyeX")
    assert eyex["value"] == 0.5


# --- token cache -------------------------------------------------------------------------------------

def test_token_is_cached_and_reused(tmp_path):
    p = tmp_path / "vts_token.txt"
    b = VTSBridge(token_path=str(p))
    assert b.load_token() is None
    b.save_token("SECRET")
    assert p.read_text() == "SECRET" and VTSBridge(token_path=str(p)).load_token() == "SECRET"


# --- async loop over the fake socket -----------------------------------------------------------------

def test_full_run_authenticates_creates_params_and_streams(tmp_path):
    ws = FakeWS()
    b = VTSBridge(token_path=str(tmp_path / "tok.txt"))
    frames = [rig_frame({"happy": 1.0}), rig_frame({"surprised": 1.0}, gaze=(0.3, -0.2))]
    pushed = asyncio.run(b.run(frames, connect=_connect_factory(ws)))
    assert pushed == 2
    # token was requested once (none cached), then the session authenticated
    assert len(ws.of_type("AuthenticationTokenRequest")) == 1
    assert len(ws.of_type("AuthenticationRequest")) == 1
    assert len(ws.of_type("ParameterCreationRequest")) == len(VTS_PARAMS)   # params created once
    assert len(ws.of_type("InjectParameterDataRequest")) == 2               # one per frame
    # a fresh token got cached for next time
    assert (tmp_path / "tok.txt").read_text() == "TKN"


def test_run_reuses_cached_token_without_requesting_a_new_one(tmp_path):
    p = tmp_path / "tok.txt"
    p.write_text("CACHED")
    ws = FakeWS()
    b = VTSBridge(token_path=str(p))
    asyncio.run(b.run([rig_frame({"neutral": 1.0})], connect=_connect_factory(ws)))
    assert ws.of_type("AuthenticationTokenRequest") == []        # cached token used directly
    assert ws.of_type("AuthenticationRequest")[0]["data"]["authenticationToken"] == "CACHED"


def test_run_rerequests_when_cached_token_is_stale(tmp_path):
    p = tmp_path / "tok.txt"
    p.write_text("STALE")
    ws = FakeWS(grant_token="FRESH", first_auth_fails=True)      # first auth (STALE) fails, then a new token
    b = VTSBridge(token_path=str(p))
    pushed = asyncio.run(b.run([rig_frame({"neutral": 1.0})], connect=_connect_factory(ws)))
    assert pushed == 1
    assert len(ws.of_type("AuthenticationTokenRequest")) == 1    # asked for a new token after the failure
    assert p.read_text() == "FRESH"                              # cache refreshed


def test_run_raises_when_auth_denied(tmp_path):
    ws = FakeWS(authenticate=False)
    b = VTSBridge(token_path=str(tmp_path / "tok.txt"))
    try:
        asyncio.run(b.run([rig_frame({"neutral": 1.0})], connect=_connect_factory(ws)))
        assert False, "expected RuntimeError on denied auth"
    except RuntimeError as e:
        assert "authentication failed" in str(e).lower()


# --- live driver: idle-animated frames around a mutable mood ------------------------------------------

def test_idle_frames_track_mood_and_layer_saccades_and_blink():
    mood = {"weights": {"happy": 1.0}}
    frames = list(idle_frames(lambda: mood["weights"], count=120, blink_hold=3))
    assert len(frames) == 120
    assert all("params" in f and "arkit" in f for f in frames)              # full rig frames
    # the mood is reflected (happy → positive smile) …
    assert frames[0]["params"]["smile"] > 0
    # … gaze roves (saccades) and blinks are held for several frames (visible at high fps)
    xs = [f["gaze"][0] for f in frames]
    assert max(xs) - min(xs) > 0.2
    blink_run = max((sum(1 for _ in grp) for grp in _runs(f["blink"] == 1.0 for f in frames)), default=0)
    assert blink_run >= 2                                                   # held, not a 1-frame flicker


def test_idle_frames_follow_a_mood_change():
    mood = {"weights": {"sad": 1.0}}
    gen = idle_frames(lambda: mood["weights"], count=4)
    first = next(gen)
    assert first["params"]["smile"] < 0                                    # sad → frown
    mood["weights"] = {"happy": 1.0}                                       # an external reaction flips it
    after = next(gen)
    assert after["params"]["smile"] > 0                                    # the driver picks it up live


def _runs(bools):
    """Group a bool sequence into runs of equal value (for measuring blink hold length)."""
    import itertools
    return [list(g) for k, g in itertools.groupby(bools) if k]
