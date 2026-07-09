from __future__ import annotations
# LIVE bridge to VTube Studio — push Crucible's face state onto a real external VTuber rig over the VTS
# public websocket API (default ws://localhost:8001). This is the CONSUMER of `rigmap.to_vtube_studio`:
# the same engine-agnostic expression/gaze/blink state that animates the TUI pixel face drives the user's
# actual Live2D model in VTube Studio, in real time.
#
# Flow (per the VTS Public API): request a one-time auth TOKEN (the user approves a popup in VTS once),
# cache it, AUTHENTICATE the session with it, CREATE our custom parameters once, then stream
# InjectParameterData frames. All message construction here is PURE + testable; the only I/O is a thin
# async loop over a websocket object (duck-typed `.send(str)` / `.recv()->str`), so tests inject a fake.
import json
import os
from typing import Callable, Optional

from crucible.rigmap import VTS_PARAMS, rig_frame, to_vtube_studio

API_NAME = "VTubeStudioPublicAPI"
API_VERSION = "1.0"

# custom VTS parameter id → (min, max, default, explanation shown in the VTS parameter UI)
PARAM_SPECS: dict[str, tuple] = {
    "CrucibleBrow": (-1.0, 1.0, 0.0, "Crucible: brow (−furrowed … +raised)"),
    "CrucibleEyeOpen": (0.0, 1.0, 1.0, "Crucible: eye open (0 shut … 1 open)"),
    "CrucibleEyeWide": (0.0, 1.0, 0.0, "Crucible: eye wide (surprise)"),
    "CrucibleSmile": (-1.0, 1.0, 0.0, "Crucible: mouth form (−frown … +smile)"),
    "CrucibleMouthOpen": (0.0, 1.0, 0.0, "Crucible: mouth open (viseme/talk)"),
    "CrucibleBlush": (0.0, 1.0, 0.0, "Crucible: blush"),
    "CrucibleHeadTilt": (-1.0, 1.0, 0.0, "Crucible: head tilt"),
    "CrucibleEyeX": (-1.0, 1.0, 0.0, "Crucible: gaze X (−left … +right)"),
    "CrucibleEyeY": (-1.0, 1.0, 0.0, "Crucible: gaze Y (−up … +down)"),
}


class VTSBridge:
    """Drives VTube Studio from Crucible face frames. Construct, then either use the pure request builders
    (for a custom transport) or `await run(...)` to open the websocket and stream."""

    def __init__(self, host: str = "localhost", port: int = 8001, token_path: Optional[str] = None,
                 plugin_name: str = "Crucible Companion", plugin_developer: str = "pq-cybarg"):
        self.host = host
        self.port = port
        self.token_path = token_path
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    # --- pure message builders (testable, no I/O) -------------------------------------------------
    def _envelope(self, message_type: str, data: dict, request_id: str = "crucible") -> dict:
        return {"apiName": API_NAME, "apiVersion": API_VERSION, "requestID": request_id,
                "messageType": message_type, "data": data}

    def token_request(self) -> dict:
        return self._envelope("AuthenticationTokenRequest",
                              {"pluginName": self.plugin_name, "pluginDeveloper": self.plugin_developer},
                              request_id="crucible-token")

    def auth_request(self, token: str) -> dict:
        return self._envelope("AuthenticationRequest",
                              {"pluginName": self.plugin_name, "pluginDeveloper": self.plugin_developer,
                               "authenticationToken": token}, request_id="crucible-auth")

    def param_creation_requests(self) -> list[dict]:
        out = []
        for name in VTS_PARAMS:
            lo, hi, default, expl = PARAM_SPECS[name]
            out.append(self._envelope("ParameterCreationRequest",
                                      {"parameterName": name, "explanation": expl,
                                       "min": lo, "max": hi, "defaultValue": default},
                                      request_id=f"crucible-create-{name}"))
        return out

    def inject_request(self, params: dict, gaze=None, blink: float = 0.0) -> dict:
        """A ready InjectParameterDataRequest for one face frame (reuses the rigmap payload)."""
        msg = to_vtube_studio(params, gaze=gaze, blink=blink)
        msg.update({"apiVersion": API_VERSION, "requestID": "crucible-inject"})
        return msg

    # --- token cache ------------------------------------------------------------------------------
    def load_token(self) -> Optional[str]:
        if self.token_path and os.path.exists(self.token_path):
            try:
                tok = open(self.token_path, encoding="utf-8").read().strip()
                return tok or None
            except OSError:
                return None
        return None

    def save_token(self, token: str) -> None:
        if self.token_path and token:
            os.makedirs(os.path.dirname(self.token_path) or ".", exist_ok=True)
            with open(self.token_path, "w", encoding="utf-8") as fh:
                fh.write(token)

    # --- thin async I/O over a duck-typed websocket -----------------------------------------------
    async def _rpc(self, ws, message: dict) -> dict:
        """Send one request, return the parsed response. VTS answers one message per request."""
        await ws.send(json.dumps(message))
        raw = await ws.recv()
        return json.loads(raw)

    async def authenticate(self, ws) -> bool:
        """Obtain-or-reuse a token, then authenticate the session. Caches a freshly granted token. Returns
        True on success. A cached-but-revoked token transparently falls back to requesting a new one."""
        token = self.load_token()
        if not token:
            resp = await self._rpc(ws, self.token_request())
            token = (resp.get("data") or {}).get("authenticationToken")
            if not token:
                return False
            self.save_token(token)
        resp = await self._rpc(ws, self.auth_request(token))
        if (resp.get("data") or {}).get("authenticated"):
            return True
        # token was stale/revoked — request a brand-new one and retry once
        resp = await self._rpc(ws, self.token_request())
        token = (resp.get("data") or {}).get("authenticationToken")
        if not token:
            return False
        self.save_token(token)
        resp = await self._rpc(ws, self.auth_request(token))
        return bool((resp.get("data") or {}).get("authenticated"))

    async def create_params(self, ws) -> None:
        for req in self.param_creation_requests():
            await self._rpc(ws, req)

    async def send_frame(self, ws, params: dict, gaze=None, blink: float = 0.0) -> None:
        await ws.send(json.dumps(self.inject_request(params, gaze=gaze, blink=blink)))

    async def run(self, frames, connect=None) -> int:
        """Open the websocket, authenticate, create params, then stream face frames. `frames` is an
        (async or sync) iterable of dicts shaped like a `rigmap.rig_frame` result (with `params`/`gaze`/
        `blink`). `connect` overrides the websocket factory (tests inject a fake); default uses the
        `websockets` library. Returns the number of frames pushed. Raises if authentication fails."""
        if connect is None:
            import websockets
            connect = websockets.connect
        pushed = 0
        async with connect(self.url) as ws:
            if not await self.authenticate(ws):
                raise RuntimeError("VTube Studio authentication failed (was the plugin request denied?)")
            await self.create_params(ws)
            if hasattr(frames, "__aiter__"):
                async for f in frames:
                    await self.send_frame(ws, f.get("params", {}), f.get("gaze"), f.get("blink", 0.0))
                    pushed += 1
            else:
                for f in frames:
                    await self.send_frame(ws, f.get("params", {}), f.get("gaze"), f.get("blink", 0.0))
                    pushed += 1
        return pushed


def idle_frames(get_weights: Callable[[], dict], count: int, seed: int = 7, blink_hold: int = 3):
    """A pure generator of live rig frames: on each tick it reads the CURRENT mood (`get_weights()` — an
    external loop updates it as reactions fire), layers the idle animation on top (saccadic gaze, a held
    blink, a faint micro-expression merged into the mood), and yields a `rig_frame` dict. Timing is the
    caller's job (sleep between yields at your target fps), which keeps this deterministic + testable.
    `blink_hold` keeps the lids shut for a few frames so a blink is visible at high frame rates."""
    from crucible.animation import IdleAnimator
    idle = IdleAnimator(seed=seed)
    held = 0
    for _ in range(count):
        s = idle.step()
        if s.blink:
            held = blink_hold
        blink = 1.0 if held > 0 else 0.0
        held = max(0, held - 1)
        weights = dict(get_weights() or {"neutral": 1.0})
        for name, w in s.micro.items():                 # merge the faint micro-expression into the mood
            weights.setdefault(name, w)
        yield rig_frame(weights, gaze=s.gaze, blink=blink)
