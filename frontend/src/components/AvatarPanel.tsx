import type { JSX } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchAvatarRender, getAvatarInfo } from "../api";
import type { AvatarInfo } from "../api";
import { browOffset, eyeGeometry, mouthPath, readFace } from "../avatar/face";
import { DEMO_AVATAR, DEMO_REACTION, demoRigFrame } from "../avatar/demoRig";
import { isDemo } from "../demo";
import { makeIdle } from "../avatar/idle";

// The web COMPANION window. LIVE (a backend is connected): it displays the ACTUAL avatar art — the same
// cute-anime sprite the TUI face shows — server-rendered to PNG per frame with gaze/blink/talk, so the
// browser shows the real character, not a stand-in. OFFLINE/demo (GitHub Pages, no node): it falls back
// to a self-contained SVG face driven by the same Live2D-style params. Either way the controls are the
// same: pick a mood, MIX several by weight (blendshape-style), toggle talk, auto/manual gaze, reactions.

const BLINK_HOLD = 4;
const REACTIONS = ["funny", "cute", "scary", "sad", "surprised", "sus", "calm"] as const;

function blendString(weights: Record<string, number>): string {
  const parts = Object.entries(weights).filter(([, w]) => w > 0).map(([k, w]) => `${k}:${w}`);
  return parts.length ? parts.join(",") : "neutral:1";
}

// --- offline SVG face (demo only) --------------------------------------------------------------------
function FaceSvg({ live2d }: { readonly live2d: Record<string, number> }): JSX.Element {
  const f = readFace(live2d);
  const L = eyeGeometry(44, 66, 12, f);
  const R = eyeGeometry(76, 66, 12, f);
  const brow = browOffset(12, f);
  const eye = (e: ReturnType<typeof eyeGeometry>): JSX.Element => (
    <g>
      <ellipse cx={e.cx} cy={e.cy} rx={e.rx} ry={e.ry} fill="#fdfaf4" stroke="#3a2e28" strokeWidth={1.5} />
      {!e.shut && <circle cx={e.pupilX} cy={e.pupilY} r={e.pupilR} fill="#5b3fa6" />}
      {!e.shut && <circle cx={e.pupilX - 1.5} cy={e.pupilY - 1.5} r={e.pupilR * 0.3} fill="#fff" opacity={0.9} />}
      {e.shut && <line x1={e.cx - e.rx} y1={e.cy} x2={e.cx + e.rx} y2={e.cy} stroke="#3a2e28" strokeWidth={2} strokeLinecap="round" />}
    </g>
  );
  return (
    <svg viewBox="0 0 120 140" role="img" aria-label="companion face" style={{ width: "100%", height: "100%" }}>
      <g transform={`rotate(${f.angleZ} 60 70)`}>
        <path d="M22 60 Q18 18 60 14 Q102 18 98 60 Q98 96 60 100 Q22 96 22 60 Z" fill="#4a3a34" />
        <ellipse cx={60} cy={68} rx={36} ry={40} fill="#f0dcc0" stroke="#3a2e28" strokeWidth={1.5} />
        <path d="M24 52 Q30 22 60 20 Q90 22 96 52 Q80 40 60 40 Q40 40 24 52 Z" fill="#4a3a34" />
        <ellipse cx={38} cy={80} rx={7} ry={4} fill="#e8918f" opacity={f.cheek} />
        <ellipse cx={82} cy={80} rx={7} ry={4} fill="#e8918f" opacity={f.cheek} />
        <line x1={34} y1={50 + brow} x2={52} y2={48 + brow} stroke="#4a3a34" strokeWidth={2.5} strokeLinecap="round" />
        <line x1={68} y1={48 + brow} x2={86} y2={50 + brow} stroke="#4a3a34" strokeWidth={2.5} strokeLinecap="round" />
        {eye(L)}{eye(R)}
        <circle cx={60} cy={86} r={1.2} fill="#c9a888" />
        <path d={mouthPath(60, 98, 12, f)} fill="#b5484a" stroke="#3a2e28" strokeWidth={1.5} strokeLinejoin="round" />
      </g>
    </svg>
  );
}

export default function AvatarPanel(): JSX.Element {
  const demo = isDemo();
  const [info, setInfo] = useState<AvatarInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [active, setActive] = useState("neutral");
  const [mixing, setMixing] = useState(false);
  const [weights, setWeights] = useState<Record<string, number>>({ neutral: 1 });
  const [talking, setTalking] = useState(false);
  const [autoGaze, setAutoGaze] = useState(true);
  const [manual, setManual] = useState<[number, number]>([0, 0]);
  const [imgUrl, setImgUrl] = useState<string | null>(null);        // live: object URL of the rendered PNG
  const [svgParams, setSvgParams] = useState<Record<string, number>>({});  // demo: live2d for the SVG

  const weightsRef = useRef<Record<string, number>>({ neutral: 1 });
  const talkRef = useRef(false);
  const autoRef = useRef(true);
  const manualRef = useRef<[number, number]>([0, 0]);
  useEffect(() => {
    weightsRef.current = weights; talkRef.current = talking; autoRef.current = autoGaze; manualRef.current = manual;
  });

  useEffect(() => {
    getAvatarInfo()
      .then((i) => { setInfo(i); setErr(null); })
      .catch((e: unknown) => {
        setInfo(DEMO_AVATAR);   // keep the controls usable
        if (e instanceof Error && e.message === "stale-backend") {
          setErr("The backend is running an older build without the avatar endpoints — restart it (stop the Crucible server and start it again) to load the companion.");
        }
      });
  }, []);

  function setMoodFrom(w: Record<string, number>, primary?: string): void {
    setWeights(w);
    if (primary) setActive(primary);
  }
  function pickExpression(name: string): void { setMoodFrom({ [name]: 1 }, name); }
  function react(word: string): void {
    const expr = DEMO_REACTION[word] ?? word;    // map reaction → expression client-side (no extra request)
    setMoodFrom({ [expr]: 1 }, word);
  }
  function setMixWeight(name: string, v: number): void {
    const next = { ...weights, [name]: v };
    for (const k of Object.keys(next)) if (next[k] <= 0) delete next[k];
    setWeights(Object.keys(next).length ? next : { neutral: 1 });
  }

  // the render loop: step idle, compute gaze/blink/talk, then either fetch the real PNG (live) or update
  // the SVG params (demo). ~14fps — low-fps by design, and we skip a tick if a fetch is still in flight.
  useEffect(() => {
    const idle = makeIdle();
    let raf = 0, last = 0, held = 0, inFlight = false;
    let lastUrl: string | null = null;
    let cancelled = false;
    const loop = (now: number): void => {
      raf = requestAnimationFrame(loop);
      if (now - last < 70) return;
      last = now;
      const it = idle();
      if (it.blink) held = BLINK_HOLD;
      const blink = held > 0 ? 1 : 0; held = Math.max(0, held - 1);
      const gaze = autoRef.current ? it.gaze : manualRef.current;
      const talk = talkRef.current ? (Math.floor(now / 140) % 2 === 0 ? 1 : 0) : 0;
      const w = weightsRef.current;
      if (demo) {
        const frame = demoRigFrame(w, gaze, blink);
        const l = { ...frame.live2d };
        if (talk) l["ParamMouthOpenY"] = 0.55;
        setSvgParams(l);
      } else if (!inFlight) {
        inFlight = true;
        fetchAvatarRender({ blend: blendString(w), gx: gaze[0], gy: gaze[1], blink, talk, scale: 384 })
          .then((blob) => {
            if (cancelled) return;
            const url = URL.createObjectURL(blob);
            setImgUrl(url);
            if (lastUrl) URL.revokeObjectURL(lastUrl);
            lastUrl = url;
            setErr((prev) => (prev && prev.startsWith("GET /api/avatar/render") ? null : prev));
          })
          .catch((e: unknown) => { if (!cancelled) setErr(e instanceof Error ? e.message : "render failed"); })
          .finally(() => { inFlight = false; });
      }
    };
    raf = requestAnimationFrame(loop);
    return () => { cancelled = true; cancelAnimationFrame(raf); if (lastUrl) URL.revokeObjectURL(lastUrl); };
  }, [demo]);

  const expressions = useMemo(() => info?.expressions ?? [], [info]);

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>com<em>panion</em></h1>
        <p>The avatar window — the real cute-anime companion (the same face the terminal shows and a
          VTube&nbsp;Studio rig would), rendered live. Pick a mood or <b>mix</b> several (blendshape-style),
          toggle <b>talk</b>, and it glances around and blinks on its own. Reactions map the co-watch
          vocabulary onto the face.</p>
      </div>

      {err && <div className="chip block" style={{ marginBottom: 14 }}>{err}</div>}

      <div className="avatar-stage">
        <div className="avatar-face">
          {demo
            ? <FaceSvg live2d={svgParams} />
            : imgUrl
              ? <img src={imgUrl} alt="companion" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
              : <div className="avatar-loading">…</div>}
        </div>

        <div className="avatar-controls">
          <div className="avatar-row">
            {expressions.map((name) => (
              <button key={name} className={`btn ghost${active === name ? " on" : ""}`}
                onClick={() => pickExpression(name)}>{name}</button>
            ))}
          </div>

          <div className="avatar-row avatar-toggles">
            <button className={`btn ghost${talking ? " on" : ""}`} onClick={() => setTalking((t) => !t)}>
              {talking ? "talking…" : "talk"}</button>
            <button className={`btn ghost${autoGaze ? " on" : ""}`} onClick={() => setAutoGaze((g) => !g)}>
              {autoGaze ? "auto gaze" : "manual gaze"}</button>
            <button className={`btn ghost${mixing ? " on" : ""}`} onClick={() => setMixing((m) => !m)}>mix</button>
          </div>

          {!autoGaze && (
            <div className="avatar-gaze">
              <label>look X<input type="range" min={-1} max={1} step={0.05} value={manual[0]}
                onChange={(e) => setManual([Number(e.target.value), manual[1]])} /></label>
              <label>look Y<input type="range" min={-1} max={1} step={0.05} value={manual[1]}
                onChange={(e) => setManual([manual[0], Number(e.target.value)])} /></label>
            </div>
          )}

          {mixing && (
            <div className="avatar-mix">
              <p className="avatar-hint">Blend expressions by weight — a real-time blendshape mix.</p>
              {expressions.map((name) => (
                <label key={name} className="avatar-slider">
                  <span>{name}</span>
                  <input type="range" min={0} max={1} step={0.05} value={weights[name] ?? 0}
                    onChange={(e) => setMixWeight(name, Number(e.target.value))} />
                  <em>{(weights[name] ?? 0).toFixed(2)}</em>
                </label>
              ))}
            </div>
          )}

          <div className="avatar-row avatar-reactions">
            {REACTIONS.map((r) => (
              <button key={r} className="chip" onClick={() => react(r)}>{r}</button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
