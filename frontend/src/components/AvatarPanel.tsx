import type { JSX } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { getAvatarInfo, getReactionFrame, postRigFrame } from "../api";
import type { AvatarInfo } from "../api";
import { browOffset, eyeGeometry, mouthPath, readFace } from "../avatar/face";
import { makeIdle } from "../avatar/idle";

// The web COMPANION window: the same engine-agnostic face state that drives the TUI pixel face and an
// external VTube-Studio rig, rendered here as a live SVG face. The backend maps a mood BLEND → Live2D-style
// params (POST /api/avatar/rig-frame); the browser overlays saccadic gaze + blink + talk locally so it
// animates smoothly. Swapping in a real Live2D/VRM renderer later means feeding the SAME params to it.

const BLINK_HOLD = 4;                                       // frames to hold lids shut so a blink reads
const REACTIONS = ["funny", "cute", "scary", "sad", "surprised", "sus", "calm"] as const;

// One rendered face from a Live2D param record. Pure geometry (avatar/face.ts) → SVG.
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
        {/* hair back + head + hair top */}
        <path d="M22 60 Q18 18 60 14 Q102 18 98 60 Q98 96 60 100 Q22 96 22 60 Z" fill="#4a3a34" />
        <ellipse cx={60} cy={68} rx={36} ry={40} fill="#f0dcc0" stroke="#3a2e28" strokeWidth={1.5} />
        <path d="M24 52 Q30 22 60 20 Q90 22 96 52 Q80 40 60 40 Q40 40 24 52 Z" fill="#4a3a34" />
        {/* blush */}
        <ellipse cx={38} cy={80} rx={7} ry={4} fill="#e8918f" opacity={f.cheek} />
        <ellipse cx={82} cy={80} rx={7} ry={4} fill="#e8918f" opacity={f.cheek} />
        {/* brows */}
        <line x1={34} y1={50 + brow} x2={52} y2={48 + brow} stroke="#4a3a34" strokeWidth={2.5} strokeLinecap="round" />
        <line x1={68} y1={48 + brow} x2={86} y2={50 + brow} stroke="#4a3a34" strokeWidth={2.5} strokeLinecap="round" />
        {eye(L)}
        {eye(R)}
        {/* nose hint + mouth */}
        <circle cx={60} cy={86} r={1.2} fill="#c9a888" />
        <path d={mouthPath(60, 98, 12, f)} fill="#b5484a" stroke="#3a2e28" strokeWidth={1.5} strokeLinejoin="round" />
      </g>
    </svg>
  );
}

export default function AvatarPanel(): JSX.Element {
  const [info, setInfo] = useState<AvatarInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [active, setActive] = useState("neutral");
  const [mixing, setMixing] = useState(false);
  const [weights, setWeights] = useState<Record<string, number>>({ neutral: 1 });
  const [talking, setTalking] = useState(false);
  const [autoGaze, setAutoGaze] = useState(true);
  const [manual, setManual] = useState<[number, number]>([0, 0]);
  const [display, setDisplay] = useState<Record<string, number>>({});

  // refs the animation loop reads without restarting (kept in sync post-render, not during it)
  const baseRef = useRef<Record<string, number>>({});
  const talkRef = useRef(false);
  const autoRef = useRef(true);
  const manualRef = useRef<[number, number]>([0, 0]);
  const postTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => { talkRef.current = talking; autoRef.current = autoGaze; manualRef.current = manual; });

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const i = await getAvatarInfo();
        if (!alive) return;
        setInfo(i);
        const frame = await postRigFrame({ neutral: 1 });
        if (alive) baseRef.current = { ...frame.live2d };
      } catch (e: unknown) { if (alive) setErr(e instanceof Error ? e.message : "avatar unavailable"); }
    })();
    return () => { alive = false; };
  }, []);

  // fetch the authoritative mood params for a blend and stash them as the animation base
  async function applyWeights(w: Record<string, number>, primary?: string): Promise<void> {
    try {
      const frame = await postRigFrame(w);
      baseRef.current = { ...frame.live2d };
      if (primary) setActive(primary);
      setErr(null);
    } catch (e: unknown) { setErr(e instanceof Error ? e.message : "rig-frame failed"); }
  }

  function pickExpression(name: string): void {
    const w = { [name]: 1 };
    setWeights(w); setActive(name); void applyWeights(w, name);
  }

  function setMixWeight(name: string, v: number): void {
    const next = { ...weights, [name]: v };
    for (const k of Object.keys(next)) if (next[k] <= 0) delete next[k];
    const w = Object.keys(next).length ? next : { neutral: 1 };
    setWeights(w);
    if (postTimer.current) clearTimeout(postTimer.current);
    postTimer.current = setTimeout(() => { void applyWeights(w); }, 80);   // debounce slider drags
  }

  async function react(word: string): Promise<void> {
    try {
      const frame = await getReactionFrame(word);
      baseRef.current = { ...frame.live2d };
      setActive(word); setWeights({ [word]: 1 }); setErr(null);
    } catch (e: unknown) { setErr(e instanceof Error ? e.message : "reaction failed"); }
  }

  // the render loop: step idle, overlay gaze/blink/talk onto the mood base, ~20fps (low-fps by design)
  useEffect(() => {
    const idle = makeIdle();
    let raf = 0, last = 0, held = 0;
    const loop = (now: number): void => {
      raf = requestAnimationFrame(loop);
      if (now - last < 50) return;
      last = now;
      const it = idle();
      if (it.blink) held = BLINK_HOLD;
      const blink = held > 0 ? 1 : 0; held = Math.max(0, held - 1);
      const gaze = autoRef.current ? it.gaze : manualRef.current;
      const base = baseRef.current;
      const merged: Record<string, number> = { ...base };
      merged["ParamEyeBallX"] = gaze[0]; merged["ParamEyeBallY"] = -gaze[1];
      const eyeBase = typeof base["ParamEyeLOpen"] === "number" ? base["ParamEyeLOpen"] : 1;
      merged["ParamEyeLOpen"] = merged["ParamEyeROpen"] = (1 - blink) * eyeBase;
      if (talkRef.current) {
        const flap = Math.floor(now / 130) % 2 === 0 ? 0.55 : 0.08;
        const baseMouth = typeof base["ParamMouthOpenY"] === "number" ? base["ParamMouthOpenY"] : 0;
        merged["ParamMouthOpenY"] = Math.max(baseMouth, flap);
      }
      setDisplay(merged);
    };
    raf = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(raf); if (postTimer.current) clearTimeout(postTimer.current); };
  }, []);

  const expressions = useMemo(() => info?.expressions ?? [], [info]);

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>com<em>panion</em></h1>
        <p>The avatar window — the same face state that drives the terminal face and a VTube&nbsp;Studio rig,
          here as a live SVG. Pick a mood or <b>mix</b> several (blendshape-style), toggle <b>talk</b>, and
          it glances around and blinks on its own. Reactions map the co-watch vocabulary onto the face.</p>
      </div>

      {err && <div className="chip block" style={{ marginBottom: 14 }}>{err}</div>}

      <div className="avatar-stage">
        <div className="avatar-face">
          <FaceSvg live2d={display} />
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
              <button key={r} className="chip" onClick={() => void react(r)}>{r}</button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
