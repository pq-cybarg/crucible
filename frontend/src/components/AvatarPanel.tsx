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

const REACTIONS = ["funny", "cute", "scary", "sad", "surprised", "sus", "calm"] as const;

function blendString(weights: Record<string, number>): string {
  const parts = Object.entries(weights).filter(([, w]) => w > 0).map(([k, w]) => `${k}:${w}`);
  return parts.length ? parts.join(",") : "neutral:1";
}

function dominant(weights: Record<string, number>): string {
  let best = "neutral", bw = -1;
  for (const [k, w] of Object.entries(weights)) if (w > bw) { bw = w; best = k; }
  return best;
}

// Per-expression MOTION so moods play as animations, not stills: a mouth flap (talk), a vertical head
// bob, and a slight tilt, as functions of time. Closed-eye moods (laughing/love) come alive through the
// bounce/sway rather than staring blankly. Amplitudes are small — lively, not seasick.
// armL/armR = shoulder rotation degrees for the two movable arm sprites. By the render convention a
// NEGATIVE armL / POSITIVE armR swing the arms OUTWARD (spread); POSITIVE armL / NEGATIVE armR draw
// them INWARD (hands toward centre / hugging).
function expressionAnim(name: string, now: number):
    { talk: number; bob: number; tilt: number; armL: number; armR: number } {
  const s = (p: number): number => Math.sin(now / p);
  const spread = (base: number, amp: number, per: number): { armL: number; armR: number } =>
    ({ armL: -(base + Math.abs(s(per)) * amp), armR: base + Math.abs(s(per)) * amp });   // symmetric OUT
  const hug = (base: number, amp: number, per: number): { armL: number; armR: number } =>
    ({ armL: base + s(per) * amp, armR: -(base + s(per) * amp) });                        // symmetric IN
  // Head/neck motion is SUBTLE on purpose — a gentle sway/breathe, not a bounce. Big bob/tilt shears the
  // neck against the rigid shoulders (reads as a stretchy blob). Keep amplitudes small; the arms + hair
  // physics carry most of the liveliness.
  switch (name) {
    case "laughing":  return { talk: 0.45 + 0.35 * s(105), bob: -Math.abs(s(150)) * 2, tilt: s(300) * 1.0, ...spread(14, 8, 150) };  // arms up, giggling
    case "love":      return { talk: 0, bob: s(520) * 1.0, tilt: s(760) * 1.6, ...hug(10, 3, 520) };                              // hands drawn in, dreamy
    case "happy":     return { talk: 0, bob: s(440) * 0.9, tilt: s(900) * 0.7, ...spread(8, 4, 440) };                            // gentle spread
    case "surprised": return { talk: 0, bob: -1 + s(230) * 0.7, tilt: 0, armL: -22, armR: 22 };                                   // arms flung out
    // trembling moods: TILT drives the hair-side sway, so a tiny tilt (old 0.8-0.9) left the sides dead while
    // the heavy hair damping filtered it out. Give them a followable sway (bigger, > laughing's tilt) PLUS a
    // fast micro-shudder for the tense read — the sway moves the side hair, the shudder sells the trembling.
    case "scared":    return { talk: 0, bob: s(60) * 0.9 + s(22) * 0.4, tilt: s(72) * 1.5 + s(20) * 0.5, ...hug(15, 3, 70) };     // hugging self, trembling
    case "angry":     return { talk: 0, bob: s(70) * 0.9 + s(26) * 0.4, tilt: s(85) * 1.7 + s(24) * 0.5, ...spread(11, 5, 95) };  // tense, shaking
    case "sad":       return { talk: 0, bob: 1.0 + s(1000) * 0.5, tilt: -1 + s(1200) * 0.5, ...hug(6, 2, 1200) };                // limp, drawn in
    default:          return { talk: 0, bob: s(1500) * 0.5, tilt: 0, armL: -3 + s(1500) * 2.5, armR: 3 - s(1500) * 2.5 };         // quiet breathing sway
  }
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
  const [off, setOff] = useState<string[]>([]);                     // EXPLICITLY-disabled part ids
  const [ready, setReady] = useState(false);                        // live: first frame arrived
  const [svgParams, setSvgParams] = useState<Record<string, number>>({});  // demo: live2d for the SVG
  const img0Ref = useRef<HTMLImageElement | null>(null);            // double-buffer for a smooth crossfade
  const img1Ref = useRef<HTMLImageElement | null>(null);
  const sidRef = useRef<string>(`av-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`);  // stateful hair-physics session

  const weightsRef = useRef<Record<string, number>>({ neutral: 1 });
  const talkRef = useRef(false);
  const autoRef = useRef(true);
  const manualRef = useRef<[number, number]>([0, 0]);
  const offRef = useRef<string[]>([]);
  useEffect(() => {
    weightsRef.current = weights; talkRef.current = talking; autoRef.current = autoGaze; manualRef.current = manual;
    offRef.current = off;
  });
  // PART HIERARCHY. A part renders unless it OR an ancestor is explicitly disabled — so re-enabling a
  // parent respects a child's own disable (its own toggle stays off). `soon` = not yet a separated part.
  const PART_TREE: { id: string; label: string; soon?: boolean; kids?: { id: string; label: string; soon?: boolean }[] }[] = [
    { id: "eyes", label: "eyes", kids: [
      { id: "irises", label: "irises" }, { id: "pupils", label: "pupils" },
      { id: "whites", label: "whites", soon: true },
      { id: "eyelashes", label: "eyelashes" }, { id: "eyelids", label: "eyelids", soon: true }] },
    { id: "nose", label: "nose", soon: true },
    { id: "mouth", label: "mouth", kids: [
      { id: "mouth-lips", label: "lips" }, { id: "mouth-inside", label: "inside" },
      { id: "mouth-teeth", label: "teeth", soon: true }, { id: "mouth-tongue", label: "tongue", soon: true }] },
    { id: "hair", label: "hair", kids: [{ id: "hair-sub", label: "subsections", soon: true }] },
    { id: "brows", label: "brows" }, { id: "blush", label: "blush" }, { id: "glasses", label: "glasses" },
    { id: "headphones", label: "headphones" }, { id: "body", label: "body" },
  ];
  const parentOf: Record<string, string> = {};
  PART_TREE.forEach((n) => n.kids?.forEach((k) => { parentOf[k.id] = n.id; }));
  const effHidden = (id: string): boolean => off.includes(id) || (parentOf[id] ? effHidden(parentOf[id]) : false);
  function togglePart(id: string): void {
    setOff((o) => (o.includes(id) ? o.filter((x) => x !== id) : [...o, id]));
  }

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
    for (const k of Object.keys(next)) if ((next[k] ?? 0) <= 0) delete next[k];
    setWeights(Object.keys(next).length ? next : { neutral: 1 });
  }

  // the render loop: step idle, GLIDE the gaze toward its target (no jump cuts), then either fetch the
  // real PNG (live) or update the SVG params (demo). Live frames crossfade through a double image buffer
  // so mood/eye changes ease in instead of hard-switching. ~16fps; a tick is skipped if a fetch is pending.
  useEffect(() => {
    const idle = makeIdle({ blinkEvery: [72, 168] });        // ~4.5–10s between blinks (natural rate)
    let raf = 0, last = 0, inFlight = false, cancelled = false;
    let cur: [number, number] = [0, 0];                      // current (eased) gaze
    const urls: (string | null)[] = [null, null];
    let top = 0;
    let curW: Record<string, number> = { neutral: 1 };        // eased weights → the face MORPHS between moods
    let blinkSeq: number[] = [];                              // blink amount curve: ease shut → hold → open
    const loop = (now: number): void => {
      raf = requestAnimationFrame(loop);
      if (now - last < 60) return;
      last = now;
      const it = idle();
      // a slower, smoother blink: ease half → shut → half → open over ~8 frames (a 4-frame flick read as
      // too fast, and its brief half frames flashed). The half frames keep the iris now, so they're a
      // natural hooded lid, not a blank eye.
      if (it.blink && blinkSeq.length === 0) blinkSeq = [0.4, 0.6, 0.85, 1, 1, 0.85, 0.6, 0.4];
      const blink = blinkSeq.shift() ?? 0;
      const target = autoRef.current ? it.gaze : manualRef.current;
      cur = [cur[0] + (target[0] - cur[0]) * 0.3, cur[1] + (target[1] - cur[1]) * 0.3];   // ease gaze
      const gaze: [number, number] = [Math.round(cur[0] * 100) / 100, Math.round(cur[1] * 100) / 100];
      // AUTO-BALANCE the mix: neutral fills only the leftover weight, so adding a mood actually shifts the
      // face toward it (instead of neutral:1 always dominating). Sum of moods >1 → neutral drops to 0.
      const w0 = weightsRef.current;
      const moodSum = Object.entries(w0).reduce((a, [k, v]) => a + (k !== "neutral" && v > 0 ? v : 0), 0);
      const targetW = moodSum > 0 ? { ...w0, neutral: Math.max(0, 1 - moodSum) } : w0;
      // EASE the weights toward the target so the parametric face MORPHS through real intermediate blends
      // (a smooth expression change), instead of the old image cross-dissolve.
      const wk = new Set([...Object.keys(curW), ...Object.keys(targetW)]);
      const nextW: Record<string, number> = {};
      wk.forEach((k) => {
        const v = (curW[k] ?? 0) + ((targetW[k] ?? 0) - (curW[k] ?? 0)) * 0.2;
        if (v > 0.004) nextW[k] = Math.round(v * 1000) / 1000;
      });
      curW = Object.keys(nextW).length ? nextW : { neutral: 1 };
      const w = curW;
      const anim = expressionAnim(dominant(w), now);
      // continuous lip-open (0..1) — a smooth oscillation, NOT a binary flap, so the mouth morphs
      const talk = talkRef.current ? (0.45 + 0.45 * Math.sin(now / 80)) : anim.talk;
      const bob = Math.round(anim.bob);
      const tilt = Math.round(anim.tilt * 100) / 100;
      const armL = Math.round(anim.armL * 100) / 100;
      const armR = Math.round(anim.armR * 100) / 100;
      if (demo) {
        const frame = demoRigFrame(w, gaze, blink);
        const l = { ...frame.live2d };
        if (talk) l["ParamMouthOpenY"] = 0.55;
        l["ParamAngleZ"] = (l["ParamAngleZ"] ?? 0) + tilt;
        setSvgParams(l);
      } else if (!inFlight) {
        inFlight = true;
        const blendKey = blendString(w);
        fetchAvatarRender({ blend: blendKey, gx: gaze[0], gy: gaze[1], blink, talk, bob, tilt, armL, armR,
                            hairPhys: true, sid: sidRef.current, hide: offRef.current.join(","), scale: 384 })
          .then((blob) => {
            if (cancelled) { inFlight = false; return; }
            const url = URL.createObjectURL(blob);
            const back = 1 - top;
            const bimg = back === 0 ? img0Ref.current : img1Ref.current;
            const fimg = top === 0 ? img0Ref.current : img1Ref.current;
            if (!bimg) { URL.revokeObjectURL(url); inFlight = false; return; }
            // Swap ONLY once the new frame has decoded (no blank flash), and hold the next fetch until then
            // (one frame in flight → buffers never overlap). Consecutive ANIMATION frames swap INSTANTLY
            // (like video — no pulsing); only a genuine MOOD change cross-dissolves (a smooth transition).
            // ALWAYS an instant swap — the face already morphs smoothly because the WEIGHTS ease (each
            // frame is a real intermediate blend). A cross-dissolve on top would just ghost the morph.
            bimg.onload = () => {
              bimg.style.transition = "none";
              if (fimg) fimg.style.transition = "none";
              bimg.style.opacity = "1";
              if (fimg) fimg.style.opacity = "0";
              const old = urls[back]; urls[back] = url; top = back;
              if (old) URL.revokeObjectURL(old);
              setReady(true);
              setErr((prev) => (prev && prev.startsWith("GET /api/avatar/render") ? null : prev));
              inFlight = false;
            };
            bimg.onerror = () => { URL.revokeObjectURL(url); inFlight = false; };
            bimg.src = url;
          })
          .catch((e: unknown) => { if (!cancelled) setErr(e instanceof Error ? e.message : "render failed"); inFlight = false; });
      }
    };
    raf = requestAnimationFrame(loop);
    return () => { cancelled = true; cancelAnimationFrame(raf); urls.forEach((u) => u && URL.revokeObjectURL(u)); };
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
            : <>
                <img ref={img0Ref} className="avatar-frame" alt="companion" />
                <img ref={img1Ref} className="avatar-frame" alt="" />
                {!ready && <div className="avatar-loading">…</div>}
              </>}
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

          {!demo && (
            <div className="avatar-mix">
              <p className="avatar-hint">Parts — toggle on/off (groups cascade; a child stays off when you re-enable its group). <em>soon</em> = not yet a separated part.</p>
              {PART_TREE.map((n) => (
                <div key={n.id} style={{ marginBottom: 4 }}>
                  <button className={`btn ghost${n.soon ? " soon" : ""}${effHidden(n.id) ? "" : " on"}`}
                    disabled={n.soon} onClick={() => togglePart(n.id)}>{n.label}{n.soon ? " ·soon" : ""}</button>
                  {n.kids && (
                    <span style={{ marginLeft: 10 }}>
                      {n.kids.map((k) => (
                        <button key={k.id} className={`btn ghost${k.soon ? " soon" : ""}${effHidden(k.id) ? "" : " on"}`}
                          disabled={k.soon} onClick={() => togglePart(k.id)}
                          style={{ opacity: effHidden(n.id) && !off.includes(k.id) ? 0.4 : 1 }}>
                          {k.label}{k.soon ? " ·soon" : ""}</button>
                      ))}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}

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
