import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";
import { fetchAvatarRender } from "../api";
import { makeIdle } from "../avatar/idle";
import { blendString, dominant, expressionAnim } from "../avatar/anim";

/**
 * A small, always-on companion avatar for the CHAT window — the same live-rendered face as the
 * companion tab (shares ../avatar/anim so motion stays in lock-step), driven by the conversation
 * rather than manual controls: she idles (blinks + glances) and TALKS while the agent is streaming.
 *
 * Self-contained (own hair-physics sid + double image buffer + render loop) so it never touches the
 * companion tab's state; if the backend lacks the avatar endpoints it silently renders nothing.
 */
export default function ChatAvatar(
  { talking = false, mood, size = 120 }:
  { readonly talking?: boolean; readonly mood?: Record<string, number>; readonly size?: number },
): JSX.Element | null {
  const img0Ref = useRef<HTMLImageElement | null>(null);
  const img1Ref = useRef<HTMLImageElement | null>(null);
  const sidRef = useRef<string>(`chat-${Math.random().toString(36).slice(2, 10)}`);
  const talkRef = useRef(talking);
  const moodRef = useRef<Record<string, number>>(mood ?? { neutral: 1 });
  const [ready, setReady] = useState(false);
  const [dead, setDead] = useState(false);          // backend has no avatar endpoints → hide entirely
  useEffect(() => { talkRef.current = talking; moodRef.current = mood ?? { neutral: 1 }; });

  useEffect(() => {
    const idle = makeIdle({ blinkEvery: [72, 168] });
    let raf = 0, last = 0, inFlight = false, cancelled = false, fails = 0;
    let cur: [number, number] = [0, 0];
    let curW: Record<string, number> = { neutral: 1 };
    let blinkSeq: number[] = [];
    const urls: (string | null)[] = [null, null];
    let top = 0;
    const loop = (now: number): void => {
      raf = requestAnimationFrame(loop);
      if (now - last < 66) return;                  // ~15fps, gentle on the chat
      last = now;
      const it = idle();
      if (it.blink && blinkSeq.length === 0) blinkSeq = [0.4, 0.6, 0.85, 1, 1, 0.85, 0.6, 0.4];
      const blink = blinkSeq.shift() ?? 0;
      cur = [cur[0] + (it.gaze[0] - cur[0]) * 0.3, cur[1] + (it.gaze[1] - cur[1]) * 0.3];
      const gaze: [number, number] = [Math.round(cur[0] * 100) / 100, Math.round(cur[1] * 100) / 100];
      const w0 = moodRef.current;
      const moodSum = Object.entries(w0).reduce((a, [k, v]) => a + (k !== "neutral" && v > 0 ? v : 0), 0);
      const targetW = moodSum > 0 ? { ...w0, neutral: Math.max(0, 1 - moodSum) } : w0;
      const wk = new Set([...Object.keys(curW), ...Object.keys(targetW)]);
      const nextW: Record<string, number> = {};
      wk.forEach((k) => {
        const v = (curW[k] ?? 0) + ((targetW[k] ?? 0) - (curW[k] ?? 0)) * 0.2;
        if (v > 0.004) nextW[k] = Math.round(v * 1000) / 1000;
      });
      curW = Object.keys(nextW).length ? nextW : { neutral: 1 };
      const anim = expressionAnim(dominant(curW), now);
      const talk = talkRef.current ? (0.45 + 0.45 * Math.sin(now / 80)) : anim.talk;
      if (inFlight) return;
      inFlight = true;
      fetchAvatarRender({ blend: blendString(curW), gx: gaze[0], gy: gaze[1], blink, talk,
                          bob: Math.round(anim.bob), tilt: Math.round(anim.tilt * 100) / 100,
                          armL: Math.round(anim.armL * 100) / 100, armR: Math.round(anim.armR * 100) / 100,
                          hairPhys: true, sid: sidRef.current, scale: 256 })
        .then((blob) => {
          if (cancelled) { inFlight = false; return; }
          fails = 0;
          const url = URL.createObjectURL(blob);
          const back = 1 - top;
          const bimg = back === 0 ? img0Ref.current : img1Ref.current;
          const fimg = top === 0 ? img0Ref.current : img1Ref.current;
          if (!bimg) { URL.revokeObjectURL(url); inFlight = false; return; }
          bimg.onload = () => {
            bimg.style.opacity = "1";
            if (fimg) fimg.style.opacity = "0";
            const old = urls[back]; urls[back] = url; top = back;
            if (old) URL.revokeObjectURL(old);
            setReady(true);
            inFlight = false;
          };
          bimg.onerror = () => { URL.revokeObjectURL(url); inFlight = false; };
          bimg.src = url;
        })
        .catch(() => { inFlight = false; if (++fails >= 3) setDead(true); });   // no avatar backend → give up
    };
    raf = requestAnimationFrame(loop);
    return () => { cancelled = true; cancelAnimationFrame(raf); urls.forEach((u) => u && URL.revokeObjectURL(u)); };
  }, []);

  if (dead) return null;
  return (
    <div className="chat-avatar" style={{ width: size, height: size }} aria-label="companion" title="companion">
      <img ref={img0Ref} alt="companion" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", objectPosition: "center 15%", opacity: 0 }} />
      <img ref={img1Ref} alt="" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", objectPosition: "center 15%", opacity: 0 }} />
      {!ready && <div className="chat-avatar-loading">…</div>}
    </div>
  );
}
