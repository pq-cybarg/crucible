// Offline/demo companion rig. With a live backend, the server is the source of truth for the mood → face
// PARAMS mapping (crucible.expression / crucible.rigmap). This module mirrors just enough of it — the
// expression presets and the Live2D param mapping — so the web avatar window still animates on GitHub
// Pages with no node connected. When a real node is attached (isDemo() === false) none of this runs.
import type { AvatarInfo, RigFrame } from "../api";

export type ParamName = "brow" | "eye_open" | "eye_wide" | "smile" | "mouth_open" | "blush" | "head_tilt";
export type Params = Record<ParamName, number>;

const PARAM_NAMES: readonly ParamName[] = ["brow", "eye_open", "eye_wide", "smile", "mouth_open", "blush", "head_tilt"];
const NEUTRAL: Params = { brow: 0, eye_open: 1, eye_wide: 0, smile: 0, mouth_open: 0, blush: 0, head_tilt: 0 };

// expression presets (subset of the backend crucible.expression.EXPRESSIONS)
export const DEMO_EXPR: Record<string, Partial<Params>> = {
  neutral: {},
  happy: { brow: 0.2, smile: 0.9, blush: 0.2, eye_open: 0.8 },
  laughing: { brow: 0.3, smile: 1, eye_open: 0.3, mouth_open: 0.6, blush: 0.3 },
  sad: { brow: 0.4, smile: -0.7, eye_open: 0.7, head_tilt: 0.3 },
  surprised: { brow: 1, eye_wide: 0.9, mouth_open: 0.7, eye_open: 1 },
  scared: { brow: 0.6, eye_wide: 1, mouth_open: 0.5, smile: -0.4 },
  angry: { brow: -1, smile: -0.5, eye_open: 0.9 },
  curious: { brow: 0.5, head_tilt: 0.5, eye_open: 0.9 },
  love: { smile: 0.8, blush: 0.9, eye_open: 0.6, head_tilt: 0.2 },
  smug: { brow: 0.2, smile: 0.5, eye_open: 0.6, head_tilt: 0.2 },
};

// reaction word → expression preset (subset of crucible.expression.REACTION_TO_EXPRESSION)
export const DEMO_REACTION: Record<string, string> = {
  funny: "laughing", cute: "happy", wholesome: "happy", beautiful: "love", romantic: "love",
  exciting: "surprised", surprising: "surprised", scary: "scared", sad: "sad", sus: "smug",
  confusing: "curious", calm: "neutral",
};

function preset(name: string): Params {
  return { ...NEUTRAL, ...(DEMO_EXPR[name] ?? {}) };
}

const clamp = (v: number, lo = 0, hi = 1): number => Math.max(lo, Math.min(hi, v));

/** Weighted, order-independent average of expression presets — the demo twin of expression.blend_params. */
export function blendParamsDemo(weights: Readonly<Record<string, number>>): Params {
  const items = Object.entries(weights).filter(([, w]) => w > 0);
  if (items.length === 0) return { ...NEUTRAL };
  const total = items.reduce((s, [, w]) => s + w, 0);
  const out = { ...NEUTRAL };
  for (const k of PARAM_NAMES) out[k] = items.reduce((s, [n, w]) => s + preset(n)[k] * w, 0) / total;
  return out;
}

/** Continuous params + gaze/blink → Live2D standard params — the demo twin of rigmap.to_live2d. */
export function paramsToLive2d(p: Params, gaze: readonly [number, number] = [0, 0], blink = 0): Record<string, number> {
  const [gx, gy] = gaze;
  const eye = clamp((1 - blink) * Math.min(1, p.eye_open + 0.4 * p.eye_wide));
  return {
    ParamEyeLOpen: eye, ParamEyeROpen: eye,
    ParamEyeBallX: gx, ParamEyeBallY: -gy,
    ParamBrowLY: p.brow, ParamBrowRY: p.brow,
    ParamMouthOpenY: clamp(p.mouth_open), ParamMouthForm: clamp(p.smile, -1, 1),
    ParamCheek: clamp(p.blush), ParamAngleZ: clamp(p.head_tilt, -1, 1) * 30,
    ParamAngleX: gx * 30, ParamAngleY: -gy * 30,
  };
}

export function demoRigFrame(
  weights: Readonly<Record<string, number>>, gaze?: readonly [number, number], blink = 0,
): RigFrame {
  const params = blendParamsDemo(weights);
  const g: readonly [number, number] = gaze ?? [0, 0];
  return {
    params, gaze: g, blink: clamp(blink),
    arkit: {}, vrm: {}, live2d: paramsToLive2d(params, g, blink),
  };
}

export const DEMO_AVATAR: AvatarInfo = {
  name: "kiri", kind: "sprites", size: [48, 60],
  expressions: ["neutral", "happy", "laughing", "sad", "surprised", "angry", "curious", "love", "smug"],
  layers: [
    { id: "skin", part: "skin", protected: false, states: ["base"], default_state: "base", pos: [0, 0], mirror: false, spacing: 0 },
    { id: "eyes", part: "eyes", protected: false, states: ["open", "closed", "wide"], default_state: "open", pos: [0, 0], mirror: false, spacing: 0 },
    { id: "pupils", part: "pupils", protected: false, states: ["on", "off"], default_state: "on", pos: [0, 0], mirror: false, spacing: 0 },
    { id: "mouth", part: "mouth", protected: false, states: ["closed", "smile", "open", "frown"], default_state: "closed", pos: [0, 0], mirror: false, spacing: 0 },
    { id: "hair", part: "hair", protected: false, states: ["base"], default_state: "base", pos: [0, 0], mirror: false, spacing: 0 },
  ],
};
