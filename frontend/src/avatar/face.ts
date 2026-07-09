// Pure geometry for the SVG companion face — turn the engine-agnostic Live2D-style params (the SAME
// param stream the backend serves and a real Live2D/VRM rig would consume) into concrete drawing numbers.
// Kept separate from the React component so it's unit-testable: params in → shape values out, no DOM.

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));
const def = (l2d: Record<string, number>, k: string, d: number): number => (typeof l2d[k] === "number" ? l2d[k] : d);

export interface FaceParams {
  readonly eyeOpen: number;   // 0 shut … 1 open
  readonly eyeballX: number;  // -1 left … +1 right
  readonly eyeballY: number;  // -1 down … +1 up  (Live2D convention: +up)
  readonly brow: number;      // -1 furrowed … +1 raised
  readonly mouthOpen: number; // 0 … 1
  readonly mouthForm: number; // -1 frown … +1 smile
  readonly cheek: number;     // 0 … 1 blush
  readonly angleZ: number;    // head tilt, degrees (-30 … 30)
}

/** Read a Live2D param record into the normalized face bundle (with sane defaults for missing keys). */
export function readFace(l2d: Record<string, number>): FaceParams {
  return {
    eyeOpen: clamp(def(l2d, "ParamEyeLOpen", 1), 0, 1),
    eyeballX: clamp(def(l2d, "ParamEyeBallX", 0), -1, 1),
    eyeballY: clamp(def(l2d, "ParamEyeBallY", 0), -1, 1),
    brow: clamp(def(l2d, "ParamBrowLY", 0), -1, 1),
    mouthOpen: clamp(def(l2d, "ParamMouthOpenY", 0), 0, 1),
    mouthForm: clamp(def(l2d, "ParamMouthForm", 0), -1, 1),
    cheek: clamp(def(l2d, "ParamCheek", 0), 0, 1),
    angleZ: clamp(def(l2d, "ParamAngleZ", 0), -30, 30),
  };
}

export interface EyeGeometry {
  readonly cx: number; readonly cy: number;   // eye centre
  readonly rx: number; readonly ry: number;   // lid opening (ry → ~0 when shut)
  readonly pupilX: number; readonly pupilY: number; // pupil centre (follows gaze, clamped inside)
  readonly pupilR: number;
  readonly shut: boolean;
}

/** One eye's geometry given the centre, a base radius, and the face params (gaze + open amount). */
export function eyeGeometry(cx: number, cy: number, baseR: number, f: FaceParams): EyeGeometry {
  const ry = baseR * (0.16 + 0.84 * f.eyeOpen);          // never fully zero so a lid line stays visible
  const pupilR = baseR * 0.42;
  const travel = baseR - pupilR - 1;                     // keep the pupil inside the white
  return {
    cx, cy, rx: baseR, ry,
    pupilX: cx + f.eyeballX * travel,
    pupilY: cy - f.eyeballY * travel,                    // eyeballY is +up → subtract for screen Y
    pupilR,
    shut: f.eyeOpen < 0.12,
  };
}

/**
 * Mouth as a quadratic-curve path around a centre. `mouthForm` bends the corners up (smile) or down
 * (frown); `mouthOpen` gives it height (an open/talking mouth). Returns an SVG path `d` string.
 */
export function mouthPath(cx: number, cy: number, halfWidth: number, f: FaceParams): string {
  const corner = -f.mouthForm * halfWidth * 0.5;          // corners rise for a smile (negative = up on screen)
  const open = f.mouthOpen * halfWidth * 0.9;
  const lx = cx - halfWidth, rx = cx + halfWidth;
  const topDip = cy + corner + (f.mouthForm > 0 ? halfWidth * 0.12 : 0);   // upper lip curve control
  const botDip = cy + corner + open + halfWidth * 0.1;                      // lower lip curve control
  // upper lip: left corner → control → right corner; lower lip back with more drop when open
  return `M ${lx.toFixed(1)} ${(cy + corner).toFixed(1)} ` +
    `Q ${cx.toFixed(1)} ${topDip.toFixed(1)} ${rx.toFixed(1)} ${(cy + corner).toFixed(1)} ` +
    `Q ${cx.toFixed(1)} ${botDip.toFixed(1)} ${lx.toFixed(1)} ${(cy + corner).toFixed(1)} Z`;
}

/** Vertical brow offset in px: raised brows sit higher (negative screen-y), furrowed sit lower. */
export function browOffset(baseR: number, f: FaceParams): number {
  return -f.brow * baseR * 0.5;
}
