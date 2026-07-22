// Shared avatar animation helpers — the mood→motion map and blend-string utilities used by BOTH the
// companion tab (AvatarPanel) and the in-chat avatar (ChatAvatar), so they stay in lock-step (parity).

export function blendString(weights: Record<string, number>): string {
  const parts = Object.entries(weights).filter(([, w]) => w > 0).map(([k, w]) => `${k}:${w}`);
  return parts.length ? parts.join(",") : "neutral:1";
}

export function dominant(weights: Record<string, number>): string {
  let best = "neutral", bw = -1;
  for (const [k, w] of Object.entries(weights)) if (w > bw) { bw = w; best = k; }
  return best;
}

// Which moods carry a special EYE SHAPE (mirrors the backend EXPRESSION_PARAMS eye_shape map). Used to
// trigger a blink when the shape changes so the swap happens behind the closing lids ("reopen as new eyes").
export const SHAPE_MOODS: Record<string, string> = {
  smug: "cat", teasing: "cat", lovestruck: "heart",
  starstruck: "star_bloom", dizzy: "swirl", mesmerized: "concentric",
  sparkly: "sparkle", ko: "x_eyes", fired_up: "flame", greedy: "money", shock: "dots", crying: "tears",
};

/** The strongest shape-bearing mood's shape name (past a small threshold), or "" — mirrors blend_params. */
export function shapeOf(weights: Record<string, number>): string {
  let best = "", bw = 0.3;
  for (const [k, w] of Object.entries(weights)) {
    const s = SHAPE_MOODS[k];
    if (s && w > bw) { bw = w; best = s; }
  }
  return best;
}

// Per-expression MOTION so moods play as animations, not stills: a mouth flap (talk), a vertical head
// bob, and a slight tilt, as functions of time. Closed-eye moods (laughing/love) come alive through the
// bounce/sway rather than staring blankly. Amplitudes are small — lively, not seasick.
// armL/armR = shoulder rotation degrees for the two movable arm sprites. By the render convention a
// NEGATIVE armL / POSITIVE armR swing the arms OUTWARD (spread); POSITIVE armL / NEGATIVE armR draw
// them INWARD (hands toward centre / hugging).
export function expressionAnim(name: string, now: number):
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
    // tense moods: a SLOW, small sway the (heavily-damped) hair can actually FOLLOW — NOT a fast shudder,
    // which shook the head distractingly AND desynced the hair (the damping filtered the fast motion, so the
    // head moved while the hair sat still = "pinned"). The tension reads through the drawn-in / spread arms.
    case "scared":    return { talk: 0, bob: s(150) * 0.5, tilt: s(200) * 0.6, ...hug(15, 3, 70) };      // hugging self, tense
    case "angry":     return { talk: 0, bob: s(170) * 0.4, tilt: s(230) * 0.55, ...spread(11, 5, 95) };  // tense, mostly still
    case "sad":       return { talk: 0, bob: 1.0 + s(1000) * 0.5, tilt: -1 + s(1200) * 0.5, ...hug(6, 2, 1200) };                // limp, drawn in
    default:          return { talk: 0, bob: s(1500) * 0.5, tilt: 0, armL: -3 + s(1500) * 2.5, armR: 3 - s(1500) * 2.5 };         // quiet breathing sway
  }
}
