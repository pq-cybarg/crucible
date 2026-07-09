// Client-side idle life for the web companion face — the presentation-layer twin of
// crucible.animation.IdleAnimator. The backend owns the mood → params mapping; the browser overlays the
// small involuntary motion (saccadic gaze + a natural blink) locally so the face animates smoothly at
// screen frame-rate without hammering the server. Seeded RNG → deterministic, so it's unit-testable.

export interface IdleTick {
  readonly gaze: readonly [number, number]; // eased look-direction in [-1,1] (+x right, +y down)
  readonly blink: boolean;                   // a blink began (hold it a few frames for visibility)
}

// mulberry32: a tiny deterministic PRNG so tests get a fixed sequence (no Math.random()).
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface IdleOptions {
  readonly seed?: number;
  readonly saccadeEvery?: readonly [number, number]; // ticks between fixations
  readonly blinkEvery?: readonly [number, number];   // ticks between blinks
  readonly ease?: number;                            // 0..1 approach to a new fixation per tick
}

/** Build a stateful stepper: call the returned function once per animation tick for the next IdleTick. */
export function makeIdle(opts: IdleOptions = {}): () => IdleTick {
  const rng = mulberry32(opts.seed ?? 7);
  const [saLo, saHi] = opts.saccadeEvery ?? [8, 24];
  const [blLo, blHi] = opts.blinkEvery ?? [24, 60];
  const ease = opts.ease ?? 0.5;
  const randInt = (lo: number, hi: number): number => lo + Math.floor(rng() * (hi - lo + 1));

  let t = 0;
  let gaze: [number, number] = [0, 0];
  let target: [number, number] = [0, 0];
  let nextSaccade = randInt(saLo, saHi);
  let nextBlink = randInt(blLo, blHi);

  const newFixation = (): [number, number] => {
    const big = rng() < 0.25;
    const span = big ? 0.9 : 0.4;
    return [
      +(rng() * 2 * span - span).toFixed(3),
      +(rng() * 2 * span * 0.6 - span * 0.6).toFixed(3),
    ];
  };

  return () => {
    t += 1;
    if (t >= nextSaccade) { target = newFixation(); nextSaccade = t + randInt(saLo, saHi); }
    gaze = [
      +(gaze[0] + (target[0] - gaze[0]) * ease).toFixed(4),
      +(gaze[1] + (target[1] - gaze[1]) * ease).toFixed(4),
    ];
    const blink = t >= nextBlink;
    if (blink) nextBlink = t + randInt(blLo, blHi);
    return { gaze, blink };
  };
}
