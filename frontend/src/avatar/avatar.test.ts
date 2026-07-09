import { describe, expect, it } from "vitest";
import { blendParamsDemo, demoRigFrame, paramsToLive2d } from "./demoRig";
import { browOffset, eyeGeometry, mouthPath, readFace } from "./face";
import { makeIdle } from "./idle";

describe("demo rig (offline mood → params → live2d)", () => {
  it("blends expression presets by weight, order-independent + normalized", () => {
    const half = blendParamsDemo({ happy: 0.5, surprised: 0.5 });
    // 50/50 sits between the two presets on smile (happy +) and eye_wide (surprised +)
    expect(half.smile).toBeGreaterThan(0);
    expect(half.eye_wide).toBeGreaterThan(0);
    const ab = blendParamsDemo({ happy: 1, angry: 1 });
    const ba = blendParamsDemo({ angry: 1, happy: 1 });
    expect(ab).toEqual(ba);                                   // order doesn't matter
    const scaled = blendParamsDemo({ happy: 10, angry: 10 });
    expect(scaled.smile).toBeCloseTo(ab.smile, 6);           // only the ratio matters
    expect(blendParamsDemo({}).eye_open).toBe(1);            // empty → neutral
  });

  it("maps params + gaze/blink to Live2D standard params", () => {
    const l = paramsToLive2d(blendParamsDemo({ happy: 1 }), [0.5, -0.5], 0);
    expect(l["ParamMouthForm"]).toBeGreaterThan(0);          // smiling form
    expect(l["ParamEyeBallX"]).toBe(0.5);
    expect(l["ParamEyeBallY"]).toBe(0.5);                    // EyeBallY = -gy (gy=-0.5 → +0.5, looks up)
    expect(paramsToLive2d(blendParamsDemo({ neutral: 1 }), [0, 0], 1)["ParamEyeLOpen"]).toBe(0);  // blink shut
  });

  it("demoRigFrame produces a full frame with live2d populated", () => {
    const f = demoRigFrame({ surprised: 1 }, [0.3, 0]);
    expect(f.gaze).toEqual([0.3, 0]);
    expect(f.live2d["ParamMouthOpenY"]).toBeGreaterThan(0);  // surprised opens the mouth
    expect(f.params["eye_wide"]).toBeGreaterThan(0);
  });
});

describe("face geometry (params → drawable numbers)", () => {
  it("reads live2d params with clamping + defaults", () => {
    const f = readFace({ ParamEyeLOpen: 0.3, ParamEyeBallX: 5, ParamMouthForm: -2, ParamAngleZ: 45 });
    expect(f.eyeOpen).toBe(0.3);
    expect(f.eyeballX).toBe(1);                              // clamped to [-1,1]
    expect(f.mouthForm).toBe(-1);
    expect(f.angleZ).toBe(30);                               // clamped to ±30
    expect(readFace({}).eyeOpen).toBe(1);                    // sensible default
  });

  it("eye opening shrinks toward a shut lid as eyeOpen → 0", () => {
    const open = eyeGeometry(44, 66, 12, readFace({ ParamEyeLOpen: 1 }));
    const shut = eyeGeometry(44, 66, 12, readFace({ ParamEyeLOpen: 0 }));
    expect(shut.ry).toBeLessThan(open.ry);
    expect(shut.shut).toBe(true);
    expect(open.shut).toBe(false);
  });

  it("pupil follows gaze and stays inside the eye white", () => {
    const right = eyeGeometry(44, 66, 12, readFace({ ParamEyeBallX: 1 }));
    const left = eyeGeometry(44, 66, 12, readFace({ ParamEyeBallX: -1 }));
    expect(right.pupilX).toBeGreaterThan(left.pupilX);
    expect(Math.abs(right.pupilX - 44)).toBeLessThanOrEqual(12);   // within the white radius
  });

  it("mouth path is a valid closed path; brow rises when raised", () => {
    const d = mouthPath(60, 98, 12, readFace({ ParamMouthForm: 1, ParamMouthOpenY: 0.5 }));
    expect(d.startsWith("M ")).toBe(true);
    expect(d.trim().endsWith("Z")).toBe(true);
    expect(browOffset(12, readFace({ ParamBrowLY: 1 }))).toBeLessThan(0);   // raised brow = up on screen
    expect(browOffset(12, readFace({ ParamBrowLY: -1 }))).toBeGreaterThan(0);
  });
});

describe("client idle (saccades + blink)", () => {
  it("is deterministic for a seed and stays in range", () => {
    const a = Array.from({ length: 80 }, makeIdle({ seed: 3 }));
    const b = Array.from({ length: 80 }, makeIdle({ seed: 3 }));
    expect(a).toEqual(b);
    for (const t of a) {
      expect(t.gaze[0]).toBeGreaterThanOrEqual(-1);
      expect(t.gaze[0]).toBeLessThanOrEqual(1);
    }
  });

  it("gaze roves and blinks occur over time", () => {
    const step = makeIdle({ seed: 7 });
    const ticks = Array.from({ length: 200 }, step);
    const xs = ticks.map((t) => t.gaze[0]);
    expect(Math.max(...xs) - Math.min(...xs)).toBeGreaterThan(0.2);   // saccades move the eyes
    expect(ticks.some((t) => t.blink)).toBe(true);                    // blinks happen
  });

  it("different seeds diverge", () => {
    const a = Array.from({ length: 40 }, makeIdle({ seed: 1 })).map((t) => t.gaze[0]);
    const b = Array.from({ length: 40 }, makeIdle({ seed: 2 })).map((t) => t.gaze[0]);
    expect(a).not.toEqual(b);
  });
});
