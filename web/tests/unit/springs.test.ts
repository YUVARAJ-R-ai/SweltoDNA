import { describe, expect, it } from "vitest";
import { stepSpring, project, rubberband } from "@/lib/springs";

describe("springs", () => {
  it("critically damped spring settles on target without overshoot", () => {
    const s = { x: 0, v: 0 };
    let max = 0;
    for (let i = 0; i < 240; i++) { stepSpring(s, 1, 1 / 60, 0.4, 1); max = Math.max(max, s.x); }
    expect(s.x).toBeCloseTo(1, 3);
    expect(max).toBeLessThanOrEqual(1.0001);
  });
  it("under-damped spring overshoots", () => {
    const s = { x: 0, v: 0 };
    let max = 0;
    for (let i = 0; i < 240; i++) { stepSpring(s, 1, 1 / 60, 0.4, 0.6); max = Math.max(max, s.x); }
    expect(max).toBeGreaterThan(1.02);
  });
  it("projects momentum with Apple's exponential-decay formula", () => {
    expect(project(1000, 0.998)).toBeCloseTo(499, 0);
  });
  it("rubber-bands progressively", () => {
    expect(rubberband(10, 100)).toBeLessThan(10);
    expect(rubberband(1000, 100)).toBeLessThan(100 / 0.55);
  });
});
