/**
 * Apple-style springs: two designer-facing parameters instead of mass/stiffness/damping.
 * `response` is roughly how fast the value reaches the target (seconds); `damping` 1 = no overshoot.
 */
export interface SpringState { x: number; v: number }

export function stepSpring(s: SpringState, target: number, dt: number, response: number, damping = 1): void {
  const k = (2 * Math.PI / response) ** 2;
  const c = (4 * Math.PI * damping) / response;
  for (let t = dt; t > 0; t -= 1 / 240) {
    const h = Math.min(t, 1 / 240);
    s.v += (-k * (s.x - target) - c * s.v) * h;
    s.x += s.v * h;
  }
}

/** Where a flick would come to rest (Apple's sample: exponential decay, not v²/2a). */
export function project(velocityPerSecond: number, decelerationRate = 0.998): number {
  return (velocityPerSecond / 1000) * decelerationRate / (1 - decelerationRate);
}

/** Progressive resistance past a boundary. */
export function rubberband(overshoot: number, dimension: number, constant = 0.55): number {
  return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot));
}
