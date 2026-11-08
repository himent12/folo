// The x range of run charts: the observed history (with headroom while a run is live), or the full planned run.

/** Rounds up to a tick-friendly epoch count: multiples of 5 below 50, of 10 above. */
function niceCeil(x: number): number {
  const step = x <= 50 ? 5 : 10;
  return Math.max(step, Math.ceil(x / step) * step);
}

/**
 * `observed` = furthest epoch with data. Live runs get ~10% headroom (in stable steps, so the axis does not jitter
 * every epoch); finished runs end at their last observed epoch; `full` shows the whole plan.
 */
export function runHorizon(observed: number, planned: number, live: boolean, mode: 'observed' | 'full'): [number, number] {
  if (mode === 'full' || observed <= 0) return [0, Math.max(1, planned)];
  if (live) return [0, Math.min(planned, niceCeil(observed * 1.1))];
  return [0, Math.max(1, Math.ceil(observed))];
}
