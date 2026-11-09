// The x range of run charts: the observed history, or the full planned run.

/**
 * `observed` = furthest epoch with data (fractional while a live run is mid-epoch). "So far" ends at the next whole
 * epoch past it, live or finished, so the data always reaches within one epoch of the right edge: no empty band
 * (a live run's axis then moves once per epoch, the most a chart domain may re-ease). `full` shows the whole plan.
 */
export function runHorizon(observed: number, planned: number, mode: 'observed' | 'full'): [number, number] {
  // No recorded plan: the history is all there is, so it sets the end.
  const plan = planned > 0 ? planned : Math.max(1, Math.ceil(observed));
  if (mode === 'full' || observed <= 0) return [0, plan];
  return [0, Math.max(1, Math.min(plan, Math.ceil(observed)))];
}
