// The x range of run charts: the observed history, or the full planned run.

/**
 * `observed` = furthest epoch with data (fractional while a live run is mid-epoch). "So far" ends at the next whole
 * epoch past it, live or finished, so the data always reaches within one epoch of the right edge: no empty band
 * (a live run's axis then moves once per epoch, the most a chart domain may re-ease). `full` shows the whole plan.
 */
export function runHorizon(observed: number, planned: number, mode: 'observed' | 'full'): [number, number] {
  if (mode === 'full' || observed <= 0) return [0, Math.max(1, planned)];
  return [0, Math.max(1, Math.min(Math.max(1, planned), Math.ceil(observed)))];
}
