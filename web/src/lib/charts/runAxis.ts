// The x axis of a run comparison. "By epoch" puts every run on one epoch scale; "By progress" puts each run over its
// own plan (% of planned epochs), so the shapes of a 50-epoch and a 150-epoch schedule can be read side by side. The
// denominator is the plan for every status: a run that failed at epoch 23 of 100 ends at 23 %, never stretched to
// 100 %, and a run that stops keeps the x positions it had while it trained.

export interface RunLength {
  status: string;
  /** Epochs done. */
  epoch: number;
  /** Epochs planned. */
  epochs: number;
}

/** Above this ratio between the longest and the shortest plan, the comparison defaults to "By progress". */
export const LENGTH_RATIO = 3;

/** The run's length in epochs: its plan, whatever its status (the epochs it ran when there is no plan; at least 1). */
export function runLength(r: RunLength): number {
  return Math.max(1, r.epochs > 0 ? r.epochs : Math.floor(r.epoch));
}

/** Longest over shortest planned length, or 1 for fewer than two runs. */
export function lengthRatio(runs: readonly RunLength[]): number {
  if (runs.length < 2) return 1;
  const ls = runs.map(runLength);
  return Math.max(...ls) / Math.min(...ls);
}

/** The default x axis: "By progress" when the runs' plans differ by more than LENGTH_RATIO. */
export function defaultAxis(runs: readonly RunLength[]): 'epoch' | 'progress' {
  return lengthRatio(runs) > LENGTH_RATIO ? 'progress' : 'epoch';
}

/** x of `epoch` on the chosen axis. */
export function runX(epoch: number, r: RunLength, axis: 'epoch' | 'progress'): number {
  return axis === 'epoch' ? epoch : (epoch / runLength(r)) * 100;
}

/** How a run's line ends: still training, stopped or failed before its plan; null for a completed (or queued) run. */
export function runEnd(status: string): 'live' | 'stopped' | 'failed' | null {
  if (status === 'running' || status === 'stopping') return 'live';
  if (status === 'stopped' || status === 'failed') return status;
  return null;
}
