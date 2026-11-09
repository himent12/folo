// The x axis of a run comparison. "By epoch" puts every run on one epoch scale; "% of run" stretches each run over its
// own length, so the shapes of a 23-epoch and a 150-epoch run can be read side by side. A finished, stopped or failed
// run's length is the epochs it ran; a live (or queued) run's is its plan, so its curve grows towards 100 % as it trains.

export interface RunLength {
  status: string;
  /** Epochs done. */
  epoch: number;
  /** Epochs planned. */
  epochs: number;
}

/** Above this ratio between the longest and the shortest run, the comparison defaults to "% of run". */
export const LENGTH_RATIO = 3;

const live = (r: RunLength) => r.status === 'running' || r.status === 'stopping' || r.status === 'queued';

/** The run's length in epochs: its plan while live, the whole epochs it ran once it has ended (at least 1). */
export function runLength(r: RunLength): number {
  return Math.max(1, live(r) ? r.epochs : Math.floor(r.epoch) || r.epochs);
}

/** Longest over shortest run length, or 1 for fewer than two runs. */
export function lengthRatio(runs: readonly RunLength[]): number {
  if (runs.length < 2) return 1;
  const ls = runs.map(runLength);
  return Math.max(...ls) / Math.min(...ls);
}

/** The default x axis: "% of run" when the runs' lengths differ by more than LENGTH_RATIO. */
export function defaultAxis(runs: readonly RunLength[]): 'epoch' | 'progress' {
  return lengthRatio(runs) > LENGTH_RATIO ? 'progress' : 'epoch';
}

/** x of `epoch` on the chosen axis. */
export function runX(epoch: number, r: RunLength, axis: 'epoch' | 'progress'): number {
  return axis === 'epoch' ? epoch : (epoch / runLength(r)) * 100;
}
