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

/** Above this ratio between the epochs the longest run trained and those of every other run, see `defaultAxis`. */
export const SPAN_RATIO = 2;

/**
 * Why the comparison defaults to "By progress", or null when "By epoch" is the default:
 * - `plans`: the plans differ by more than LENGTH_RATIO (a 30- and a 150-epoch schedule);
 * - `span`: one run trained more than SPAN_RATIO times the epochs of every other, so an epoch axis (to the longest
 *   plan) squeezes the others into its first part, and their shorter plans give them more of a progress axis. Round
 *   24's comparison ran to epoch 150 for one run while three ended by 61: they filled 40 % of the plot, 61 % by
 *   progress. A run that failed early on the same plan is not helped by progress, so it does not switch the axis.
 */
export function axisReason(runs: readonly RunLength[]): 'plans' | 'span' | null {
  if (lengthRatio(runs) > LENGTH_RATIO) return 'plans';
  if (runs.length < 2) return null;
  const done = runs.map((r) => Math.max(0, r.epoch));
  const top = Math.max(...done);
  const i = done.indexOf(top);
  const others = runs.filter((_, k) => k !== i);
  const plan = Math.max(...runs.map(runLength));
  const rest = Math.max(...others.map((r) => Math.max(0, r.epoch)));
  return top > SPAN_RATIO * rest && others.some((r) => runLength(r) < plan) ? 'span' : null;
}

/** The default x axis: "By progress" when `axisReason` gives one, else "By epoch". */
export function defaultAxis(runs: readonly RunLength[]): 'epoch' | 'progress' {
  return axisReason(runs) ? 'progress' : 'epoch';
}

/** The header's note on an axis chosen for the reader: why it is "By progress". */
export const AXIS_REASON = { plans: 'plans differ in length', span: 'one run trained far longer' } as const;

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
