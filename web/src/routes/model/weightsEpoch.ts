// Which epoch a run model's weights are from (docs/API.md, Models): `<run>-best` is the run's best epoch, `<run>-last`
// (id `m_<run>-last`) its last finished one. When they are the same epoch, the two are the same weights, and each
// says so.

import type { Model, Run } from '$lib/api/types';

/** True for a run's `last.folo` model. */
export const isLast = (m: Pick<Model, 'id'>) => m.id.endsWith('-last');

/**
 * "best epoch 1 of 5", "last epoch, 5 of 5, same weights as best", or "last finished epoch" when the run stopped
 * before its end; null without the run, or when the server did not say which epoch was best.
 */
export function weightsEpoch(m: Pick<Model, 'id' | 'run_id'>, run: Pick<Run, 'id' | 'status' | 'epochs' | 'best'> | null | undefined): string | null {
  if (!run || m.run_id !== run.id) return null;
  const done = run.status === 'completed';
  const best = run.best?.epoch ?? null;
  if (isLast(m)) {
    if (!done) return 'last finished epoch';
    return `last epoch, ${run.epochs} of ${run.epochs}${best === run.epochs ? ', same weights as best' : ''}`;
  }
  if (best === null) return null;
  return `best epoch ${best} of ${run.epochs}${done && best === run.epochs ? ', same weights as last' : ''}`;
}
