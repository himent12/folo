// Which epoch a run model's weights are from (docs/API.md, Models): `<run>-best` is the run's best epoch, `<run>-last`

/**
 * The epochs `run` finished (docs/API.md, Runs: `epoch` is the one in progress, once ended the last one finished, or the
 * one a failed run failed in).
 */
export function finishedEpochs(run: Pick<Run, 'status' | 'epoch'>): number {
  return run.status === 'completed' || run.status === 'stopped' ? run.epoch : Math.max(0, run.epoch - 1);
}

/**
 * The Details' "Trained for": the epochs the model's run trained, and the planned ones when it ended early (a model's
 * `trained_with.epochs` is the planned count, docs/API.md). "50 epochs at 640 px", "1 of 50 epochs at 320 px".
 */
export function trainedFor(
  m: Pick<Model, 'run_id' | 'trained_with'>,
  run: Pick<Run, 'id' | 'status' | 'epoch' | 'epochs'> | null | undefined,
): string | null {
  const t = m.trained_with;
  if (!t) return null;
  const planned = t.epochs;
  const unit = (n: number) => (n === 1 ? 'epoch' : 'epochs');
  const ran = run && m.run_id === run.id && run.status !== 'completed' ? finishedEpochs(run) : null;
  if (ran !== null && ran > 0 && ran < planned) return `${ran.toLocaleString('en-US')} of ${planned.toLocaleString('en-US')} ${unit(planned)} at ${t.imgsz} px`;
  return `${planned.toLocaleString('en-US')} ${unit(planned)} at ${t.imgsz} px`;
}

// (id `m_<run>-last`) its last finished one. When they are the same epoch, the two are the same weights, and each
// says so.

import type { Model, Run } from '$lib/api/types';

/** True for a run's `last.folo` model. */
export const isLast = (m: Pick<Model, 'id'>) => m.id.endsWith('-last');

/**
 * "best epoch 1 of 5", "last epoch, 5 of 5, same weights as best", or "last finished epoch, 2 of 5" when the run
 * ended before its end; null without the run, or when the server did not say which epoch was best.
 */
export function weightsEpoch(
  m: Pick<Model, 'id' | 'run_id'>,
  run: Pick<Run, 'id' | 'status' | 'epoch' | 'epochs' | 'best'> | null | undefined,
): string | null {
  if (!run || m.run_id !== run.id) return null;
  const done = run.status === 'completed';
  const best = run.best?.epoch ?? null;
  if (isLast(m)) {
    if (!done) {
      const n = finishedEpochs(run);
      return n > 0 ? `last finished epoch, ${n} of ${run.epochs}` : 'last finished epoch';
    }
    return `last epoch, ${run.epochs} of ${run.epochs}${best === run.epochs ? ', same weights as best' : ''}`;
  }
  if (best === null) return null;
  return `best epoch ${best} of ${run.epochs}${done && best === run.epochs ? ', same weights as last' : ''}`;
}
