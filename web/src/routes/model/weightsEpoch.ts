// Which epoch a run model's weights are from (docs/API.md, Models): `<run>-best` is the run's best epoch, `<run>-last`
// (id `m_<run>-last`) its last finished one. When they are the same epoch, the two are the same weights, and each
// says so.
//
// How many epochs the run finished (`finishedEpochs`), never the planned count: a completed run may have stopped
// early (patience). A completed or stopped run's `Run.epoch` says it, checkpoint-backed; for any other run its
// recorded history (`GET /api/runs/{id}/metrics`: one `Epoch` per validated epoch) does when `Run.epoch` agrees.

import type { Epoch, Model, Run } from '$lib/api/types';

/** True for a run's `last.folo` model. */
export const isLast = (m: Pick<Model, 'id'>) => m.id.endsWith('-last');

/**
 * The epochs a run finished, null when not known. A `completed` or `stopped` run's `epoch` is the server's count of
 * finished epochs, which it brings up to the run's `weights/last.folo` after a crash between writing that checkpoint
 * and its history line (docs/API.md, Runs; `runs/history.rs` `reconcile`), so it is the authority and the history may
 * lag it. Any other run's `epoch` may be one in progress or the one it failed in, so its history's highest validated
 * epoch counts, but only when `epoch` is not past it: past it, the checkpoint may hold one more (a crash or a failed
 * write after saving it), and that is not known.
 */
export function finishedEpochs(run: Pick<Run, 'status' | 'epoch'>, epochs: readonly Pick<Epoch, 'epoch'>[] | null | undefined): number | null {
  if (run.status === 'completed' || run.status === 'stopped') return run.epoch > 0 ? run.epoch : null;
  if (!epochs) return null;
  let n = 0;
  for (const e of epochs) if (e.epoch > n) n = e.epoch;
  return n > 0 && run.epoch <= n ? n : null;
}

/**
 * The Details' "Trained for": the epochs the model's run finished, of the planned ones when it ended early (a model's
 * `trained_with.epochs` is the planned count, docs/API.md). "50 epochs at 640 px", "1 of 50 epochs at 320 px"; while
 * that is not known (loading, the run or its history did not load, a checkpoint copied out of its run) only the plan,
 * labelled so: "50 epochs planned at 320 px".
 */
export function trainedFor(m: Pick<Model, 'run_id' | 'trained_with'>, run: Pick<Run, 'id'> | null | undefined, done: number | null): string | null {
  const t = m.trained_with;
  if (!t) return null;
  const planned = t.epochs;
  const unit = (n: number) => (n === 1 ? 'epoch' : 'epochs');
  const ran = run && m.run_id === run.id ? done : null;
  if (ran === null) return `${planned.toLocaleString('en-US')} ${unit(planned)} planned at ${t.imgsz} px`;
  if (ran < planned) return `${ran.toLocaleString('en-US')} of ${planned.toLocaleString('en-US')} ${unit(planned)} at ${t.imgsz} px`;
  return `${planned.toLocaleString('en-US')} ${unit(planned)} at ${t.imgsz} px`;
}

/**
 * "best epoch 1 of 5", "last epoch, 5 of 5, same weights as best", or "last finished epoch, 2 of 5" when the run
 * ended before its planned end; `done` is the epochs it finished (`finishedEpochs`). Null without the run, or when
 * the server did not say which epoch was best.
 */
export function weightsEpoch(
  m: Pick<Model, 'id' | 'run_id'>,
  run: Pick<Run, 'id' | 'epochs' | 'best'> | null | undefined,
  done: number | null,
): string | null {
  if (!run || m.run_id !== run.id) return null;
  const best = run.best?.epoch ?? null;
  if (isLast(m)) {
    if (done === null) return 'last finished epoch';
    const same = best === done ? ', same weights as best' : '';
    return done >= run.epochs ? `last epoch, ${done} of ${run.epochs}${same}` : `last finished epoch, ${done} of ${run.epochs}${same}`;
  }
  if (best === null) return null;
  return `best epoch ${best} of ${run.epochs}${done !== null && best === done ? ', same weights as last' : ''}`;
}
