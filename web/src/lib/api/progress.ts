// How far along a run is. The server reports `steps_per_epoch: 0` until the trainer has loaded the data (the first
// seconds of a real run), so step-based progress must not divide by it.
import type { Run, Step } from './types';

/**
 * Steps per epoch: the server's figure, or (while it is still 0) the steps the history shows for a finished epoch.
 * 0 while neither is known.
 */
export function stepsPerEpoch(run: Pick<Run, 'steps_per_epoch'>, steps: readonly Step[] = []): number {
  if (run.steps_per_epoch > 0) return run.steps_per_epoch;
  const first = steps[0];
  const next = first && steps.find((s) => s.epoch > first.epoch);
  return next ? next.step - first.step : 0;
}

/** Fraction of the planned run done, 0–1. Before steps per epoch are known: finished epochs only. */
export function runProgress(run: Pick<Run, 'step' | 'epoch' | 'epochs' | 'steps_per_epoch' | 'status'>): number {
  const planned = Math.max(1, run.epochs);
  const p =
    run.steps_per_epoch > 0
      ? run.step / (run.steps_per_epoch * planned)
      : (run.status === 'completed' ? planned : Math.max(0, run.epoch - 1)) / planned;
  return Math.min(1, Math.max(0, p));
}

/**
 * The epochs at the end of a run with mosaic off: `min(close_mosaic, floor(epochs / 2))`, so at least half of a short
 * run keeps mosaic (docs/MODEL.md §8.6; folo-data `close_mosaic_epochs`, which the trainer applies itself).
 */
export function closeMosaicEpochs(closeMosaic: number, epochs: number): number {
  return Math.max(0, Math.min(closeMosaic, Math.floor(Math.max(0, epochs) / 2)));
}
