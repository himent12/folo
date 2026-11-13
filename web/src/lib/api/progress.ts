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
 * The run's progress as people read it, "37%": whole percents, rounded down (a run is never shown further along than it
 * is). Every screen that prints progress says it through this one value (Overview's live card, the run page's epoch
 * row), so two screens of one run never disagree in a decimal the step stream moves several times a second.
 */
export function progressText(run: Pick<Run, 'step' | 'epoch' | 'epochs' | 'steps_per_epoch' | 'status'>): string {
  return `${Math.floor(runProgress(run) * 100 + 1e-9)}%`;
}

/**
 * Whether a run's time left has settled enough to show: once its first epoch's steps are done (halfway through a
 * one-epoch run). The first epoch carries the start (loading the data, compiling the GPU pipelines), so an estimate made
 * during it swings by minutes; until then every screen says "estimating time left…" (the run page, Overview's Training
 * card, the sidebar's live card), so no two disagree.
 */
export function etaSettled(run: Pick<Run, 'step' | 'epoch' | 'epochs' | 'steps_per_epoch' | 'status'>): boolean {
  if (run.epoch > 1 || (run.steps_per_epoch > 0 && run.step >= run.steps_per_epoch)) return true;
  return run.epochs === 1 && runProgress(run) >= 0.5;
}

/**
 * A run that completed before its last planned epoch: the trainer's early stop (`patience`, docs/MODEL.md §8.1: no
 * better validation score for that many epochs) ended it. Every screen that says "Epoch 256 of 2000" for a finished
 * run says why with this.
 */
export function stoppedEarly(run: Pick<Run, 'status' | 'epoch' | 'epochs'>): boolean {
  return run.status === 'completed' && run.epoch > 0 && run.epoch < run.epochs;
}

/**
 * A run's time left once it has settled ([`etaSettled`]), else null: what another screen quotes for it (a queued run's
 * wait, the queue panel, New run's queue note), so none of them says "23s left" while the run's own page is still
 * estimating.
 */
export function settledEta(run: Pick<Run, 'step' | 'epoch' | 'epochs' | 'steps_per_epoch' | 'status' | 'eta_s'>): number | null {
  return run.eta_s !== null && etaSettled(run) ? run.eta_s : null;
}

/** What a live run says about its time left: "3m left", or "estimating time left…" until it has settled. */
export const ETA_PENDING = 'estimating time left…';

/**
 * The epochs at the end of a run with mosaic off: `min(close_mosaic, floor(epochs / 2))`, so at least half of a short
 * run keeps mosaic (docs/MODEL.md §8.6; folo-data `close_mosaic_epochs`, which the trainer applies itself).
 */
export function closeMosaicEpochs(closeMosaic: number, epochs: number): number {
  return Math.max(0, Math.min(closeMosaic, Math.floor(Math.max(0, epochs) / 2)));
}

/**
 * Seconds a run has trained: from its start to its end, or to `now` while it is active, less the time it sat ended
 * before a resume (`paused_s`, which the server records). Every screen that says how long a run took or has taken
 * says this one value (the Runs list, the run page's outcome line and metric strip). Null before it started.
 */
export function trainingSeconds(r: Pick<Run, 'started' | 'finished' | 'paused_s'>, now = Date.now()): number | null {
  if (!r.started) return null;
  return Math.max(0, ((r.finished ? Date.parse(r.finished) : now) - Date.parse(r.started)) / 1000 - (r.paused_s ?? 0));
}
