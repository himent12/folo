// A failed run in plain words: its outcome callout leads with what went wrong by the run's error code (docs/API.md,
// Errors), the raw message goes under Details. Out of GPU memory has its own callout (lib/oom).

import type { Run } from '$lib/api/types';

/**
 * The epoch the run failed in: the one its last step was in (older servers reported the last finished epoch), or
 * null when it failed before its first step.
 */
export function failedEpoch(run: Pick<Run, 'epoch' | 'step' | 'steps_per_epoch'>): number | null {
  const fromSteps = run.steps_per_epoch > 0 && run.step > 0 ? Math.ceil(run.step / run.steps_per_epoch) : 0;
  const e = Math.max(run.epoch, fromSteps);
  return e > 0 ? e : null;
}

/** The callout's title: "Failed in epoch 2", "Failed before training started". */
export function failedTitle(run: Pick<Run, 'epoch' | 'step' | 'steps_per_epoch'>): string {
  const e = failedEpoch(run);
  return e === null ? 'Failed before training started' : `Failed in epoch ${e}`;
}

/**
 * What went wrong, in a sentence, for the codes a training run fails with; null for any other (the server's message
 * is then the cause). `dataset` is the dataset's name; `trained`: it failed after its first step (one that failed
 * before is "Failed before training started", so its cause never says "during training").
 */
export function failureSummary(code: string | undefined, dataset: string, trained = true): string | null {
  switch (code) {
    case 'data_error':
      return trained ? `Files of dataset ${dataset} went missing or changed during training.` : `The files of dataset ${dataset} are missing or could not be read.`;
    case 'io_error':
      return trained ? 'Reading or writing a file failed during training.' : 'Reading or writing a file failed.';
    case 'training_failed':
      return 'Training diverged or produced no usable batch.';
    case 'not_supported':
      return 'This machine cannot train with these settings.';
    case 'internal':
      return 'The trainer failed unexpectedly (a bug in FOLO).';
    case 'cloud_boot_failed':
      return 'The cloud instance never started a usable worker.';
    case 'cloud_sync_failed':
      return 'Copying data to or from the cloud worker failed.';
    case 'cloud_instance_lost':
      return 'The cloud instance was gone before the run ended.';
    default:
      return null;
  }
}
