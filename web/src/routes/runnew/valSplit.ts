// A dataset with training images and no validation split: the trainer then scores each epoch on the training images
// (docs/API.md, "Split fallback"), so its mAP measures fit, not held-out accuracy, and New run says so before Start.

import type { Dataset } from '$lib/api/types';

/** The line under the dataset picker for a train-only dataset, else null. */
export function noValNote(d: Pick<Dataset, 'splits'> | undefined): string | null {
  if (!d || !(d.splits.train > 0) || d.splits.val > 0) return null;
  return 'No validation images: each epoch is scored on the training images, so its mAP shows fit to them, not accuracy on new images. Add a val split for a held-out score.';
}
