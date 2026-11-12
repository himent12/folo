// What a run's epochs were scored on once its dataset is gone (v-data-runs N7): the dataset no longer says whether it
// had validation images, but the trainer's log does. It warns once, at the start, when it validates on the training
// split (folo_vision::train), so a run whose log has no such line validated on the validation split, and its panels
// keep "Validation …" instead of turning into "Evaluation …" when the dataset is removed.

import type { EvalSource } from '$lib/evaluated';
import type { LogLine } from '$lib/api/types';

/** How the trainer's warning starts. */
const TRAIN_SPLIT = 'the dataset has no validation images';

/** The scores' source of a run whose dataset `name` (as recorded with the run) was removed, from its `logs`. */
export function goneEvalOn(name: string | null, logs: readonly LogLine[]): EvalSource {
  const onTrain = logs.some((l) => l.level === 'warn' && l.message.startsWith(TRAIN_SPLIT));
  return {
    datasetId: null,
    datasetName: name,
    split: onTrain ? 'train' : 'val',
    fallback: onTrain ? { requested: 'val', reason: 'empty_split', message: `${name ?? 'the dataset'} has no validation images` } : null,
  };
}
