// How a list names a run's dataset (Overview, the Runs list): never by its raw id (`ds_cbb5`), which means nothing to
// people (shell F15). A run records its dataset's name when it is created (`Run.dataset`), so one whose dataset was
// removed still says which it was, marked as the run page marks it: "shapes (removed)".

import type { Dataset, Run } from '$lib/api/types';

/**
 * The dataset's name; `undefined` while the datasets load (the caller shows a placeholder); the recorded name marked
 * "(removed)" when they loaded without it ("removed dataset" for a run from a server that recorded none); when they
 * failed to load, the recorded name, else null (nothing to name it by, so the caller leaves it out).
 */
export function runDatasetLabel(
  run: Pick<Run, 'config' | 'dataset'>,
  datasets: ReadonlyMap<string, Pick<Dataset, 'name'>> | undefined,
  failed: boolean,
): string | null | undefined {
  const recorded = run.dataset?.name;
  if (!datasets) return failed ? (recorded ?? null) : undefined;
  const name = datasets.get(run.config.dataset_id)?.name;
  if (name !== undefined) return name;
  return recorded !== undefined ? `${recorded} (removed)` : 'removed dataset';
}
