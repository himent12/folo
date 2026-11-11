// How Overview names a run's dataset: never by its raw id (`ds_cbb5`), which means nothing to people (shell F15).

import type { Dataset } from '$lib/api/types';

/**
 * The dataset's name; `undefined` while the datasets load (the caller shows a placeholder); "deleted dataset" when
 * they loaded without it; null when they failed to load (nothing to name it by, so the caller leaves it out).
 */
export function datasetLabel(
  id: string,
  datasets: ReadonlyMap<string, Pick<Dataset, 'name'>> | undefined,
  failed: boolean,
): string | null | undefined {
  if (!datasets) return failed ? null : undefined;
  return datasets.get(id)?.name ?? 'deleted dataset';
}
