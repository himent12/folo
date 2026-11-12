// What deleting a run deletes, in words (docs/API.md, DELETE /api/runs/{id}): the server removes the run's folder, its
// weights included, so the models saved from them go too. The Run page and the Runs list confirm with these.

import type { Model, Run } from '$lib/api/types';
import { bytes } from '$lib/format';

/** The models saved from `run`'s weights (its best and last), the ones its delete takes with it. */
export const runModels = (run: Pick<Run, 'id'>, models: readonly Model[]) => models.filter((m) => m.run_id === run.id && (m.source ?? 'run') === 'run');

/**
 * The confirm's description and button. `models` is a list fetched since the confirm opened (./DeleteRun), never an
 * older one: a run gets its models when it ends, so a list from before can miss them. Undefined until it arrives (or
 * when it fails): the words then say the weights and models go without counting them, and never that there are none.
 */
export function deleteWords(run: Pick<Run, 'id'>, models: readonly Model[] | undefined, units: 'binary' | 'metric' = 'binary') {
  if (!models)
    return {
      description: "Deletes the run and its metric history. Its weights and models are deleted too. This can't be undone.",
      button: 'Delete run',
    };
  const mine = runModels(run, models);
  if (!mine.length) return { description: "Deletes the run and its metric history. It has no saved weights. This can't be undone.", button: 'Delete run' };
  const size = bytes(mine.reduce((a, m) => a + m.bytes, 0), units);
  const n = mine.length === 1 ? '1 model' : `${mine.length} models`;
  return {
    description: `Deletes the run, its metric history and its weights on disk: ${n} (${mine.map((m) => m.name).join(', ')}), ${size}. This can't be undone.`,
    button: `Delete run and ${n}`,
  };
}
