// What a listed dataset's own read says about it: the list endpoint carries no state, so the list reads each dataset
// (for its stats) and keeps what went wrong.

import { ApiError } from '$lib/api/client';
import type { Dataset } from '$lib/api/types';

/** `missing`: its folder is gone (moved, renamed, deleted); `unreadable`: the folder is there but is not a dataset now. */
export type DatasetProblem = 'missing' | 'unreadable';

/** A row of the Datasets list: the dataset, with stats when it could be read (`pending` while its detail is in flight). */
export type ListedDataset = Dataset & { problem?: DatasetProblem; pending?: boolean };

/** What a failed read of a registered dataset means for its row; nothing for a failure that is not about it (offline). */
export function datasetProblem(e: unknown): DatasetProblem | undefined {
  if (!(e instanceof ApiError)) return undefined;
  if (e.code === 'path_not_found') return 'missing';
  return e.status === 400 ? 'unreadable' : undefined;
}

export const PROBLEM_LABEL: Record<DatasetProblem, string> = { missing: 'Folder missing', unreadable: "Can't read" };

/** A registration refused because the folder holds images but no labels (the server says so in these words). */
export function isUnlabeled(e: unknown): boolean {
  return e instanceof ApiError && e.code === 'unknown_format' && /but no labels$/.test(e.message);
}

/**
 * The datasets whose folder was found missing, with that read's error, per API instance (a switch of server starts
 * afresh, as lib/api/metricsCache): reading one fails until its folder is back, so the list and its page say so from
 * here instead of asking again on every visit (v-data-runs N6). They ask again on Retry, when the tab comes back and
 * after a dataset action in this tab.
 */
const missing = new WeakMap<object, Map<string, unknown>>();

/** The error of the read that found dataset `id`'s folder missing on `api`, while nothing has read it since. */
export function missingError(api: object, id: string): unknown {
  return missing.get(api)?.get(id);
}

/** Records how a read of dataset `id` on `api` ended: `error` undefined for a success. */
export function noteRead(api: object, id: string, error?: unknown): void {
  let m = missing.get(api);
  if (error === undefined) m?.delete(id);
  else if (datasetProblem(error) === 'missing') {
    if (!m) missing.set(api, (m = new Map()));
    m.set(id, error);
  }
}

/** Reads dataset `id` on `api`, or fails at once with the remembered error of a missing folder unless `again`. */
export function readDataset<T>(api: object, id: string, read: () => Promise<T>, again: boolean): Promise<T> {
  const known = again ? undefined : missingError(api, id);
  if (known !== undefined) return Promise.reject(known);
  return read().then(
    (d) => (noteRead(api, id), d),
    (e: unknown) => {
      noteRead(api, id, e);
      throw e;
    },
  );
}
