// What a listed dataset's own read says about it: the list endpoint carries no state, so the list reads each dataset
// (for its stats) and keeps what went wrong.

import { ApiError } from '$lib/api/client';
import type { Dataset } from '$lib/api/types';

/** `missing`: its folder is gone (moved, renamed, deleted); `unreadable`: the folder is there but is not a dataset now. */
export type DatasetProblem = 'missing' | 'unreadable';

/** A row of the Datasets list: the dataset, with stats when it could be read. */
export type ListedDataset = Dataset & { problem?: DatasetProblem };

/** What a failed read of a registered dataset means for its row; nothing for a failure that is not about it (offline). */
export function datasetProblem(e: unknown): DatasetProblem | undefined {
  if (!(e instanceof ApiError)) return undefined;
  if (e.code === 'path_not_found') return 'missing';
  return e.status === 400 ? 'unreadable' : undefined;
}

export const PROBLEM_LABEL: Record<DatasetProblem, string> = { missing: 'Folder missing', unreadable: "Can't read" };
