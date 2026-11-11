// What a change of the run list means for the rest of Overview: a run that ended (completed, stopped, failed) or was
// deleted changes the models (a finished run saves its best weights as a model; deleting a run deletes its models), so
// Best models refetches then, the moment the user is most likely looking at it, and never on an unchanged poll.

import type { Run, RunStatus } from '$lib/api/types';

const ENDED = new Set<RunStatus>(['stopped', 'completed', 'failed']);

/** Whether a run in `prev` has ended or gone in `next` (false on the first list, when `prev` is undefined). */
export function modelsMayHaveChanged(prev: readonly Run[] | undefined, next: readonly Run[]): boolean {
  if (!prev) return false;
  const now = new Map(next.map((r) => [r.id, r.status]));
  return prev.some((r) => {
    const status = now.get(r.id);
    return status === undefined || (!ENDED.has(r.status) && ENDED.has(status));
  });
}
