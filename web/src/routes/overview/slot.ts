// The run in this machine's training slot, as Overview's Training card shows it: the run training (or stopping), else
// the run that is starting. A starting run is still `queued` (docs/API.md, the training queue): the worker has taken
// it and builds its trainer (`queue_position` null), or it is first in the queue with the slot free (an older server
// reports no position). Cloud runs wait in no queue here and never hold this slot.

import type { Run } from '$lib/api/types';
import type { OwnStatus } from '$lib/ui/StatusBadge';

const training = (r: Run) => r.status === 'running' || r.status === 'stopping';
const waiting = (r: Run) => r.status === 'queued' && !r.cloud;

/** The run the Training card is about, or undefined when the slot is free and nothing is about to take it. */
export function slotRun(runs: readonly Run[] | undefined): Run | undefined {
  if (!runs) return undefined;
  return runs.find(training) ?? runs.find((r) => waiting(r) && r.queue_position == null) ?? runs.find((r) => waiting(r) && r.queue_position === 1);
}

/** A run in the slot that has not started training yet: the card says "Starting", not "Queued". */
export const starting = (r: Pick<Run, 'status'>) => r.status === 'queued';

/** Its badge: the mark of a waiting run, the word of one about to train (never a cloud run: those hold no slot). */
export const STARTING: OwnStatus = { status: 'queued', label: 'Starting' };
