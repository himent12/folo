// Moving a waiting run in the training queue from a screen (docs/API.md, POST /api/runs/{id}/queue), with one wording
// for the Run page and the Runs list. The new order shows at once; a refusal puts the old one back. FOLO trains one
// run at a time and never interrupts it, so "start" moves a run to the front: it starts when the active run ends
// (at once only when nothing trains).

import ArrowDown from '@lucide/svelte/icons/arrow-down';
import ArrowUp from '@lucide/svelte/icons/arrow-up';
import ChevronsUp from '@lucide/svelte/icons/chevrons-up';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index.svelte';
import type { QueueRequest, Run } from '$lib/api/types';
import type { MenuItem } from '$lib/ui/Menu.svelte';
import { toast } from '$lib/ui/toast.svelte';

/** "Start next" while another run trains (the move puts it at the front), "Start now" when the slot is free. */
export const startLabel = (training: Pick<Run, 'name'> | null | undefined) => (training ? 'Start next' : 'Start now');

/** What Start next does, in one line (its tooltip and accessible description). */
export const startHint = (training: Pick<Run, 'name'> | null | undefined) =>
  training ? `Moves it to the front. It starts when ${training.name} finishes: FOLO never interrupts a run.` : 'Starts it now: nothing else is training.';

/**
 * Moves `run`: `apply` shows the new order and returns its undo, which runs when the server refuses. A run the worker
 * took meanwhile (`409 run_not_queued`) says so plainly and `refresh`es; anything else is an error toast. Resolves to
 * the server's run, or null when the move failed.
 */
export async function moveQueued(
  run: Pick<Run, 'id' | 'name'>,
  req: QueueRequest,
  apply: () => () => void,
  refresh: () => void,
  training?: Pick<Run, 'name'> | null,
): Promise<Run | null> {
  const undo = apply();
  try {
    const r = await hub.api.moveRun(run.id, req);
    if ('action' in req && req.action === 'start') {
      if (r.queue_position == null) toast.success(`Starting ${r.name}`);
      else toast.success(`${r.name} starts next`, training ? `When ${training.name} finishes.` : undefined);
    }
    return r;
  } catch (e) {
    undo();
    if (e instanceof ApiError && e.code === 'run_not_queued') {
      toast.info('It already started', `${run.name} left the queue before the move.`);
      refresh();
    } else toast.error(e, `Could not move ${run.name}`);
    return null;
  }
}

/**
 * A waiting run's menu items: Start next (unless it already is next), Move up, Move down (none at the end they move
 * towards). `position` is its 1-based place, `waiting` how many runs wait.
 */
export function queueItems(
  position: number,
  waiting: number,
  training: Pick<Run, 'name'> | null | undefined,
  move: (req: QueueRequest) => void,
  opts: { start?: boolean } = {},
): MenuItem[] {
  const out: MenuItem[] = [];
  if ((opts.start ?? true) && (position > 1 || !training)) out.push({ label: startLabel(training), icon: ChevronsUp, onclick: () => move({ action: 'start' }) });
  if (position > 1) out.push({ label: 'Move up', icon: ArrowUp, onclick: () => move({ action: 'up' }) });
  if (position < waiting) out.push({ label: 'Move down', icon: ArrowDown, onclick: () => move({ action: 'down' }) });
  return out;
}
