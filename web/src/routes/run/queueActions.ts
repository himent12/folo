// Moving a waiting run in the training queue from a screen (docs/API.md, POST /api/runs/{id}/queue), with one wording
// for the Run page and the Runs list. The new order shows at once; a refusal puts the old one back. FOLO trains one
// run at a time and never interrupts it, so "start" moves a run to the front: it starts when the active run ends
// (at once only when nothing trains).

import ArrowDown from '$icons/arrow-down';
import ArrowUp from '$icons/arrow-up';
import ChevronsUp from '$icons/chevrons-up';
import X from '$icons/x';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import type { QueueRequest, Run } from '$lib/api/types';
import { toast } from '$lib/ui/toast';
import type { IconType } from '$lib/ui/types';

/**
 * ui/Menu's `MenuItem`, by shape: Menu is a .tsx module, which the Svelte app's tsconfig (it checks this .ts file too)
 * cannot read until the switch.
 */
interface MenuItem {
  label: string;
  icon?: IconType;
  href?: string;
  onclick?: () => void;
  danger?: boolean;
}

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

/** Taking a waiting run out of the queue: one name, and one confirm, on the Runs list and the run's page. */
export const REMOVE_LABEL = 'Remove from queue';
export const removeTitle = (name: string) => `Remove ${name} from the queue?`;
/**
 * What removing does: a run that never started is deleted (DELETE /api/runs/{id}), so it leaves the list too; a resumed
 * run that waits to continue is stopped and keeps what it trained.
 */
export const removeDescription = (run: Pick<Run, 'started'>) =>
  run.started
    ? 'It stops waiting and keeps what it trained so far. You can resume it later.'
    : 'It has not started yet, so nothing is lost: it leaves the queue and the runs list. You can start it again from New run.';

/**
 * The toast key of a run's place in the queue: "Queued q-two" (New run) and "Removed q-two from the queue" report on
 * the same thing, so the second takes the first one's place instead of showing beside it.
 */
export const queueToastKey = (id: string) => `run-queue:${id}`;

/** The toast once a waiting run was taken out of the queue. */
export function removedToast(run: Pick<Run, 'id' | 'name'>): void {
  toast.success(`Removed ${run.name} from the queue`, undefined, undefined, { key: queueToastKey(run.id) });
}

/** Removes a waiting run from the queue: deleted when it never started (null), stopped otherwise (the stopped run). */
export async function removeQueued(run: Pick<Run, 'id' | 'started'>): Promise<Run | null> {
  if (run.started) return hub.api.stopRun(run.id);
  await hub.api.deleteRun(run.id);
  return null;
}

/**
 * A waiting run's queue actions, one set in one order wherever they appear (the Runs row's ⋯ and the run page's):
 * Start next (unless it already is next), Move up, Move down (none at the end they move towards), Remove from queue.
 * The set is whole in every menu, even where the page also shows Start next as a button (the run page's callout): an
 * open menu covers that button, and an object has one action set (DESIGN_DIRECTION §2.13). `position` is its 1-based
 * place, `waiting` how many runs wait, `remove` opens the one confirm.
 */
export function queueItems(
  position: number,
  waiting: number,
  training: Pick<Run, 'name'> | null | undefined,
  move: (req: QueueRequest) => void,
  remove: () => void,
): MenuItem[] {
  const out: MenuItem[] = [];
  if (position > 1 || !training) out.push({ label: startLabel(training), icon: ChevronsUp, onclick: () => move({ action: 'start' }) });
  if (position > 1) out.push({ label: 'Move up', icon: ArrowUp, onclick: () => move({ action: 'up' }) });
  if (position < waiting) out.push({ label: 'Move down', icon: ArrowDown, onclick: () => move({ action: 'down' }) });
  out.push({ label: REMOVE_LABEL, icon: X, danger: true, onclick: remove });
  return out;
}

/** A run's expected length: the server's estimate of its settings (seconds, low and high), null where it gives none. */
export type Length = { lo: number; hi: number } | null;

/**
 * How long a waiting run waits: what the training run has left (`left`; undefined when nothing trains, null when its
 * time left is unknown) plus the length of each queued run in front of it. `known: false` when any part is unknown:
 * the sum is then a floor ("more than 7m"), never an estimate of the whole. Null when no part is known.
 */
export function queueWait(left: number | null | undefined, lengths: readonly Length[]): { lo: number; hi: number; known: boolean } | null {
  const parts: Length[] = [...(left === undefined ? [] : [left === null ? null : { lo: left, hi: left }]), ...lengths];
  const got = parts.filter((p): p is { lo: number; hi: number } => p !== null);
  if (!got.length) return null;
  return { lo: got.reduce((a, p) => a + p.lo, 0), hi: got.reduce((a, p) => a + p.hi, 0), known: got.length === parts.length };
}
