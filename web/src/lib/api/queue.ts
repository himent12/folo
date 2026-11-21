// The training queue (docs/API.md, POST /api/runs/{id}/queue): one pure rule for a move, shared by the mock (its queue
// of run ids) and the screens (their optimistic order before the server answers), and the words for a place in it.

import type { QueueRequest, Run } from './types';

/** Whether the server lists an optional API feature (`queue_control`, `weights_onnx`, ...) in `/api/health` or `/api/system`. */
export function hasFeature(system: { features?: readonly string[] } | undefined | null, name: string): boolean {
  return !!system?.features?.includes(name);
}

/**
 * The queue after `req` moves `id` (front first). `top` / `bottom` to an end, `up` / `down` one place (a no-op at the
 * end it moves towards), `position: n` to place n (past the end means the back), `start` to the front (it never
 * preempts the active run: the worker takes it when the slot is free). The other runs keep their order. Null when
 * `id` is not waiting.
 */
export function applyQueueMove(order: readonly string[], id: string, req: QueueRequest): string[] | null {
  const from = order.indexOf(id);
  if (from < 0) return null;
  const rest = order.filter((x) => x !== id);
  let to: number;
  if ('position' in req) to = Math.max(0, Math.min(rest.length, Math.floor(req.position) - 1));
  else if (req.action === 'top' || req.action === 'start') to = 0;
  else if (req.action === 'bottom') to = rest.length;
  else if (req.action === 'up') to = Math.max(0, from - 1);
  else to = Math.min(rest.length, from + 1);
  rest.splice(to, 0, id);
  return rest;
}

/**
 * `runs` with the waiting runs' `queue_position` as they will be after `req` moves `id`: the optimistic list a screen
 * shows until the server answers. Unchanged when `id` is not waiting.
 */
export function withQueueMove(runs: readonly Run[], id: string, req: QueueRequest): Run[] {
  const waiting = runs.filter((r) => r.queue_position != null).sort((a, b) => (a.queue_position ?? 0) - (b.queue_position ?? 0));
  const order = applyQueueMove(
    waiting.map((r) => r.id),
    id,
    req,
  );
  if (!order) return [...runs];
  return runs.map((r) => (r.queue_position != null ? { ...r, queue_position: order.indexOf(r.id) + 1 } : r));
}

/** 1 → "1st", 2 → "2nd", 11 → "11th", 23 → "23rd". */
export function ordinal(n: number): string {
  const tens = n % 100;
  const suffix = tens >= 11 && tens <= 13 ? 'th' : (['th', 'st', 'nd', 'rd'][n % 10] ?? 'th');
  return `${n}${suffix}`;
}

/** "2nd in queue". */
export const queuePlace = (position: number) => `${ordinal(position)} in queue`;
