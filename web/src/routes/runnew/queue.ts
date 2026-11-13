// Whether a new run starts at once or waits (docs/API.md, POST /api/runs/{id}/queue): this machine trains one run at a
// time, so while another is training (or starting) a new one waits in the queue, behind the runs already waiting. A
// cloud run trains elsewhere and never waits for this machine.

import { eta } from '$lib/format';
import { settledEta } from '$lib/api/progress';
import { ordinal, queuePlace } from '$lib/api/queue';
import type { Run } from '$lib/api/types';

/** What a new run of this machine would wait for: the run holding the slot, and how many already wait. */
export interface Ahead {
  /** The run training (or being started) now; null when the slot is free but runs wait anyway. */
  training: Run | null;
  waiting: number;
  /** The new run's place in the queue, 1-based (1 starts next). */
  position: number;
}

/** Null when a new run would start at once: nothing holds this machine's slot or waits for it, or it trains in the cloud. */
export function aheadOf(runs: readonly Run[] | undefined, onCloud: boolean): Ahead | null {
  if (onCloud || !runs) return null;
  const local = runs.filter((r) => !r.cloud && !r.config.compute);
  // A run the worker took (still `queued` while its trainer is built, so out of the queue) holds the slot too.
  const training = local.find((r) => r.status === 'running' || r.status === 'stopping' || (r.status === 'queued' && r.queue_position == null)) ?? null;
  const waiting = local.filter((r) => r.status === 'queued' && r.queue_position != null).length;
  return training || waiting ? { training, waiting, position: waiting + 1 } : null;
}

/** The Summary's line: what the run waits for, its place, and when the run training now should end. */
export function aheadLine(a: Ahead): string {
  const now = a.training ? `${a.training.name} is ${a.training.status === 'queued' ? 'starting' : 'training'}` : 'Another run is starting';
  const left = a.training ? settledEta(a.training) : null;
  const ends = left !== null ? ` (it ends in about ${eta(left)})` : '';
  if (a.waiting === 0) return `${now}: this run waits in the queue and starts when it ends${ends}.`;
  const others = `${a.waiting} ${a.waiting === 1 ? 'run waits' : 'runs wait'}`;
  return `${now} and ${others}: this run will be ${ordinal(a.position)} in the queue${ends}.`;
}

/**
 * The toast after Start, by what the server answered: a run with a place in the queue waits only behind a run that
 * holds the slot (or behind other waiting runs); a first place on a free machine is the worker about to take it.
 */
export function startedToast(run: Run, before: Ahead | null): { title: string; place: string | null } {
  const pos = run.queue_position;
  const waits = pos != null && (pos > 1 || !!before?.training);
  return waits ? { title: `Queued ${run.name}`, place: queuePlace(pos) } : { title: `Started ${run.name}`, place: null };
}
