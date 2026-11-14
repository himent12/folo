// // The run list as the shell polls it (RunWatch, every 2 s): the sidebar's live row and the Runs badge read it, so
// the shell asks once. It belongs to the server it came from (`hub.epoch` when it was published): after a reconnect, which may be to another
// server or the mock, it reads as no list until the server in use has sent its own.

import { signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import type { Run } from '$lib/api/types';

const runs = signal<{ v: Run[] | undefined; epoch: number }>({ v: undefined, epoch: -1 });
const since = signal<{ v: number | null; epoch: number }>({ v: null, epoch: -1 });

/** `since`: when the list last arrived, while its polls fail (the list is then a snapshot, not the current state). */
export const watched = {
  get runs(): Run[] | undefined {
    const r = runs.get();
    return r.epoch === hub.epoch ? r.v : undefined;
  },
  set runs(v: Run[] | undefined) {
    runs.set({ v, epoch: untrack(() => hub.epoch) });
  },
  get since(): number | null {
    const s = since.get();
    return s.epoch === hub.epoch ? s.v : null;
  },
  set since(v: number | null) {
    since.set({ v, epoch: untrack(() => hub.epoch) });
  },
};

/** A run that holds this machine's trainer: training, or stopping after it. */
export const training = (r: { status: string }) => r.status === 'running' || r.status === 'stopping';

/**
 * The Runs badge (sidebar, icon rail, tab bar): how many runs train and how many wait, from the list the shell polls.
 * `live`: one of them is training now (the icon rail's and the tab bar's lime dot).
 */
export function activeRuns(): { training: number; queued: number; live: boolean } {
  const list = watched.runs ?? [];
  const t = list.filter(training).length;
  return { training: t, queued: list.filter((r) => r.status === 'queued').length, live: t > 0 };
}

/** The Runs row's name with its badge said in words ("Runs, 1 training, 2 queued"), for its tooltip and screen readers. */
export function runsBadgeText(): string {
  const a = activeRuns();
  const parts = [a.training ? `${a.training} training` : '', a.queued ? `${a.queued} queued` : ''].filter(Boolean);
  return parts.join(', ');
}
