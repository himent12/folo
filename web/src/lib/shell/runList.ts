// // The run list as the shell polls it (RunWatch, every 2 s): the sidebar's live card reads it, so the shell asks once.
// It belongs to the server it came from (`hub.epoch` when it was published): after a reconnect, which may be to another
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
