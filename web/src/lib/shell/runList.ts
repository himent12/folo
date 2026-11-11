// fw port of runList.svelte.ts (the same shared object; its two fields are fw signals behind getters/setters).
// The run list as the shell polls it (RunWatch, every 2 s): the sidebar's live card reads it, so the shell asks once.

import { signal } from 'fw';
import type { Run } from '$lib/api/types';

const runs = signal<Run[] | undefined>(undefined);
const since = signal<number | null>(null);

/** `since`: when the list last arrived, while its polls fail (the list is then a snapshot, not the current state). */
export const watched = {
  get runs(): Run[] | undefined {
    return runs.get();
  },
  set runs(v: Run[] | undefined) {
    runs.set(v);
  },
  get since(): number | null {
    return since.get();
  },
  set since(v: number | null) {
    since.set(v);
  },
};
