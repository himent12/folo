// The results of a leaderboard's finished entries, loaded once each (a finished entry's result never changes). The
// leaderboard needs two things only results carry: each entry's 95% interval of mAP@0.5:0.95 (the list's `BenchEntry`
// has the mAP but not its interval) and, for an external entry, how many of the plan's conditions it answered.

import { untrack } from 'svelte';
import { SvelteMap } from 'svelte/reactivity';
import { hub } from '$lib/api/index.svelte';
import type { BenchEntry, BenchResult } from '$lib/api/types';

/** Requests at a time. */
const PARALLEL = 4;

export class BenchResults {
  #results = new SvelteMap<string, BenchResult>();
  /** Entries asked for (or failed): never asked twice by the same page. */
  #asked = new Set<string>();

  constructor(benchId: () => string | null, entries: () => readonly BenchEntry[]) {
    $effect(() => {
      const id = benchId();
      const want = entries().filter((e) => e.status === 'done' && e.score);
      if (!id || hub.mode === 'connecting') return;
      void hub.epoch;
      const api = hub.api;
      untrack(() => {
        const todo = want.filter((e) => !this.#asked.has(e.id));
        for (const e of todo) this.#asked.add(e.id);
        let next = 0;
        const worker = async () => {
          for (let e = todo[next++]; e; e = todo[next++]) {
            const entryId = e.id;
            try {
              this.#results.set(entryId, await api.benchResult(id, entryId));
            } catch {
              // The row draws no interval; the entry's own page says what is wrong.
            }
          }
        };
        for (let k = 0; k < Math.min(PARALLEL, todo.length); k++) void worker();
      });
    });
  }

  get(entryId: string): BenchResult | undefined {
    return this.#results.get(entryId);
  }
}
