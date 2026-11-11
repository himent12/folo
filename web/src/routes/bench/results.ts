// fw port of results.svelte.ts.
// The results of finished entries, loaded once each (a finished entry's result never changes): Compare's charts need
// each entry's whole result. (The leaderboard reads its interval and conditions from `BenchEntry` itself.)

import { effect, ReactiveMap, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import type { BenchEntry, BenchResult } from '$lib/api/types';

/** Requests at a time. */
const PARALLEL = 4;

export class BenchResults {
  #results = new ReactiveMap<string, BenchResult>();
  /** Per entry, the error of its last load: a page shows it with a retry instead of a skeleton forever. */
  #errors = new ReactiveMap<string, unknown>();
  /** Entries asked for: never asked twice by the same page, unless `retry` asks again. */
  #asked = new Set<string>();
  #again = signal(0);

  /** Call during a component's setup: the loads follow its life. */
  constructor(benchId: () => string | null, entries: () => readonly BenchEntry[]) {
    effect(() => {
      const id = benchId();
      const want = entries().filter((e) => e.status === 'done' && e.score);
      if (!id || hub.mode === 'connecting') return;
      void hub.epoch;
      void this.#again.get();
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
              this.#errors.delete(entryId);
            } catch (err) {
              this.#errors.set(entryId, err);
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

  /** The error of an entry's result that could not be loaded, else undefined. */
  error(entryId: string): unknown {
    return this.#errors.get(entryId);
  }

  /** Asks again for every result that failed. */
  retry() {
    for (const id of this.#errors.keys()) this.#asked.delete(id);
    this.#errors.clear();
    this.#again.set((n) => n + 1);
  }
}
