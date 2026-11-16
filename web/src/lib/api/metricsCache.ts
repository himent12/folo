// Small cache of per-run mAP histories for sparklines in lists, per API instance (live server vs mock). Finished
// runs are fetched once; an active run's entry is replaced as its status and progress advance. One entry per run,
// least recently used evicted first, with a bound well above any list: a list longer than the bound would evict,
// in order, every entry it is about to ask for and so refetch every history on each visit.

import type { Api } from './client';
import type { Run } from './types';

const FINAL = new Set(['completed', 'stopped', 'failed']);
/** Most runs kept (a history is one number per epoch). */
export const MAX_RUNS = 5000;

interface Entry {
  key: string;
  p: Promise<number[]>;
}
const caches = new WeakMap<Api, Map<string, Entry>>();

export function mapHistory(api: Api, run: Run): Promise<number[]> {
  let cache = caches.get(api);
  if (!cache) caches.set(api, (cache = new Map()));
  const key = FINAL.has(run.status) ? `final:${run.epoch}` : `${run.status}:${run.epoch}`;
  let entry = cache.get(run.id);
  if (!entry || entry.key !== key) {
    const c = cache;
    const e: Entry = { key, p: api.runMetrics(run.id).then((m) => m.epochs.map((ep) => ep.val.map50_95)) };
    e.p.catch(() => {
      if (c.get(run.id) === e) c.delete(run.id);
    });
    entry = e;
  }
  // Re-inserted on every use: the Map's order is the eviction order (least recently used first).
  cache.delete(run.id);
  cache.set(run.id, entry);
  while (cache.size > MAX_RUNS) cache.delete(cache.keys().next().value as string);
  return entry.p;
}
