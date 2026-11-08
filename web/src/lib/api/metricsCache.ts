// Small cache of per-run mAP histories for sparklines in lists, per API instance (live server vs mock). Finished
// runs are fetched once; active runs are keyed by status and progress so their entry refreshes as they advance.

import type { Api } from './client';
import type { Run } from './types';

const FINAL = new Set(['completed', 'stopped', 'failed']);
const caches = new WeakMap<Api, Map<string, Promise<number[]>>>();

export function mapHistory(api: Api, run: Run): Promise<number[]> {
  let cache = caches.get(api);
  if (!cache) caches.set(api, (cache = new Map()));
  const key = FINAL.has(run.status) ? `${run.id}:final:${run.epoch}` : `${run.id}:${run.status}:${run.epoch}`;
  let p = cache.get(key);
  if (!p) {
    const c = cache;
    p = api.runMetrics(run.id).then((m) => m.epochs.map((e) => e.val.map50_95));
    p.catch(() => c.delete(key));
    c.set(key, p);
    if (c.size > 200) c.delete(c.keys().next().value as string);
  }
  return p;
}
