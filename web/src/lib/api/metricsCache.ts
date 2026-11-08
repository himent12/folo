// Small cache of per-run mAP histories for sparklines in lists. Keyed by run id and epoch count, so a live run's
// entry refreshes as it advances while finished runs are fetched once.

import type { Api } from './client';
import type { Run } from './types';

const cache = new Map<string, Promise<number[]>>();

export function mapHistory(api: Api, run: Run): Promise<number[]> {
  const key = `${run.id}:${run.status === 'running' ? run.epoch : 'final'}`;
  let p = cache.get(key);
  if (!p) {
    p = api.runMetrics(run.id).then((m) => m.epochs.map((e) => e.val.map50_95));
    p.catch(() => cache.delete(key));
    cache.set(key, p);
    if (cache.size > 200) cache.delete(cache.keys().next().value as string);
  }
  return p;
}
