// Overview reads the datasets once: each one is read for its class balance (lib/api/datasetReads), too many requests to
// poll. A run on a dataset made since (a loop's student trains on the dataset the loop just wrote; a dataset made in
// another tab) would then read "<name> (removed)" (lib/runDataset), and the Datasets panel would never list it.
// `datasetPresence` states the rule for pages that link to one dataset by id (a labeling job, a model, a failed run).

import { effect, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import type { Run } from '$lib/api/types';

/**
 * Reads `datasets` again when a run of `runs` names a dataset the list does not have: once per dataset (one that was
 * really removed is asked about once, not on every poll). Returns whether a dataset is being looked up, so its runs show
 * the name they recorded meanwhile instead of calling it removed.
 */
export function followNewDatasets(
  datasets: { readonly data: readonly { id: string }[] | undefined; refresh(): Promise<void> },
  runs: () => readonly Pick<Run, 'config'>[] | undefined,
): (id: string) => boolean {
  /** `<server> <dataset id>` of every dataset looked up already. */
  const asked = new Set<string>();
  const checking = signal<ReadonlySet<string>>(new Set());
  effect(() => {
    const list = datasets.data;
    const rs = runs();
    if (!list || !rs) return;
    const have = new Set(list.map((d) => d.id));
    const server = hub.server;
    const fresh = [...new Set(rs.map((r) => r.config.dataset_id))].filter((id) => !have.has(id) && !asked.has(`${server} ${id}`));
    if (!fresh.length) return;
    for (const id of fresh) asked.add(`${server} ${id}`);
    untrack(() => {
      checking.set(new Set([...checking.peek(), ...fresh]));
      void datasets.refresh().finally(() => checking.set(new Set([...checking.peek()].filter((id) => !fresh.includes(id)))));
    });
  });
  return (id) => checking.get().has(id);
}

/** What a page knows of a dataset it names by id: see `datasetPresence`. */
export type DatasetPresence = 'registered' | 'removed' | 'unknown';

/** The datasets list as a page reads it: a Resource of `hub.api.datasets`, polled (lib/api/resource). */
export interface DatasetList {
  readonly data: readonly { id: string }[] | undefined;
  readonly error: unknown;
  readonly stale: unknown;
  readonly loading: boolean;
  readonly refreshing: boolean;
  refresh(): Promise<void>;
}

/**
 * Whether the datasets a page names by id (a labeling job's, a model's evaluation, a failed run's) are registered in
 * FOLO, one rule for every page that links to one or offers an action on it (sweep 4, #319):
 * - `registered`: a successful answer of `datasets` requested after the page learned the id has it: link, offer actions;
 * - `removed`: such an answer does not have it: say so, link nowhere;
 * - `unknown`: no such answer yet (the list loads; it is read again for an id it was not asked about), or the last read
 *   failed: neutral text, no link and no claim.
 * Each id is read once more when the page learns it (an answer to a request sent before then may predate a dataset made
 * in another tab); after that the list's own poll keeps it current (one read in flight at a time, stopped with the page).
 */
export function datasetPresence(
  datasets: DatasetList,
  ids: () => readonly (string | null | undefined)[] | undefined,
): (id: string | null | undefined) => DatasetPresence {
  const key = (id: string) => `${hub.server} ${id}`;
  /** The ids (by `key`) learned so far. */
  const asked = new Set<string>();
  /** The ids read again since they were learned: every answer from then on was requested after. */
  const ready = signal<ReadonlyMap<string, string>>(new Map());
  const known = signal<ReadonlyMap<string, DatasetPresence>>(new Map());
  effect(() => {
    const fresh = new Map<string, string>();
    for (const id of ids() ?? []) if (id && !asked.has(key(id))) fresh.set(key(id), id);
    if (!fresh.size) return;
    for (const k of fresh.keys()) asked.add(k);
    // A later read (a poll, another new id) supersedes this one; it was requested later still.
    untrack(() => void datasets.refresh().finally(() => ready.set(new Map([...ready.peek(), ...fresh]))));
  });
  effect(() => {
    // While a read is in flight, what the last answer settled stands (no flicker at every poll).
    if (datasets.loading || datasets.refreshing) return;
    const list = datasets.error === undefined && datasets.stale === undefined ? datasets.data : undefined;
    const have = new Set((list ?? []).map((d) => d.id));
    const next = new Map<string, DatasetPresence>();
    for (const [k, id] of ready.get()) next.set(k, !list ? 'unknown' : have.has(id) ? 'registered' : 'removed');
    const was = known.peek();
    if (next.size !== was.size || [...next].some(([k, p]) => was.get(k) !== p)) known.set(next);
  });
  return (id) => (id ? (known.get().get(key(id)) ?? 'unknown') : 'unknown');
}
