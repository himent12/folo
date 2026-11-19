// Overview reads the datasets once: each one is read for its class balance (lib/api/datasetReads), too many requests to
// poll. A run on a dataset made since (a loop's student trains on the dataset the loop just wrote; a dataset made in
// another tab) would then read "<name> (removed)" (lib/runDataset), and the Datasets panel would never list it.
// `datasetPresence` states the rule for pages that link to one dataset by id (a labeling job, a model, a failed run).

import { computed, effect, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import type { Dataset, Run } from '$lib/api/types';

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

/** The datasets list as a page reads it: `datasetsFor`, or a page's own polled Resource of `hub.api.datasets`. */
export interface DatasetList {
  readonly data: readonly { id: string }[] | null | undefined;
  readonly error: unknown;
  readonly stale: unknown;
  readonly loading: boolean;
  readonly refreshing: boolean;
  refresh(): Promise<void>;
}

/**
 * The datasets list for `datasetPresence` on a page that reads it for nothing else: read, and polled every 5 s, only
 * while `ids` names a dataset, or is undefined (the page does not know yet: read alongside what it waits for, not after
 * it); a page that names none sends no request (null).
 */
export function datasetsFor(ids: () => readonly (string | null | undefined)[] | undefined): Resource<Dataset[] | null> {
  const named = computed(() => ids()?.some(Boolean) ?? true);
  return new Resource((s) => (named.get() ? hub.api.datasets(s) : Promise.resolve(null)), { poll: 5000 });
}

/**
 * Whether the datasets a page names by id (a labeling job's, a model's run and evaluation, a failed run's) are registered
 * in FOLO (sweep 4, #319). The invariant: for a dataset that is registered, a page never shows less than it did before
 * this rule, and never later; the only difference is that an id a successful read of the list says is absent is shown
 * as removed. So:
 * - `registered`: a successful answer has the id, whenever it was requested (the first one too: nothing more is read);
 * - `removed`: a successful answer requested after the page learned the id does not have it (one requested before may
 *   predate a dataset made since, in another tab: the list is read once more for an id it lacks);
 * - `unknown`: neither yet; the page shows what it showed before the rule.
 * A failed read changes nothing: only a successful answer moves an id, either way. The list's poll keeps it current.
 */
export function datasetPresence(
  datasets: DatasetList,
  ids: () => readonly (string | null | undefined)[] | undefined,
): (id: string | null | undefined) => DatasetPresence {
  const key = (id: string) => `${hub.server} ${id}`;
  /** The ids (by `key`) the list was read again for: the page learned them while the list lacked them. */
  const asked = new Set<string>();
  /** Those whose read is back: every answer from then on was requested after the page learned them. */
  const ready = signal<ReadonlySet<string>>(new Set());
  const known = signal<ReadonlyMap<string, DatasetPresence>>(new Map());
  effect(() => {
    // An answer is on its way: what the last ones settled stands until it lands.
    if (datasets.loading || datasets.refreshing) return;
    const failed = datasets.error !== undefined || datasets.stale !== undefined;
    const list = failed ? null : datasets.data;
    if (!list && !failed) return; // nothing read (`datasetsFor` while the page names no dataset)
    const have = new Set((list ?? []).map((d) => d.id));
    const was = known.peek();
    const next = new Map(was);
    const fresh: string[] = [];
    for (const id of ids() ?? []) {
      if (!id) continue;
      const k = key(id);
      if (list && have.has(id)) next.set(k, 'registered');
      else if (ready.get().has(k)) {
        if (list) next.set(k, 'removed');
      } else if (!asked.has(k)) fresh.push(k);
    }
    if ([...next].some(([k, p]) => was.get(k) !== p)) known.set(next);
    if (!fresh.length) return;
    for (const k of fresh) asked.add(k);
    // A later read (a poll) supersedes this one; it was requested later still.
    untrack(() => void datasets.refresh().finally(() => ready.set(new Set([...ready.peek(), ...fresh]))));
  });
  return (id) => (id ? (known.get().get(key(id)) ?? 'unknown') : 'unknown');
}
