// Overview reads the datasets once: each one is read for its class balance (lib/api/datasetReads), too many requests to
// poll. A run on a dataset made since (a loop's student trains on the dataset the loop just wrote; a dataset made in
// another tab) would then read "<name> (removed)" (lib/runDataset), and the Datasets panel would never list it. The
// Runs list and a labeling job's page (its "made" dataset) follow new datasets the same way.

import { effect, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';

/**
 * Reads `datasets` again when `ids` (the datasets the page names: its runs', a job's) has one the list does not have:
 * once per dataset (one that was really removed is asked about once, not on every poll). Returns whether a dataset is
 * being looked up, so the page shows what it knew meanwhile instead of calling it removed.
 */
export function followNewDatasets(
  datasets: { readonly data: readonly { id: string }[] | undefined; refresh(): Promise<void> },
  ids: () => readonly string[] | undefined,
): (id: string) => boolean {
  /** `<server> <dataset id>` of every dataset looked up already. */
  const asked = new Set<string>();
  const checking = signal<ReadonlySet<string>>(new Set());
  effect(() => {
    const list = datasets.data;
    const want = ids();
    if (!list || !want) return;
    const have = new Set(list.map((d) => d.id));
    const server = hub.server;
    const fresh = [...new Set(want)].filter((id) => !have.has(id) && !asked.has(`${server} ${id}`));
    if (!fresh.length) return;
    for (const id of fresh) asked.add(`${server} ${id}`);
    untrack(() => {
      checking.set(new Set([...checking.peek(), ...fresh]));
      void datasets.refresh().finally(() => checking.set(new Set([...checking.peek()].filter((id) => !fresh.includes(id)))));
    });
  });
  return (id) => checking.get().has(id);
}
