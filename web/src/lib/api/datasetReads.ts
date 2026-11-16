// The datasets with their stats, as Overview and New run show them: the list endpoint has none, so each dataset is
// read. A read that says its folder is missing (or is not a dataset now) keeps the list's entry, with that `problem`
// (routes/datasets/problem), and the same server is not asked again for that dataset for RECHECK ms: going from one
// screen to another does not fire the same failing request each time (v-data-runs N6). A folder put back shows on the
// first read after that.

import { effect, untrack } from 'fw';
import { datasetProblem, type DatasetProblem, type ListedDataset } from '../../routes/datasets/problem';
import { hub } from './index';
import { Resource } from './resource';

/** How long a dataset's problem is taken as known before its folder is read again. */
export const RECHECK = 15_000;

/** `<server> <id>` → the problem its last read found, and when. */
const known = new Map<string, { problem: DatasetProblem; at: number }>();
/** `<server> <id>` → the name it was last listed by: one removed since is still named. */
const names = new Map<string, string>();

/** The name a dataset was last listed by on this server (it may be gone now). */
export const listedName = (id: string): string | undefined => names.get(`${hub.server} ${id}`);

/** Every registered dataset, read for its stats; one that cannot be read is its list entry with the `problem`. */
export async function readDatasets(s: AbortSignal): Promise<ListedDataset[]> {
  const list = await hub.api.datasets(s);
  const server = hub.server;
  return Promise.all(
    list.map(async (d): Promise<ListedDataset> => {
      const key = `${server} ${d.id}`;
      names.set(key, d.name);
      const k = known.get(key);
      if (k && Date.now() - k.at < RECHECK) return { ...d, problem: k.problem };
      try {
        const full = await hub.api.dataset(d.id, s);
        known.delete(key);
        return full;
      } catch (e) {
        const problem = datasetProblem(e);
        if (!problem) return d;
        known.set(key, { problem, at: Date.now() });
        return { ...d, problem };
      }
    }),
  );
}

/**
 * The datasets with their stats, kept current while a form shows them (PR #287 review, sibling of stress s3-live-3): the
 * list alone is polled (the Run page's request, shared) and every dataset read again only when the list changed, so one
 * removed or added in another window shows without a stats read of each dataset every few seconds. Each list answer is
 * looked at once (re-review 1): a stats read that fails is tried again at the next answer, not as soon as it failed, and
 * stats read newer than the list (Start's `dataset_not_found`) wait for it instead of being read again and again.
 */
export function followedDatasets(every = 5000): Resource<ListedDataset[]> {
  const datasets = new Resource(readDatasets);
  const listed = new Resource((s) => hub.api.datasets(s), { poll: every });
  let handled: number | null = null;
  effect(() => {
    const at = listed.updated;
    const now = listed.data;
    if (at === null || !now || at === handled) return;
    const shown = datasets.data;
    if (!shown || datasets.loading || datasets.refreshing) return;
    handled = at;
    const ids = (l: readonly { id: string }[]) => l.map((d) => d.id).sort().join('\n');
    if (ids(now) !== ids(shown)) void untrack(() => datasets.refresh());
  });
  return datasets;
}
