// The datasets with their stats, as Overview and New run show them: the list endpoint has none, so each dataset is
// read. A read that says its folder is missing (or is not a dataset now) keeps the list's entry, with that `problem`
// (routes/datasets/problem), and the same server is not asked again for that dataset for RECHECK ms: going from one
// screen to another does not fire the same failing request each time (v-data-runs N6). A folder put back shows on the
// first read after that.

import { datasetProblem, type DatasetProblem, type ListedDataset } from '../../routes/datasets/problem';
import { hub } from './index';

/** How long a dataset's problem is taken as known before its folder is read again. */
export const RECHECK = 15_000;

/** `<server> <id>` → the problem its last read found, and when. */
const known = new Map<string, { problem: DatasetProblem; at: number }>();

/** Every registered dataset, read for its stats; one that cannot be read is its list entry with the `problem`. */
export async function readDatasets(s: AbortSignal): Promise<ListedDataset[]> {
  const list = await hub.api.datasets(s);
  const server = hub.server;
  return Promise.all(
    list.map(async (d): Promise<ListedDataset> => {
      const key = `${server} ${d.id}`;
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
