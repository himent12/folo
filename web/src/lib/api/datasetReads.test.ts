// @vitest-environment happy-dom
import { createRoot, flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { followedDatasets } from './datasetReads';
import { hub } from './index';
import type { Resource } from './resource';
import type { ListedDataset } from '../../routes/datasets/problem';

const root = (fn: () => void) => createRoot((dispose) => (fn(), dispose));

/** A server whose answers take 10 ms: a loop of requests driven by their answers shows up as a request count. */
function server() {
  const s = {
    ids: ['a', 'b'],
    fail: false,
    lists: 0,
    stats: 0,
    datasets: () => {
      s.lists++;
      const ids = [...s.ids];
      const fail = s.fail;
      return new Promise((ok, no) => setTimeout(() => (fail ? no(new Error('offline')) : ok(ids.map((id) => ({ id, name: id })))), 10));
    },
    dataset: (id: string) => {
      s.stats++;
      return new Promise((ok) => setTimeout(() => ok({ id, name: id, splits: { train: 1, val: 0, test: 0 } }), 10));
    },
  };
  return s;
}

const ids = (r: Resource<ListedDataset[]>) => r.data?.map((d) => d.id);

describe('followedDatasets', () => {
  let s: ReturnType<typeof server>;
  let stop: (() => void) | undefined;
  beforeEach(() => {
    vi.useFakeTimers();
    hub.mode = 'mock';
    s = server();
    hub.api = s as never;
  });
  afterEach(() => {
    stop?.();
    stop = undefined;
    vi.useRealTimers();
  });

  async function follow(): Promise<Resource<ListedDataset[]>> {
    let r!: Resource<ListedDataset[]>;
    stop = root(() => (r = followedDatasets(1000)));
    flushSync();
    await vi.advanceTimersByTimeAsync(50);
    expect(ids(r)).toEqual(['a', 'b']);
    return r;
  }

  // PR #287 re-review 1: a stats read that failed was started again as soon as it failed, with no delay.
  it('tries a failed re-read again at the next list answer, not in a loop', async () => {
    const r = await follow();
    s.ids = ['a'];
    await vi.advanceTimersByTimeAsync(955); // the poll asked (its answer says b is gone) ...
    s.fail = true; // ... and every request from now on fails, the re-read it starts too
    const before = s.lists;
    await vi.advanceTimersByTimeAsync(3000);
    expect(s.lists - before).toBeLessThanOrEqual(4); // the polls alone, no retry loop
    expect(ids(r)).toEqual(['a', 'b']); // the failed read keeps what is shown
    s.fail = false;
    await vi.advanceTimersByTimeAsync(1100);
    expect(ids(r)).toEqual(['a']);
  });

  it('does not re-read when the stats are newer than the list (a refresh before the poll)', async () => {
    const r = await follow();
    s.ids = ['a'];
    const done = r.refresh();
    await vi.advanceTimersByTimeAsync(30);
    await done;
    flushSync();
    expect(ids(r)).toEqual(['a']);
    const stats = s.stats;
    await vi.advanceTimersByTimeAsync(900); // before the next poll: nothing more
    expect(s.stats).toBe(stats);
    await vi.advanceTimersByTimeAsync(1100); // the poll agrees: still nothing
    expect(s.stats).toBe(stats);
  });

  it('re-reads the stats once when a dataset is added', async () => {
    const r = await follow();
    const stats = s.stats;
    s.ids = ['a', 'b', 'c'];
    await vi.advanceTimersByTimeAsync(1100);
    expect(ids(r)).toEqual(['a', 'b', 'c']);
    expect(s.stats - stats).toBe(3);
    await vi.advanceTimersByTimeAsync(3000);
    expect(s.stats - stats).toBe(3);
  });
});