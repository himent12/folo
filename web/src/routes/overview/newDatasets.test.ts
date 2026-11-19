// @vitest-environment happy-dom
// datasetPresence (sweep 4, #319 and its review): a page links to a dataset only once an answer requested after it
// learned the id has it, says "removed" only once such an answer does not, and claims nothing in between.
import { createRoot, flush as flushSync, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import { datasetPresence, type DatasetPresence } from './newDatasets';

type List = { id: string }[];
interface Call {
  resolve: (v: List) => void;
  reject: (e: unknown) => void;
}

/** A polled datasets Resource whose every read waits for the test, and `datasetPresence` over `ids`. */
function setup(initial: string[]) {
  const calls: Call[] = [];
  const ids = signal<string[]>(initial);
  let presence!: (id: string) => DatasetPresence;
  const stop = createRoot((dispose) => {
    const datasets = new Resource<List>(
      () =>
        new Promise<List>((resolve, reject) => {
          calls.push({ resolve, reject });
        }),
      { poll: 5000 },
    );
    presence = datasetPresence(datasets, () => ids.get());
    return dispose;
  });
  flushSync();
  const at = (id: string) => (flushSync(), presence(id));
  return { calls, ids, at, stop, last: () => calls[calls.length - 1]! };
}

const settle = () => vi.advanceTimersByTimeAsync(0);

describe('datasetPresence', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    hub.mode = 'mock';
  });
  afterEach(() => vi.useRealTimers());

  it('is unknown while the first read is delayed, then registered or removed by its answer', async () => {
    const t = setup(['ds_a', 'ds_b']);
    await settle();
    expect(t.at('ds_a')).toBe('unknown');
    expect(t.at('ds_b')).toBe('unknown');
    // The read sent for the ids (the first one is superseded by it).
    for (const c of t.calls) c.resolve([{ id: 'ds_a' }]);
    await settle();
    expect(t.at('ds_a')).toBe('registered');
    expect(t.at('ds_b')).toBe('removed');
    t.stop();
  });

  it('stays unknown when the first read fails, and the poll settles it', async () => {
    const t = setup(['ds_a']);
    await settle();
    for (const c of t.calls) c.reject(new Error('503'));
    await settle();
    expect(t.at('ds_a')).toBe('unknown');
    await vi.advanceTimersByTimeAsync(5000);
    t.last().resolve([{ id: 'ds_a' }]);
    await settle();
    expect(t.at('ds_a')).toBe('registered');
    t.stop();
  });

  it('says removed once the dataset leaves the list while the page is open', async () => {
    const t = setup(['ds_a']);
    await settle();
    for (const c of t.calls) c.resolve([{ id: 'ds_a' }]);
    await settle();
    expect(t.at('ds_a')).toBe('registered');
    await vi.advanceTimersByTimeAsync(5000);
    expect(t.at('ds_a')).toBe('registered'); // no flicker while the poll is in flight
    t.last().resolve([]);
    await settle();
    expect(t.at('ds_a')).toBe('removed');
    t.stop();
  });

  it('never calls a new id removed when the read for it fails (the old list does not have it)', async () => {
    const t = setup(['ds_old']);
    await settle();
    for (const c of t.calls) c.resolve([]);
    await settle();
    expect(t.at('ds_old')).toBe('removed');
    // Exported again from another tab: the page learns a new id; the read for it fails.
    t.ids.set(['ds_new']);
    flushSync();
    t.last().reject(new Error('503'));
    await settle();
    expect(t.at('ds_new')).toBe('unknown');
    // The poll keeps trying, one read at a time.
    const before = t.calls.length;
    await vi.advanceTimersByTimeAsync(5000);
    expect(t.calls.length).toBe(before + 1);
    await vi.advanceTimersByTimeAsync(5000);
    expect(t.calls.length).toBe(before + 1);
    t.last().resolve([{ id: 'ds_new' }]);
    await settle();
    expect(t.at('ds_new')).toBe('registered');
    t.stop();
  });

  it('does not count an answer to a read sent before the id was learned', async () => {
    const t = setup([]);
    await settle();
    t.last().resolve([]);
    await settle();
    await vi.advanceTimersByTimeAsync(5000); // a poll goes out before the dataset is made
    const early = t.last();
    t.ids.set(['ds_new']);
    flushSync();
    const fresh = t.last();
    expect(fresh).not.toBe(early);
    early.resolve([]);
    await settle();
    expect(t.at('ds_new')).toBe('unknown');
    fresh.resolve([{ id: 'ds_new' }]);
    await settle();
    expect(t.at('ds_new')).toBe('registered');
    t.stop();
  });

  it('turns unknown again when a later read fails', async () => {
    const t = setup(['ds_a']);
    await settle();
    for (const c of t.calls) c.resolve([{ id: 'ds_a' }]);
    await settle();
    await vi.advanceTimersByTimeAsync(5000);
    t.last().reject(new Error('503'));
    await settle();
    expect(t.at('ds_a')).toBe('unknown');
    t.stop();
  });
});
