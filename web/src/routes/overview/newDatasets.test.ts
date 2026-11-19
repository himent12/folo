// @vitest-environment happy-dom
// datasetPresence (sweep 4, #319 and its reviews): any good answer that has an id makes it registered; "removed" only
// once an answer requested after the page learned the id does not have it; a failed read changes nothing.
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

  it('is unknown while the first read is delayed, then registered by it; removed only by a read sent after', async () => {
    const t = setup([]);
    await settle();
    // The page learns its ids while the first read is on its way (it was sent alongside what names them).
    t.ids.set(['ds_a', 'ds_b']);
    expect(t.at('ds_a')).toBe('unknown');
    expect(t.at('ds_b')).toBe('unknown');
    expect(t.calls).toHaveLength(1);
    t.last().resolve([{ id: 'ds_a' }]);
    await settle();
    expect(t.at('ds_a')).toBe('registered');
    // The first read may predate ds_b: it is read again for it.
    expect(t.at('ds_b')).toBe('unknown');
    expect(t.calls).toHaveLength(2);
    t.last().resolve([{ id: 'ds_a' }]);
    await settle();
    expect(t.at('ds_b')).toBe('removed');
    t.stop();
  });

  // Review of #335, round 2: the read sent again for every new id held the link back (and cost a request) even when
  // the list already had the id.
  it('is registered by the first list that has the id, with no read after it', async () => {
    const t = setup(['ds_a']);
    await settle();
    expect(t.calls).toHaveLength(1);
    t.last().resolve([{ id: 'ds_a' }, { id: 'ds_b' }]);
    await settle();
    expect(t.at('ds_a')).toBe('registered');
    expect(t.calls).toHaveLength(1);
    // An id learned later that the list already has: registered at once, nothing read.
    t.ids.set(['ds_a', 'ds_b']);
    expect(t.at('ds_b')).toBe('registered');
    await settle();
    expect(t.calls).toHaveLength(1);
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
    t.last().resolve([{ id: 'ds_a' }]);
    await settle();
    expect(t.at('ds_a')).toBe('registered');
    await vi.advanceTimersByTimeAsync(5000);
    expect(t.at('ds_a')).toBe('registered'); // no flicker while the poll is in flight
    const polls = t.calls.length;
    t.last().resolve([]);
    await settle();
    // Review of #335, round 3: the poll was requested after the page learned the id, so its answer settles it alone.
    expect(t.at('ds_a')).toBe('removed');
    expect(t.calls).toHaveLength(polls);
    t.stop();
  });

  it('never calls a new id removed when the read for it fails (the old list does not have it)', async () => {
    const t = setup(['ds_old']);
    await settle();
    t.last().resolve([]);
    await settle();
    t.last().resolve([]);
    await settle();
    expect(t.at('ds_old')).toBe('removed');
    // Exported again from another tab: the page learns a new id; the read for it fails.
    const before = t.calls.length;
    t.ids.set(['ds_new']);
    flushSync();
    expect(t.calls.length).toBe(before + 1);
    t.last().reject(new Error('503'));
    await settle();
    expect(t.at('ds_new')).toBe('unknown');
    // The poll keeps trying, one read at a time.
    await vi.advanceTimersByTimeAsync(5000);
    expect(t.calls.length).toBe(before + 2);
    await vi.advanceTimersByTimeAsync(5000);
    expect(t.calls.length).toBe(before + 2);
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
    expect(t.last()).toBe(early); // the poll in flight may have the id: it is waited for, not cut short
    early.resolve([]);
    await settle();
    expect(t.at('ds_new')).toBe('unknown');
    const fresh = t.last();
    expect(fresh).not.toBe(early);
    fresh.resolve([{ id: 'ds_new' }]);
    await settle();
    expect(t.at('ds_new')).toBe('registered');
    t.stop();
  });

  // Review of #335, round 2: a failed read changes nothing, either way.
  it('keeps what the last good answer said when a later read fails', async () => {
    const t = setup(['ds_a', 'ds_b']);
    await settle();
    t.last().resolve([{ id: 'ds_a' }]);
    await settle();
    t.last().resolve([{ id: 'ds_a' }]);
    await settle();
    expect([t.at('ds_a'), t.at('ds_b')]).toEqual(['registered', 'removed']);
    await vi.advanceTimersByTimeAsync(5000);
    t.last().reject(new Error('503'));
    await settle();
    expect([t.at('ds_a'), t.at('ds_b')]).toEqual(['registered', 'removed']);
    t.stop();
  });
});
