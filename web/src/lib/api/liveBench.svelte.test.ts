// @vitest-environment happy-dom
import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, type Api, type BenchEventHandlers } from './client';
import { hub } from './index.svelte';
import { applyById, LiveBench, RESYNC_MS } from './liveBench.svelte';
import { STREAM_GRACE_MS } from './live.svelte';
import { MockApi } from './mock';
import type { Bench, BenchEntry, BenchLoop } from './types';

const entry = (id: string, status: BenchEntry['status'], total: number | null = null, created = '2026-11-09T10:00:00Z'): BenchEntry =>
  ({ id, bench_id: 'bn_1', name: id, status, created, score: total === null ? null : { total, partial: false, axes: [] }, progress: { done: 0, total: 10, condition: null, severity: null } }) as unknown as BenchEntry;
const loop = (id: string, created: string, status: BenchLoop['status'] = 'running'): BenchLoop => ({ id, bench_id: 'bn_1', name: id, status, created }) as unknown as BenchLoop;
const bench = (name: string, entries = 0): Bench => ({ id: 'bn_1', name, entries }) as unknown as Bench;

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

describe('applyById', () => {
  it('replaces in place, removes, and adds new ones at the end', () => {
    const out = applyById([entry('a', 'done'), entry('b', 'running'), entry('c', 'failed')], [
      ['b', entry('b', 'done')],
      ['c', null],
      ['d', entry('d', 'queued')],
    ]);
    expect(out.map((e) => `${e.id} ${e.status}`)).toEqual(['a done', 'b done', 'd queued']);
  });
});

describe('LiveBench ordering', () => {
  let handlers: BenchEventHandlers = {};
  let entries = deferred<BenchEntry[]>();
  let snapshot: Bench = bench('first');
  const loads = vi.fn();
  const saved = hub.api;

  beforeEach(() => {
    vi.useFakeTimers();
    entries = deferred<BenchEntry[]>();
    snapshot = bench('first');
    loads.mockReset();
    const fake: Partial<Api> = {
      mock: false,
      bench: () => Promise.resolve(snapshot),
      benchEntries: () => {
        loads();
        return entries.promise;
      },
      benchLoops: () => Promise.resolve([loop('l1', '2026-11-01T00:00:00Z', 'done')]),
      benchEvents: (_id, h) => {
        handlers = h;
        return () => {};
      },
    };
    hub.api = fake as Api;
    hub.mode = 'live';
  });
  afterEach(() => {
    hub.api = saved;
    vi.useRealTimers();
  });

  function mount() {
    let live!: LiveBench;
    const stop = $effect.root(() => {
      live = new LiveBench(() => 'bn_1');
    });
    flushSync();
    return { live, stop };
  }

  it('applies entry, loop and removal events heard during the load over its answer, in leaderboard order', async () => {
    const { live, stop } = mount();
    expect(live.loading).toBe(true);
    handlers.entry?.(entry('b', 'done', 80));
    handlers.removed?.({ id: 'c' });
    handlers.loop?.(loop('l2', '2026-11-09T00:00:00Z'));
    entries.resolve([entry('a', 'done', 70), entry('b', 'running'), entry('c', 'failed')]);
    await vi.advanceTimersByTimeAsync(0);
    expect(live.loading).toBe(false);
    // b finished with the higher score while the list was on its way: it leads.
    expect(live.entries.map((e) => `${e.id} ${e.status}`)).toEqual(['b done', 'a done']);
    expect(live.loops.map((l) => l.id)).toEqual(['l2', 'l1']);
    stop();
  });

  it('keeps the benchmark of a `bench` event over an older snapshot, and loads the lists again after it', async () => {
    const { live, stop } = mount();
    handlers.bench?.(bench('renamed', 3));
    entries.resolve([entry('a', 'done', 70)]);
    await vi.advanceTimersByTimeAsync(0);
    expect(live.bench?.name).toBe('renamed');
    expect(loads).toHaveBeenCalledTimes(1);
    // A subscriber that fell behind gets the current `bench`: the lists load again, once for events close together.
    entries = deferred<BenchEntry[]>();
    handlers.bench?.(bench('renamed', 4));
    handlers.bench?.(bench('renamed', 4));
    await vi.advanceTimersByTimeAsync(RESYNC_MS);
    entries.resolve([entry('a', 'done', 70), entry('z', 'queued')]);
    await vi.advanceTimersByTimeAsync(0);
    expect(loads).toHaveBeenCalledTimes(2);
    expect(live.entries.map((e) => e.id)).toEqual(['a', 'z']);
    stop();
  });

  it("shows an action's answer, except for an entry the stream told of after the action began", async () => {
    const { live, stop } = mount();
    entries.resolve([entry('a', 'running')]);
    await vi.advanceTimersByTimeAsync(0);
    let mark = live.mark();
    handlers.entry?.(entry('a', 'stopped'));
    live.putEntry(entry('a', 'stopping'), mark);
    expect(live.entries[0]?.status).toBe('stopped');
    mark = live.mark();
    live.putEntry(entry('n', 'queued', null, '2026-11-09T11:00:00Z'), mark);
    expect(live.entries.map((e) => e.id)).toEqual(['n', 'a']);
    live.dropEntry('n');
    expect(live.entries.map((e) => e.id)).toEqual(['a']);
    stop();
  });

  it("keeps a bench event that arrived while an action was answered, over the action's older answer", async () => {
    const { live, stop } = mount();
    entries.resolve([]);
    await vi.advanceTimersByTimeAsync(0);
    const answer = deferred<Bench>();
    const acting = live.act(() => answer.promise);
    handlers.bench?.(bench('from the stream'));
    answer.resolve(bench('from the answer'));
    await acting;
    expect(live.bench?.name).toBe('from the stream');
    await live.act(() => Promise.resolve(bench('saved')));
    expect(live.bench?.name).toBe('saved');
    stop();
  });

  it('loads again when the stream opens and after a reconnect, and calls a stream lost only after the grace', async () => {
    const { live, stop } = mount();
    entries.resolve([]);
    await vi.advanceTimersByTimeAsync(0);
    expect(loads).toHaveBeenCalledTimes(1);
    handlers.connection?.('open');
    await vi.advanceTimersByTimeAsync(RESYNC_MS);
    expect(loads).toHaveBeenCalledTimes(2);
    handlers.connection?.('reconnecting');
    // A dropped stream loads once (a deleted benchmark answers 404 there, see below).
    await vi.advanceTimersByTimeAsync(RESYNC_MS);
    expect(loads).toHaveBeenCalledTimes(3);
    await vi.advanceTimersByTimeAsync(STREAM_GRACE_MS - RESYNC_MS - 1);
    expect(live.lost).toBeNull();
    await vi.advanceTimersByTimeAsync(1);
    expect(live.lost).not.toBeNull();
    handlers.connection?.('open');
    handlers.reconnect?.();
    await vi.advanceTimersByTimeAsync(0);
    expect(loads).toHaveBeenCalledTimes(4);
    expect(live.lost).toBeNull();
    stop();
  });

  it('notices a benchmark deleted elsewhere: a 404 on a load shows it gone, and nothing retries', async () => {
    const { live, stop } = mount();
    entries.resolve([entry('a', 'done', 70)]);
    await vi.advanceTimersByTimeAsync(0);
    expect(live.bench?.name).toBe('first');
    const gone = new ApiError(404, 'bench_not_found', 'No benchmark with id bn_1.');
    hub.api.bench = () => Promise.reject(gone);
    handlers.connection?.('reconnecting');
    await vi.advanceTimersByTimeAsync(RESYNC_MS);
    expect(live.error).toBe(gone);
    expect(live.bench).toBeUndefined();
    expect(live.entries).toEqual([]);
    const n = loads.mock.calls.length;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(loads.mock.calls.length).toBe(n);
    stop();
  });

  it('shows a first load that failed as the error, even after a bench event, never as an empty board', async () => {
    const { live, stop } = mount();
    handlers.bench?.(bench('from the stream'));
    const down = new ApiError(500, 'internal', 'boom');
    entries = deferred<BenchEntry[]>();
    hub.api.benchEntries = () => Promise.reject(down);
    live.retry();
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.error).toBe(down);
    expect(live.bench).toBeUndefined();
    expect(live.listed).toBe(false);
    handlers.bench?.(bench('again'));
    expect(live.bench).toBeUndefined();
    stop();
  });

  it('says so when the benchmark cannot be loaded, and loads again on retry', async () => {
    hub.api = { ...hub.api, bench: () => Promise.reject(new Error('gone')) } as Api;
    const { live, stop } = mount();
    entries.resolve([]);
    await vi.advanceTimersByTimeAsync(0);
    expect(live.error).toBeInstanceOf(Error);
    expect(live.loading).toBe(false);
    hub.api = { ...hub.api, bench: () => Promise.resolve(bench('back')) } as Api;
    live.retry();
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.bench?.name).toBe('back');
    stop();
  });
});

describe('LiveBench with the mock', () => {
  const saved = hub.api;
  afterEach(() => {
    hub.api = saved;
    vi.useRealTimers();
  });

  it('follows the running entry through its conditions, and the leaderboard when it finishes', async () => {
    vi.useFakeTimers({ now: Date.parse('2026-11-09T12:00:00Z') });
    hub.api = new MockApi('normal', 0);
    hub.mode = 'mock';
    let live!: LiveBench;
    const stop = $effect.root(() => {
      live = new LiveBench(() => 'bn_3c90');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(400);
    expect(live.bench?.name).toBe('Fasteners val');
    const run = () => live.entries.find((e) => e.id === 'be_f108')!;
    expect(run().status).toBe('running');
    const before = run().progress.done;
    await vi.advanceTimersByTimeAsync(5000);
    expect(run().progress.done).toBeGreaterThan(before);
    await vi.advanceTimersByTimeAsync(70_000);
    expect(run().status).toBe('done');
    // Ranked among the finished entries now: the complete ones by score, then the partial ones by score.
    const done = live.entries.filter((e) => e.status === 'done');
    const complete = done.filter((e) => !e.score!.partial);
    const partial = done.filter((e) => e.score!.partial);
    expect(done.map((e) => e.id)).toEqual([...complete, ...partial].map((e) => e.id));
    for (const g of [complete, partial]) expect(g.map((e) => e.score!.total)).toEqual([...g.map((e) => e.score!.total)].sort((a, b) => b - a));
    expect(done.find((e) => e.id === 'be_f108')!.score!.partial).toBe(false);
    expect(live.bench?.entries).toBe(live.entries.length);
    stop();
  });
});
