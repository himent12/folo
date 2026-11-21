// @vitest-environment happy-dom
// A Resource's first fetch can take the answer of another Resource that reads the same thing (`startFrom`, #309: the
// cold Overview asked for the run list and the system info twice each), and only a success: what fails, stalls or is
// old is asked for again, so what a page reports stays its own.
import { createRoot, flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from './index';
import { Resource, START_FRESH } from './resource';

const root = (fn: () => void) => createRoot((dispose) => (fn(), dispose));

interface Call<T> {
  resolve: (v: T) => void;
  reject: (e: unknown) => void;
  signal: AbortSignal;
}

function reads<T>() {
  const calls: Call<T>[] = [];
  const fetcher = (signal: AbortSignal) =>
    new Promise<T>((resolve, reject) => {
      calls.push({ resolve, reject, signal });
    });
  return { calls, fetcher };
}

const settle = () => vi.advanceTimersByTimeAsync(0);

describe('Resource startFrom', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    hub.mode = 'mock';
  });
  afterEach(() => vi.useRealTimers());

  it('waits on the read the other is loading, and gets a copy of its own', async () => {
    const lead = reads<{ n: number[] }>();
    const own = reads<{ n: number[] }>();
    let a!: Resource<{ n: number[] }>;
    let b!: Resource<{ n: number[] }>;
    const stopA = root(() => (a = new Resource(lead.fetcher)));
    flushSync();
    const stopB = root(() => (b = new Resource(own.fetcher, { startFrom: () => a })));
    flushSync();
    expect(own.calls).toHaveLength(0);
    expect(b.loading).toBe(true);
    lead.calls[0]!.resolve({ n: [1] });
    await settle();
    expect(b.data).toEqual({ n: [1] });
    expect(b.data).not.toBe(a.data);
    expect(b.loading).toBe(false);
    expect(own.calls).toHaveLength(0);
    stopA();
    stopB();
  });

  it('takes an answer that has just arrived, with its time; an older one is asked for again', async () => {
    const lead = reads<number>();
    let a!: Resource<number>;
    const stopA = root(() => (a = new Resource(lead.fetcher)));
    flushSync();
    lead.calls[0]!.resolve(1);
    await settle();
    const at = a.updated;
    await vi.advanceTimersByTimeAsync(START_FRESH - 1);

    const fresh = reads<number>();
    let b!: Resource<number>;
    const stopB = root(() => (b = new Resource(fresh.fetcher, { startFrom: () => a })));
    flushSync();
    expect(fresh.calls).toHaveLength(0);
    expect([b.data, b.updated, b.loading]).toEqual([1, at, false]);

    await vi.advanceTimersByTimeAsync(1);
    const old = reads<number>();
    let c!: Resource<number>;
    const stopC = root(() => (c = new Resource(old.fetcher, { startFrom: () => a })));
    flushSync();
    expect(old.calls).toHaveLength(1);
    expect(c.data).toBeUndefined();
    stopA();
    stopB();
    stopC();
  });

  it('asks for itself when the read it waits on fails, and reports its own outcome', async () => {
    const lead = reads<number>();
    const own = reads<number>();
    let a!: Resource<number>;
    let b!: Resource<number>;
    const stopA = root(() => (a = new Resource(lead.fetcher)));
    flushSync();
    const stopB = root(() => (b = new Resource(own.fetcher, { startFrom: () => a })));
    flushSync();
    expect(own.calls).toHaveLength(0); // it waits on the read that is out
    lead.calls[0]!.reject(new Error('the shell read failed'));
    await settle();
    expect((a.error as Error).message).toBe('the shell read failed');
    expect(own.calls).toHaveLength(1);
    expect(b.error).toBeUndefined();
    expect(b.loading).toBe(true);
    own.calls[0]!.reject(new Error('its own'));
    await settle();
    expect((b.error as Error).message).toBe('its own');
    stopA();
    stopB();
  });

  it('asks nothing once it is gone, when the read it waited on fails', async () => {
    const lead = reads<number>();
    const own = reads<number>();
    let a!: Resource<number>;
    const stopA = root(() => (a = new Resource(lead.fetcher)));
    flushSync();
    const stopB = root(() => void new Resource(own.fetcher, { startFrom: () => a }));
    flushSync();
    stopB();
    lead.calls[0]!.reject(new Error('failed'));
    await settle();
    expect(own.calls).toHaveLength(0);
    stopA();
  });

  it('stops lending a read once it is gone, while the read it waited on is still out', async () => {
    const lead = reads<number>();
    const waiting = reads<number>();
    const own = reads<number>();
    let a!: Resource<number>;
    let b!: Resource<number>;
    const stopA = root(() => (a = new Resource(lead.fetcher)));
    flushSync();
    const stopB = root(() => (b = new Resource(waiting.fetcher, { startFrom: () => a })));
    flushSync();
    expect(waiting.calls).toHaveLength(0); // B waits on A's read
    stopB();
    await settle();
    expect(waiting.calls).toHaveLength(0);
    const stopC = root(() => void new Resource(own.fetcher, { startFrom: () => b }));
    flushSync();
    expect(own.calls).toHaveLength(1); // B let go at once, without waiting for A's read
    stopA();
    stopC();
  });

  it('stops waiting on a stalled read at its first poll, and a late answer changes nothing', async () => {
    const lead = reads<number>();
    const own = reads<number>();
    let a!: Resource<number>;
    let b!: Resource<number>;
    const stopA = root(() => (a = new Resource(lead.fetcher)));
    flushSync();
    const stopB = root(() => (b = new Resource(own.fetcher, { poll: 100, startFrom: () => a })));
    flushSync();
    await vi.advanceTimersByTimeAsync(99);
    expect(own.calls).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(own.calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(300);
    expect(own.calls).toHaveLength(1); // its own read is out: the polls do not pile on
    own.calls[0]!.resolve(2);
    await settle();
    expect(b.data).toBe(2);
    lead.calls[0]!.resolve(1);
    await settle();
    expect(b.data).toBe(2);
    stopA();
    stopB();
  });

  it('does not take data whose refresh failed, or wait on a quiet refresh', async () => {
    const lead = reads<number>();
    let a!: Resource<number>;
    const stopA = root(() => (a = new Resource(lead.fetcher)));
    flushSync();
    lead.calls[0]!.resolve(1);
    await settle();
    const whileFresh = reads<number>();
    let fresh!: Resource<number>;
    const stopFresh = root(() => (fresh = new Resource(whileFresh.fetcher, { startFrom: () => a })));
    flushSync();
    expect(whileFresh.calls).toHaveLength(0); // the same answer, fresh and whole: taken as it is
    expect(fresh.data).toBe(1);
    stopFresh();

    void a.refresh();
    lead.calls[1]!.reject(new Error('offline'));
    await settle();
    expect(a.stale).toBeInstanceOf(Error);
    const afterFailure = reads<number>();
    const stopB = root(() => void new Resource(afterFailure.fetcher, { startFrom: () => a }));
    flushSync();
    expect(afterFailure.calls).toHaveLength(1);

    void a.refresh();
    lead.calls[2]!.resolve(2);
    await settle();
    await vi.advanceTimersByTimeAsync(START_FRESH);
    void a.refresh(); // quiet: its data is shown meanwhile, and is too old to start from
    const duringRefresh = reads<number>();
    const stopC = root(() => void new Resource(duringRefresh.fetcher, { startFrom: () => a }));
    flushSync();
    expect(duringRefresh.calls).toHaveLength(1);
    stopA();
    stopB();
    stopC();
  });

  it('starts from nothing of another server, and only its first fetch follows', async () => {
    const lead = reads<number>();
    const own = reads<number>();
    let a!: Resource<number>;
    let b!: Resource<number>;
    const stopA = root(() => (a = new Resource(lead.fetcher)));
    flushSync();
    lead.calls[0]!.resolve(1);
    await settle();
    const stopB = root(() => (b = new Resource(own.fetcher, { startFrom: () => a })));
    flushSync();
    expect(b.data).toBe(1);
    hub.epoch++; // a reconnect: both read the new server, each asking
    flushSync();
    expect(lead.calls).toHaveLength(2);
    expect(own.calls).toHaveLength(1);
    expect(b.data).toBeUndefined();
    stopB();

    // One that is gone keeps the last server's answer: a Resource on the next server does not start from it.
    lead.calls[1]!.resolve(2);
    await settle();
    expect(a.data).toBe(2);
    stopA();
    hub.epoch++;
    const next = reads<number>();
    let c!: Resource<number>;
    const stopC = root(() => (c = new Resource(next.fetcher, { startFrom: () => a })));
    flushSync();
    expect(next.calls).toHaveLength(1);
    expect(c.data).toBeUndefined();
    stopC();
  });
});
