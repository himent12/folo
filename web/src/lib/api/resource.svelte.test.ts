// @vitest-environment happy-dom
import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from './index.svelte';
import { Resource } from './resource.svelte';

const deferred = <T,>() => {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
};

describe('Resource', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    hub.mode = 'mock';
  });
  afterEach(() => vi.useRealTimers());

  it('does not let polls abort a slow refresh', async () => {
    const calls: { resolve: (v: number) => void }[] = [];
    let r!: Resource<number>;
    const stop = $effect.root(() => {
      r = new Resource(() => {
        const d = deferred<number>();
        calls.push(d);
        return d.promise;
      }, { poll: 100 });
    });
    flushSync();
    calls[0]!.resolve(1);
    await vi.advanceTimersByTimeAsync(0);
    expect(r.data).toBe(1);
    await vi.advanceTimersByTimeAsync(100); // poll #1 starts a slow refresh
    expect(calls).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(500); // further polls must not pile on while it is in flight
    expect(calls).toHaveLength(2);
    calls[1]!.resolve(2);
    await vi.advanceTimersByTimeAsync(0);
    expect(r.data).toBe(2);
    stop();
  });

  it('keeps a known error on screen through polls (no loading flash, so offline actions stay gated)', async () => {
    const calls: { reject: (e: unknown) => void }[] = [];
    let r!: Resource<number>;
    const stop = $effect.root(() => {
      r = new Resource(() => {
        let reject!: (e: unknown) => void;
        const promise = new Promise<number>((_, j) => (reject = j));
        calls.push({ reject });
        return promise;
      }, { poll: 100 });
    });
    flushSync();
    calls[0]!.reject(new Error('offline'));
    await vi.advanceTimersByTimeAsync(0);
    expect(r.error).toBeInstanceOf(Error);
    await vi.advanceTimersByTimeAsync(100); // a poll is in flight
    expect(calls).toHaveLength(2);
    expect(r.error).toBeInstanceOf(Error);
    expect(r.loading).toBe(false);
    calls[1]!.reject(new Error('still offline'));
    await vi.advanceTimersByTimeAsync(0);
    expect((r.error as Error).message).toBe('still offline');
    // A Retry (refresh) does go back to loading: the press has an answer on screen.
    void r.refresh();
    expect(r.loading).toBe(true);
    stop();
  });

  it('does not poll while the hub reconnects, and aborts on destroy', async () => {

    let calls = 0;
    let lastSignal: AbortSignal | undefined;
    let r!: Resource<number>;
    const stop = $effect.root(() => {
      r = new Resource((signal) => {
        calls++;
        lastSignal = signal;
        return Promise.resolve(calls);
      }, { poll: 100 });
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(r.data).toBe(1);
    hub.mode = 'connecting';
    flushSync();
    await vi.advanceTimersByTimeAsync(500);
    expect(calls).toBe(1);
    hub.mode = 'mock';
    stop();
    expect(lastSignal?.aborted).toBe(true);
  });

  it('ignores results that arrive after cleanup', async () => {
    const d = deferred<number>();
    let r!: Resource<number>;
    const stop = $effect.root(() => {
      r = new Resource(() => d.promise);
    });
    flushSync();
    stop();
    d.resolve(5);
    await vi.advanceTimersByTimeAsync(0);
    expect(r.data).toBeUndefined();
  });
});
