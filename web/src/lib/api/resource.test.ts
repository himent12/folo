// @vitest-environment happy-dom
import { createRoot, flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from './index';
import { Resource } from './resource';

/** Port of `root(fn)`: runs `fn` in a root of its own and returns its teardown. */
const root = (fn: () => void) => createRoot((dispose) => (fn(), dispose));

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
    const stop = root(() => {
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
    const stop = root(() => {
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
    const stop = root(() => {
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

  it("drops the last server's data on a reconnect: the new one's failure shows, not that data (review R4)", async () => {
    let fail = false;
    let r!: Resource<string>;
    const stop = root(() => {
      r = new Resource(() => (fail ? Promise.reject(new Error('HTTP 500')) : Promise.resolve('server A')));
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(r.data).toBe('server A');

    fail = true;
    hub.epoch++;
    flushSync();
    expect([r.data, r.updated, r.loading]).toEqual([undefined, null, true]);
    await vi.advanceTimersByTimeAsync(0);
    expect(r.data).toBeUndefined();
    expect((r.error as Error).message).toBe('HTTP 500');
    stop();
  });

  it('ignores results that arrive after cleanup', async () => {
    const d = deferred<number>();
    let r!: Resource<number>;
    const stop = root(() => {
      r = new Resource(() => d.promise);
    });
    flushSync();
    stop();
    d.resolve(5);
    await vi.advanceTimersByTimeAsync(0);
    expect(r.data).toBeUndefined();
  });

  it('ignores refresh() and invalidate() calls after cleanup without issuing leaked requests', async () => {
    let calls = 0;
    let r!: Resource<number>;
    const stop = root(() => {
      r = new Resource(() => {
        calls++;
        return Promise.resolve(calls);
      });
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(calls).toBe(1);
    expect(r.data).toBe(1);

    stop();
    await r.refresh();
    await r.invalidate();
    expect(calls).toBe(1);
    expect(r.data).toBe(1);
    expect(r.refreshing).toBe(false);
  });

  it('turns synchronous fetcher throws into error state without crashing reactive effects', async () => {
    let fail = true;
    let r!: Resource<string>;
    const stop = root(() => {
      r = new Resource(() => {
        if (fail) throw new Error('sync boom');
        return Promise.resolve('ok');
      });
    });
    expect(() => flushSync()).not.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    expect((r.error as Error)?.message).toBe('sync boom');
    expect(r.loading).toBe(false);

    fail = false;
    await r.refresh();
    expect(r.error).toBeUndefined();
    expect(r.data).toBe('ok');
    stop();
  });

  it('marks disposed synchronously before first flush and consumes rejections if a refresh fetcher disposes and throws', async () => {
    let calls = 0;
    let r!: Resource<number>;
    const stop = root(() => {
      r = new Resource(() => {
        calls++;
        return Promise.resolve(1);
      });
    });
    // Dispose BEFORE flushSync(): synchronous onCleanup must prevent later refresh()/invalidate() fetches.
    stop();
    await r.refresh();
    await r.invalidate();
    expect(calls).toBe(0);

    // If a refresh fetcher synchronously disposes its owner and throws, no unhandled rejection escapes.
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
      let disposeNow!: () => void;
      let throwOnSecond = false;
      let r2!: Resource<number>;
      disposeNow = root(() => {
        r2 = new Resource(() => {
          if (throwOnSecond) {
            disposeNow();
            throw new Error('disposed mid-refresh');
          }
          return Promise.resolve(7);
        });
      });
      flushSync();
      await vi.advanceTimersByTimeAsync(0);
      expect(r2.data).toBe(7);

      throwOnSecond = true;
      await r2.refresh();
      await vi.advanceTimersByTimeAsync(0);
      expect(unhandled).toEqual([]);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });

  it('clears previous server data on epoch change while connecting, surfaces connectError, and delegates refresh() to hub.connect()', async () => {
    let r!: Resource<string>;
    const stop = root(() => {
      r = new Resource(() => Promise.resolve('server A'));
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(r.data).toBe('server A');

    // Switching to another server that refuses access (`folo serve --token`): epoch bumps while mode is connecting.
    const refused = new Error('token required');
    const errSpy = vi.spyOn(hub, 'connectError', 'get').mockReturnValue(refused as never);
    const connectSpy = vi.spyOn(hub, 'connect').mockResolvedValue(undefined);
    try {
      hub.mode = 'connecting';
      hub.epoch++;
      flushSync();
      expect(r.data).toBeUndefined();
      expect(r.loading).toBe(false);
      expect(r.error).toBe(refused);

      // Clicking Retry while still connecting calls hub.connect() instead of querying the previous server's Api.
      await r.refresh();
      expect(connectSpy).toHaveBeenCalledTimes(1);
    } finally {
      errSpy.mockRestore();
      connectSpy.mockRestore();
      hub.mode = 'mock';
      stop();
    }
  });
});
