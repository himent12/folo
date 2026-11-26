// @vitest-environment happy-dom
// The demo's mock module (`?mock=1`) comes from the server like any chunk: while it cannot be loaded Studio is offline
// and tries again (index.ts, #useMock), and a newer connect cancels the wait of the one it replaces.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PROBE_DELAYS } from './reach';

describe('the mock module that cannot be loaded', () => {
  let loads = 0;
  let fail: boolean | Error | { raw: unknown } = true;
  let gate: Promise<void> | null = null;
  let actual: unknown;
  beforeEach(async () => {
    actual ??= await vi.importActual('./mock');
    vi.useFakeTimers();
    vi.resetModules();
    window.history.replaceState(null, '', '/?mock=1');
    loads = 0;
    fail = true;
    gate = null;
    vi.doMock('./mock', async () => {
      loads++;
      if (gate) await gate;
      if (fail instanceof Error) throw fail;
      if (typeof fail === 'object' && fail !== null && 'raw' in fail) throw fail.raw;
      if (fail) throw new TypeError('Failed to fetch dynamically imported module');
      return actual as object;
    });
  });
  afterEach(() => {
    vi.doUnmock('./mock');
    window.history.replaceState(null, '', '/');
    vi.useRealTimers();
  });

  it('says offline, tries again, and a newer connect cancels the wait of the one it replaces', async () => {
    const { hub } = await import('./index');
    await hub.connect();
    expect([hub.mode, hub.offline, hub.demoError]).toEqual(['connecting', true, null]);
    const one = vi.getTimerCount();
    // A second connect (Settings › Reconnect) while the first waits to try again: the first's timer goes.
    await hub.connect();
    expect(vi.getTimerCount()).toBe(one);
    fail = false;
    const before = loads;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    // One try, the newer connect's: the replaced one never wakes up to try too.
    expect(loads - before).toBe(1);
    expect([hub.mode, hub.offline, hub.reason, hub.demoError]).toEqual(['mock', false, 'forced', null]);
  });

  it('stops retrying and surfaces demoError (not Server offline) when the mock chunk throws while evaluating', async () => {
    const boom = new SyntaxError('Unexpected token <');
    fail = boom;
    const { hub } = await import('./index');
    await hub.connect();
    expect(hub.mode).toBe('connecting');
    expect(hub.offline).toBe(false);
    expect(hub.demoSlow).toBe(false);
    const err = hub.demoError as (Error & { cause?: unknown }) | null;
    expect(err === boom || err?.cause === boom).toBe(true);
    // No endless retry timers scheduled: only a reload can clear a cached module evaluation error.
    const before = loads;
    expect(vi.getTimerCount()).toBe(0);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]! * 3);
    expect(loads).toBe(before);
  });

  it('marks demoSlow when the mock chunk hangs past SLOW_CONNECT, then recovers once it finishes', async () => {
    let release!: () => void;
    gate = new Promise<void>((ok) => (release = ok));
    fail = false;
    const { hub, SLOW_CONNECT } = await import('./index');
    const connecting = hub.connect();
    expect(hub.demoSlow).toBe(false);

    await vi.advanceTimersByTimeAsync(SLOW_CONNECT);
    await connecting;
    expect([hub.mode, hub.offline, hub.demoSlow, hub.demoError]).toEqual(['connecting', false, true, null]);

    release();
    await vi.waitFor(() => expect(hub.mode).toBe('mock'));
    expect([hub.offline, hub.demoSlow, hub.demoError, hub.reason]).toEqual([false, false, null, 'forced']);
  });

  it('surfaces demoSlow when a retry after a failed fetch stalls, and clears it on recovery', async () => {
    const { hub, SLOW_CONNECT } = await import('./index');
    await hub.connect();
    expect([hub.mode, hub.offline, hub.demoSlow]).toEqual(['connecting', true, false]);

    // Next retry hangs open past SLOW_CONNECT while `hub.offline` is still true.
    let release!: () => void;
    gate = new Promise<void>((ok) => (release = ok));
    fail = false;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    await vi.advanceTimersByTimeAsync(SLOW_CONNECT);
    expect([hub.mode, hub.offline, hub.demoSlow, hub.demoError]).toEqual(['connecting', true, true, null]);

    release();
    await vi.waitFor(() => expect(hub.mode).toBe('mock'));
    expect([hub.offline, hub.demoSlow, hub.demoError]).toEqual([false, false, null]);
  });

  it('normalizes a null/undefined evaluation rejection so demoError is never swallowed', async () => {
    fail = { raw: null };
    const { hub } = await import('./index');
    await hub.connect();
    expect(hub.mode).toBe('connecting');
    expect(hub.offline).toBe(false);
    expect(hub.demoError).not.toBeNull();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('settles a superseded connect() immediately when loadMock() is hanging', async () => {
    let release!: () => void;
    gate = new Promise<void>((ok) => (release = ok));
    fail = false;
    const { hub } = await import('./index');
    let firstSettled = false;
    const first = hub.connect().then(() => {
      firstSettled = true;
    });
    // Start a second connect before SLOW_CONNECT elapses: the first must settle immediately.
    const second = hub.connect();
    await first;
    expect(firstSettled).toBe(true);

    release();
    await second;
    await vi.waitFor(() => expect(hub.mode).toBe('mock'));
  });
});
