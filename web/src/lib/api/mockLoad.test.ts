// @vitest-environment happy-dom
// The demo's mock module (`?mock=1`) comes from the server like any chunk: while it cannot be loaded Studio is offline
// and tries again (index.ts, #useMock), and a newer connect cancels the wait of the one it replaces.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PROBE_DELAYS } from './reach';

describe('the mock module that cannot be loaded', () => {
  let loads = 0;
  let fail = true;
  let actual: unknown;
  beforeEach(async () => {
    actual ??= await vi.importActual('./mock');
    vi.useFakeTimers();
    vi.resetModules();
    window.history.replaceState(null, '', '/?mock=1');
    loads = 0;
    fail = true;
    vi.doMock('./mock', async () => {
      loads++;
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
    expect([hub.mode, hub.offline]).toEqual(['connecting', true]);
    const one = vi.getTimerCount();
    // A second connect (Settings › Reconnect) while the first waits to try again: the first's timer goes.
    await hub.connect();
    expect(vi.getTimerCount()).toBe(one);
    fail = false;
    const before = loads;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    // One try, the newer connect's: the replaced one never wakes up to try too.
    expect(loads - before).toBe(1);
    expect([hub.mode, hub.offline, hub.reason]).toEqual(['mock', false, 'forced']);
  });
});
