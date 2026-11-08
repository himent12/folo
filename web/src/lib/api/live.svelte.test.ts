// @vitest-environment happy-dom
import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Api, RunEventHandlers } from './client';
import { hub } from './index.svelte';
import { LiveRun, mergeBy } from './live.svelte';
import { MockApi } from './mock';
import type { Run, RunMetrics, Step } from './types';

const step = (n: number): Step => ({ epoch: 0, step: n, lr: 0.01, loss: { box: 1, cls: 1, dfl: 1, total: 3 }, imgs_per_s: 100, eta_s: 10 });

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

describe('mergeBy', () => {
  it('unions by key, preferring the newer list', () => {
    const a = [1, 2, 3].map(step);
    const b = [{ ...step(3), lr: 9 }, step(4)];
    const m = mergeBy(a, b, (s) => s.step);
    expect(m.map((s) => s.step)).toEqual([1, 2, 3, 4]);
    expect(m[2]!.lr).toBe(9);
  });
});

describe('LiveRun ordering', () => {
  let handlers: RunEventHandlers = {};
  let metrics = deferred<RunMetrics>();
  let snapshot: Run;
  const saved = hub.api;

  beforeEach(async () => {
    vi.useFakeTimers();
    snapshot = await new MockApi('normal', 0).run('run_7d1e');
    metrics = deferred<RunMetrics>();
    const fake: Partial<Api> = {
      mock: false,
      run: () => Promise.resolve({ ...snapshot, status: 'running', step: 2 }),
      runMetrics: () => metrics.promise,
      runEvents: (_id, h) => {
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

  it('keeps events that arrive while history loads, and newer status wins', async () => {
    let live!: LiveRun;
    const stop = $effect.root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    // The stream is open before history arrives.
    handlers.step?.(step(3));
    handlers.status?.({ status: 'stopping', error: null });
    metrics.resolve({ steps: [step(1), step(2)], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3]);
    expect(live.run?.status).toBe('stopping');
    expect(live.run?.step).toBe(3);

    // A reconnect whose history is older than what we already have never moves backwards.
    metrics = deferred<RunMetrics>();
    handlers.reconnect?.();
    handlers.step?.(step(4));
    metrics.resolve({ steps: [step(1), step(2), step(3)], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3, 4]);
    stop();
  });

  it('reloads history once the stream opens, so nothing emitted before the subscription is lost', async () => {
    let live!: LiveRun;
    const stop = $effect.root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    metrics.resolve({ steps: [step(1), step(2)], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2]);
    // Step 3 happened on the server before our subscription was live; the history fetched on open has it.
    metrics = deferred<RunMetrics>();
    handlers.connection?.('open');
    metrics.resolve({ steps: [step(1), step(2), step(3)], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3]);
    stop();
  });

  it('refreshes the snapshot on a terminal status even while history is loading', async () => {
    let calls = 0;
    (hub.api as { run: Api['run'] }).run = () => {
      calls++;
      return Promise.resolve({ ...snapshot, status: calls > 1 ? 'completed' : 'running', model_id: calls > 1 ? 'm_new' : null });
    };
    let live!: LiveRun;
    const stop = $effect.root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    metrics.resolve({ steps: [step(1)], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('completed');
    expect(live.run?.model_id).toBe('m_new');
    stop();
  });
});
