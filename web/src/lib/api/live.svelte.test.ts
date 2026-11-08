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
      runLogs: () => Promise.resolve({ lines: [] }),
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

  it('retries a failed history sync until it succeeds', async () => {
    let live!: LiveRun;
    const stop = $effect.root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    metrics.resolve({ steps: [step(1), step(2)], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    // The reload on open fails once...
    let fail = true;
    (hub.api as { runMetrics: Api['runMetrics'] }).runMetrics = () =>
      fail ? Promise.reject(new Error('flaky')) : Promise.resolve({ steps: [step(1), step(2), step(3)], epochs: [] });
    handlers.connection?.('open');
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2]);
    // ...and the retry fills the gap.
    fail = false;
    await vi.advanceTimersByTimeAsync(1000);
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

describe('LiveRun resume', () => {
  let handlers: RunEventHandlers = {};
  let snapshot: Run;
  let serverStatus: Run['status'] = 'stopped';
  let history: RunMetrics;
  /** When set, snapshots never answer: the test sees only what events and the resume response did. */
  let hold = false;
  const saved = hub.api;
  const at = (n: number, epoch: number, total = 3): Step => ({ ...step(n), epoch, loss: { box: 1, cls: 1, dfl: 1, total } });
  const ep = (epoch: number) => ({ epoch, train_loss: { box: 1, cls: 1, dfl: 1, total: 3 }, val: { map50: 0.5, map50_95: 0.3, precision: 0.5, recall: 0.5, per_class: [] }, lr: 0.01, seconds: 1 });

  beforeEach(async () => {
    vi.useFakeTimers();
    serverStatus = 'stopped';
    hold = false;
    snapshot = await new MockApi('normal', 0).run('run_7d1e');
    // Epochs are 1-based (folo-api). Stopped in epoch 2: epoch 1 finished (steps 1-2), epoch 2 had steps 3-4. The
    // server keeps those interrupted steps in /metrics until the resumed run writes their replacements.
    history = { steps: [at(1, 1), at(2, 1), at(3, 2), at(4, 2)], epochs: [ep(1)] as never };
    const fake: Partial<Api> = {
      mock: false,
      run: () => (hold ? new Promise<Run>(() => {}) : Promise.resolve({ ...snapshot, status: serverStatus, step: 4, epoch: 2, best: { epoch: 1, map50_95: 0.3 } })),
      runMetrics: () => Promise.resolve(history),
      runLogs: () => Promise.resolve({ lines: [] }),
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

  it('rewinds to the last finished epoch so repeated steps are shown', async () => {
    let live!: LiveRun;
    const stop = $effect.root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3, 4]);
    serverStatus = 'queued';
    // The response still carries the interrupted step count; the rewound progress wins.
    await live.resume(() => Promise.resolve({ ...snapshot, status: 'queued', step: 4, epoch: 2 }));
    expect(live.steps.map((s) => s.step)).toEqual([1, 2]);
    expect(live.run?.epoch).toBe(1);
    expect(live.run?.step).toBe(2);
    // The resync after resuming returns the interrupted steps 3-4 (and a snapshot still counting step 4): they stay
    // out, and progress stays rewound.
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2]);
    expect(live.run?.step).toBe(2);
    expect(live.run?.epoch).toBe(1);
    // The resumed epoch repeats step 3 with new values: it is shown, not dropped as old.
    handlers.step?.(at(3, 2, 90));
    expect(live.steps.at(-1)?.loss.total).toBe(90);
    stop();
  });

  it('keeps a status event that arrives before the resume response', async () => {
    let live!: LiveRun;
    const stop = $effect.root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    hold = true;
    const response = deferred<Run>();
    const pending = live.resume(() => response.promise);
    // The worker starts first: the stream says running before the HTTP response (queued) lands.
    serverStatus = 'running';
    handlers.status?.({ status: 'running', error: null });
    expect(live.steps.map((s) => s.step)).toEqual([1, 2]);
    response.resolve({ ...snapshot, status: 'queued', step: 4, epoch: 2 });
    await pending;
    expect(live.run?.status).toBe('running');
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('running');
    stop();
  });

  it('accepts history again once it reaches past the interrupted attempt (reconnect after the run went on)', async () => {
    let live!: LiveRun;
    const stop = $effect.root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    serverStatus = 'queued';
    await live.resume(() => Promise.resolve({ ...snapshot, status: 'queued', step: 4, epoch: 2 }));
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2]);
    // We missed the resumed steps while disconnected; the server's history now has replacements 3-4 and new 5-6.
    history = { steps: [at(1, 1), at(2, 1), at(3, 2, 70), at(4, 2, 60), at(5, 3), at(6, 3)], epochs: [ep(1), ep(2)] as never };
    serverStatus = 'completed';
    handlers.reconnect?.();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(live.steps[2]?.loss.total).toBe(70);
    stop();
  });
});
