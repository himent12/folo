// @vitest-environment happy-dom
import { createRoot, flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, type Api, type RunEventHandlers } from './client';
import { hub } from './index';
import { keepStepsPerEpoch, LiveRun, mergeBy, STREAM_GRACE_MS, streamedRun } from './live';
import { MockApi } from './mock';
import type { Run, RunLogs, RunMetrics, Step } from './types';

/** Port of `root(fn)`: runs `fn` in a root of its own and returns its teardown. */
const root = (fn: () => void) => createRoot((dispose) => (fn(), dispose));

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

describe('keepStepsPerEpoch', () => {
  const run = (id: string, spe: number) => ({ id, steps_per_epoch: spe }) as Run;
  it("keeps a learned figure over a snapshot that still says 0, takes the snapshot's own, never another run's", () => {
    expect(keepStepsPerEpoch(run('a', 0), run('a', 12)).steps_per_epoch).toBe(12);
    expect(keepStepsPerEpoch(run('a', 10), run('a', 12)).steps_per_epoch).toBe(10);
    expect(keepStepsPerEpoch(run('b', 0), run('a', 12)).steps_per_epoch).toBe(0);
    expect(keepStepsPerEpoch(run('b', 0), null).steps_per_epoch).toBe(0);
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

  it('calls the stream lost only after the grace, timed from its first error, as of the last event heard', async () => {
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    metrics.resolve({ steps: [], epochs: [] });
    handlers.connection?.('open');
    await vi.advanceTimersByTimeAsync(1000);
    handlers.step?.(step(3));
    const heard = Date.now();
    await vi.advanceTimersByTimeAsync(1000);
    handlers.connection?.('reconnecting');
    await vi.advanceTimersByTimeAsync(STREAM_GRACE_MS - 1000);
    // EventSource reports every failed retry: that must not restart the grace.
    handlers.connection?.('reconnecting');
    expect(live.lost).toBeNull();
    await vi.advanceTimersByTimeAsync(1000);
    expect(live.lost).toBe(heard);
    // Back, but what the view holds is from before the outage until the reconnect's history load applies; a slow
    // (or failing) load keeps it last known.
    metrics = deferred<RunMetrics>();
    handlers.connection?.('open');
    handlers.reconnect?.();
    await vi.advanceTimersByTimeAsync(5000);
    expect(live.lost).toBe(heard);
    metrics.resolve({ steps: [step(1), step(2), step(3), step(4)], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.lost).toBeNull();
    // A blip shorter than the grace never shows.
    handlers.connection?.('reconnecting');
    await vi.advanceTimersByTimeAsync(STREAM_GRACE_MS - 100);
    handlers.connection?.('open');
    await vi.advanceTimersByTimeAsync(1000);
    expect(live.lost).toBeNull();
    stop();
  });

  it('notices a run deleted elsewhere: a dropped stream whose run answers 404 shows it gone, and the stream closes', async () => {
    const closes = vi.fn();
    hub.api.runEvents = (_id, h) => {
      handlers = h;
      return closes;
    };
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    // Disposed even when an assertion fails, so a LiveRun left polling cannot break the tests after this one.
    try {
      flushSync();
      metrics.resolve({ steps: [step(1)], epochs: [] });
      handlers.connection?.('open');
      await vi.advanceTimersByTimeAsync(0);
      expect(live.run?.id).toBe('run_7d1e');
      // A blip with the run still there changes nothing.
      handlers.connection?.('reconnecting');
      await vi.advanceTimersByTimeAsync(0);
      expect(live.run?.id).toBe('run_7d1e');
      expect(live.error).toBeUndefined();
      handlers.connection?.('open');
      // Deleted in another tab: the stream drops and answers 404 from then on.
      const gone = new ApiError(404, 'run_not_found', 'No run with id run_7d1e.');
      const asked = vi.fn(() => Promise.reject(gone));
      hub.api.run = asked;
      handlers.connection?.('reconnecting');
      await vi.advanceTimersByTimeAsync(0);
      expect(live.error).toBe(gone);
      expect(live.run).toBeUndefined();
      expect(live.connection).toBe('idle');
      expect(live.lost).toBeNull();
      expect(closes).toHaveBeenCalledTimes(1);
      // Nothing retries: neither the stream's later errors nor the timers ask again.
      const n = asked.mock.calls.length;
      handlers.connection?.('reconnecting');
      await vi.advanceTimersByTimeAsync(60_000);
      expect(asked.mock.calls.length).toBe(n);
      expect(live.lost).toBeNull();
    } finally {
      stop();
    }
  });

  it("lends its run to the shell (streamedRun) while it is up, never while it loads or is lost, nor after unmount", async () => {
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    expect(streamedRun('run_7d1e')).toBeUndefined();
    metrics.resolve({ steps: [], epochs: [] });
    handlers.connection?.('open');
    await vi.advanceTimersByTimeAsync(0);
    handlers.step?.({ ...step(3), epoch: 59 });
    await vi.advanceTimersByTimeAsync(20);
    expect(streamedRun('run_7d1e')?.epoch).toBe(59);
    expect(streamedRun('run_other')).toBeUndefined();
    // Down past the grace: the shell falls back to its own (polled) source.
    handlers.connection?.('reconnecting');
    await vi.advanceTimersByTimeAsync(STREAM_GRACE_MS);
    expect(live.lost).not.toBeNull();
    expect(streamedRun('run_7d1e')).toBeUndefined();
    stop();
    expect(streamedRun('run_7d1e')).toBeUndefined();
  });

  it('calls a reopened stream current again on its first step, even while the history load is still pending', async () => {
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    metrics.resolve({ steps: [], epochs: [] });
    handlers.connection?.('open');
    await vi.advanceTimersByTimeAsync(0);
    handlers.connection?.('reconnecting');
    await vi.advanceTimersByTimeAsync(STREAM_GRACE_MS);
    expect(live.lost).not.toBeNull();
    metrics = deferred<RunMetrics>();
    handlers.connection?.('open');
    handlers.reconnect?.();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.lost).not.toBeNull();
    handlers.step?.(step(9));
    expect(live.lost).toBeNull();
    stop();
  });

  it('learns steps per epoch once steps flow when the start snapshot still said 0 (a real run loading its data)', async () => {
    let spe = 0;
    hub.api = { ...hub.api, run: () => Promise.resolve({ ...snapshot, status: 'running', step: 0, steps_per_epoch: spe }) } as Api;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    metrics.resolve({ steps: [], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.steps_per_epoch).toBe(0);
    spe = 12;
    handlers.step?.(step(1));
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.steps_per_epoch).toBe(12);
    expect(live.run?.step).toBe(1);
    stop();
  });

  it('takes the run length from step and status events, without re-reading the run', async () => {
    let reads = 0;
    hub.api = { ...hub.api, run: () => (reads++, Promise.resolve({ ...snapshot, status: 'running', step: 0, steps_per_epoch: 0 })) } as Api;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    metrics.resolve({ steps: [], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    const before = reads;
    handlers.status?.({ status: 'running', error: null, steps_per_epoch: 12, epochs: 3 });
    expect(live.run?.steps_per_epoch).toBe(12);
    expect(live.run?.epochs).toBe(3);
    handlers.step?.({ ...step(1), steps_per_epoch: 12, epochs: 3 });
    await vi.advanceTimersByTimeAsync(0);
    expect(reads).toBe(before);
    // A status that still says 0 (the trainer not built yet) never takes the figure back.
    handlers.status?.({ status: 'running', error: null, steps_per_epoch: 0, epochs: 3 });
    expect(live.run?.steps_per_epoch).toBe(12);
    stop();
  });

  it('keeps a learned steps per epoch when an older snapshot (0) is applied after it', async () => {
    let spe = 0;
    hub.api = { ...hub.api, run: () => Promise.resolve({ ...snapshot, status: 'running', step: 0, steps_per_epoch: spe }) } as Api;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    metrics.resolve({ steps: [], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    // A resync takes its snapshot (still 0) and waits for /metrics; meanwhile steps flow and the figure is learned.
    metrics = deferred<RunMetrics>();
    handlers.reconnect?.();
    await vi.advanceTimersByTimeAsync(0);
    spe = 12;
    handlers.step?.(step(1));
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.steps_per_epoch).toBe(12);
    metrics.resolve({ steps: [], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.steps_per_epoch).toBe(12);
    stop();
  });

  it('keeps events that arrive while history loads, and newer status wins', async () => {
    let live!: LiveRun;
    const stop = root(() => {
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
    const stop = root(() => {
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
    const stop = root(() => {
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

  it("keeps a failed first history as an error beside the run (a finished run's snapshot still comes), until a sync gets it", async () => {
    (hub.api as { runMetrics: Api['runMetrics'] }).runMetrics = () => Promise.reject(new Error('500'));
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    // A finished run: the stream's terminal status fetches the snapshot even though the history failed.
    handlers.status?.({ status: 'stopped', error: null });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run).toBeDefined();
    expect(live.error).toBeInstanceOf(Error);
    // The page's panels say so; a later sync that gets through clears it.
    (hub.api as { runMetrics: Api['runMetrics'] }).runMetrics = () => Promise.resolve({ steps: [step(1)], epochs: [] });
    handlers.reconnect?.();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.error).toBeUndefined();
    expect(live.steps).toHaveLength(1);
    stop();
  });

  it('refreshes the snapshot on a terminal status even while history is loading', async () => {
    let calls = 0;
    (hub.api as { run: Api['run'] }).run = () => {
      calls++;
      return Promise.resolve({ ...snapshot, status: calls > 1 ? 'completed' : 'running', model_id: calls > 1 ? 'm_new' : null });
    };
    let live!: LiveRun;
    const stop = root(() => {
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
  /** When set, the next history request waits for it (to deliver an old response late). */
  let gate: Promise<RunMetrics> | null = null;
  let serverStep = 4;
  /** When set, the next snapshot request waits for it. */
  let snapGate: Promise<Run> | null = null;
  /** When set, snapshots never answer: the test sees only what events and the resume response did. */
  let hold = false;
  const saved = hub.api;
  const at = (n: number, epoch: number, total = 3): Step => ({ ...step(n), epoch, loss: { box: 1, cls: 1, dfl: 1, total } });
  const ep = (epoch: number) => ({ epoch, train_loss: { box: 1, cls: 1, dfl: 1, total: 3 }, val: { map50: 0.5, map50_95: 0.3, precision: 0.5, recall: 0.5, per_class: [] }, lr: 0.01, seconds: 1 });

  beforeEach(async () => {
    vi.useFakeTimers();
    serverStatus = 'stopped';
    hold = false;
    gate = null;
    snapGate = null;
    serverStep = 4;
    snapshot = await new MockApi('normal', 0).run('run_7d1e');
    // Epochs are 1-based (folo-api). Stopped in epoch 2: epoch 1 finished (steps 1-2), epoch 2 had steps 3-4. The
    // server keeps those interrupted steps in /metrics until the resumed run writes their replacements.
    history = { steps: [at(1, 1), at(2, 1), at(3, 2), at(4, 2)], epochs: [ep(1)] as never };
    const fake: Partial<Api> = {
      mock: false,
      run: () => {
        if (hold) return new Promise<Run>(() => {});
        const g = snapGate;
        snapGate = null;
        return g ?? Promise.resolve({ ...snapshot, status: serverStatus, step: serverStep, epoch: 2, best: { epoch: 1, map50_95: 0.3 } });
      },
      runMetrics: () => {
        const g = gate;
        gate = null;
        return g ?? Promise.resolve(history);
      },
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
    const stop = root(() => {
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
    const stop = root(() => {
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
    const stop = root(() => {
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

  it('drops history requested before the resume, even when it lands after resumed steps', async () => {
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    // A reconnect asks for history; the answer (the interrupted attempt, steps 1-4) is slow.
    const old = deferred<RunMetrics>();
    gate = old.promise;
    handlers.reconnect?.();
    serverStatus = 'running';
    await live.resume(() => Promise.resolve({ ...snapshot, status: 'queued', step: 4, epoch: 2 }));
    handlers.step?.(at(3, 2, 70));
    handlers.step?.(at(4, 2, 60));
    handlers.step?.(at(5, 3, 50));
    // Now the old answer lands: it must not bring back the interrupted 3-4.
    history = { steps: [at(1, 1), at(2, 1), at(3, 2, 70), at(4, 2, 60), at(5, 3, 50)], epochs: [ep(1), ep(2)] as never };
    old.resolve({ steps: [at(1, 1), at(2, 1), at(3, 2), at(4, 2)], epochs: [ep(1)] as never });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => [s.step, s.loss.total])).toEqual([[1, 3], [2, 3], [3, 70], [4, 60], [5, 50]]);
    stop();
  });

  it('recovers a resumed attempt that stopped inside the repeated range while disconnected', async () => {
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    serverStatus = 'queued';
    await live.resume(() => Promise.resolve({ ...snapshot, status: 'queued', step: 4, epoch: 2 }));
    // Disconnected: the resumed run did step 3 again and was stopped. The server's history replaced 3-4 with 3.
    history = { steps: [at(1, 1), at(2, 1), at(3, 2, 80)], epochs: [ep(1)] as never };
    serverStatus = 'stopped';
    serverStep = 3;
    handlers.reconnect?.();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => [s.step, s.loss.total])).toEqual([[1, 3], [2, 3], [3, 80]]);
    expect(live.run?.step).toBe(3);
    expect(live.run?.status).toBe('stopped');
    stop();
  });

  it('keeps the steps the new attempt streamed over history that still holds the interrupted attempt', async () => {
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    serverStatus = 'running';
    // History requested after the resume is slow, and the server has not written replacements yet.
    const slow = deferred<RunMetrics>();
    gate = slow.promise;
    await live.resume(() => Promise.resolve({ ...snapshot, status: 'queued', step: 4, epoch: 2 }));
    handlers.reconnect?.();
    handlers.step?.(at(3, 2, 70));
    slow.resolve({ steps: [at(1, 1), at(2, 1), at(3, 2), at(4, 2)], epochs: [ep(1)] as never });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => [s.step, s.loss.total])).toEqual([[1, 3], [2, 3], [3, 70]]);
    stop();
  });

  it('never removes data for a late or duplicate step in an ordinary run', async () => {
    let live!: LiveRun;
    serverStatus = 'running';
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3, 4]);
    handlers.step?.(at(3, 2));
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3, 4]);
    expect(live.epochs.map((e) => e.epoch)).toEqual([1]);
    stop();
  });

  it('drops a cached interrupted suffix when a resume happened elsewhere while disconnected', async () => {
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    // Another client resumed it; the resumed run repeated step 3 and was stopped again. We saw none of it.
    history = { steps: [at(1, 1), at(2, 1), at(3, 2, 80)], epochs: [ep(1)] as never };
    serverStep = 3;
    handlers.reconnect?.();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => [s.step, s.loss.total])).toEqual([[1, 3], [2, 3], [3, 80]]);
    expect(live.run?.step).toBe(3);
    stop();
  });

  it('ignores a terminal snapshot requested before the resume response rewound the view', async () => {
    let live!: LiveRun;
    serverStatus = 'running';
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    // The run stops; its terminal snapshot is slow.
    const late = deferred<Run>();
    snapGate = late.promise;
    handlers.status?.({ status: 'stopped', error: null });
    await live.resume(() => Promise.resolve({ ...snapshot, status: 'queued', step: 4, epoch: 2 }));
    expect(live.run?.status).toBe('queued');
    late.resolve({ ...snapshot, status: 'stopped', step: 4, epoch: 2 });
    await vi.advanceTimersByTimeAsync(0);
    // The late 'stopped' snapshot is ignored (progress follows the server's history, which is fetched again).
    expect(live.run?.status).not.toBe('stopped');
    stop();
  });

  it('drops a history response from the replay that lands after the final reconcile', async () => {
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    serverStatus = 'running';
    await live.resume(() => Promise.resolve({ ...snapshot, status: 'queued', step: 4, epoch: 2 }));
    // Request A (during the replay) still sees the interrupted attempt and is slow.
    const a = deferred<RunMetrics>();
    gate = a.promise;
    handlers.reconnect?.();
    // The resumed run does step 3 again and stops; the final reconcile (request B) has the true history.
    handlers.step?.(at(3, 2, 70));
    history = { steps: [at(1, 1), at(2, 1), at(3, 2, 70)], epochs: [ep(1)] as never };
    serverStatus = 'stopped';
    serverStep = 3;
    handlers.status?.({ status: 'stopped', error: null });
    await vi.advanceTimersByTimeAsync(0);
    a.resolve({ steps: [at(1, 1), at(2, 1), at(3, 2), at(4, 2)], epochs: [ep(1)] as never });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => [s.step, s.loss.total])).toEqual([[1, 3], [2, 3], [3, 70]]);
    stop();
  });

  it('drops an older history response in an ordinary finished run', async () => {
    let live!: LiveRun;
    serverStatus = 'running';
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    const a = deferred<RunMetrics>();
    gate = a.promise;
    handlers.reconnect?.();
    handlers.step?.(at(5, 3));
    history = { steps: [at(1, 1), at(2, 1), at(3, 2), at(4, 2), at(5, 3)], epochs: [ep(1), ep(2)] as never };
    serverStatus = 'completed';
    handlers.reconnect?.();
    await vi.advanceTimersByTimeAsync(0);
    a.resolve({ steps: [at(1, 1), at(2, 1), at(3, 2), at(4, 2)], epochs: [ep(1)] as never });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3, 4, 5]);
    stop();
  });

  it('keeps the final steps when a partial history lands after the run completed', async () => {
    let live!: LiveRun;
    serverStatus = 'running';
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    // Request A (while running) captures a partial history and is slow.
    const a = deferred<RunMetrics>();
    gate = a.promise;
    handlers.reconnect?.();
    // The run finishes: last step, last epoch, completed; the server's final history has them.
    handlers.step?.(at(5, 3));
    history = { steps: [at(1, 1), at(2, 1), at(3, 2), at(4, 2), at(5, 3)], epochs: [ep(1), ep(2), ep(3)] as never };
    serverStatus = 'completed';
    handlers.epoch?.(ep(3) as never);
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    a.resolve({ steps: [at(1, 1), at(2, 1)], epochs: [] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3, 4, 5]);
    expect(live.epochs.map((e) => e.epoch)).toEqual([1, 2, 3]);
    expect(live.run?.status).toBe('completed');
    stop();
  });

  it('fetches the final history when the run ends while the first history loads are still pending', async () => {
    serverStatus = 'running';
    // Both the initial load and the load on stream open capture a partial history and are slow.
    const first = deferred<RunMetrics>();
    gate = first.promise;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    const second = deferred<RunMetrics>();
    gate = second.promise;
    handlers.connection?.('open');
    // The stream is rate-limited: it delivers step 5 but never 3-4, then the run completes.
    handlers.step?.(at(5, 3));
    history = { steps: [at(1, 1), at(2, 1), at(3, 2), at(4, 2), at(5, 3)], epochs: [ep(1), ep(2), ep(3)] as never };
    serverStatus = 'completed';
    handlers.status?.({ status: 'completed', error: null });
    const partial = { steps: [at(1, 1), at(2, 1)], epochs: [ep(1)] as never };
    first.resolve(partial);
    second.resolve(partial);
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3, 4, 5]);
    stop();
  });

  it('fetches the final history even when a terminal snapshot was applied before the terminal event', async () => {
    serverStatus = 'running';
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    // A reload whose /metrics was read before the end but whose /run already says completed.
    serverStatus = 'completed';
    history = { steps: [at(1, 1), at(2, 1)], epochs: [ep(1)] as never };
    handlers.reconnect?.();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('completed');
    // The terminal event arrives afterwards; by now the server's final history is complete.
    history = { steps: [at(1, 1), at(2, 1), at(3, 2), at(4, 2), at(5, 3)], epochs: [ep(1), ep(2), ep(3)] as never };
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3, 4, 5]);
    stop();
  });
});

describe('LiveRun: the training queue (docs/API.md, SSE queue)', () => {
  let handlers: RunEventHandlers = {};
  const saved = hub.api;
  afterEach(() => {
    hub.api = saved;
    vi.useRealTimers();
  });

  it('keeps the latest queue event as the run’s place, drops it once the run starts, then reads `resolved`', async () => {
    vi.useFakeTimers();
    const queued = await new MockApi('normal', 0).run('run_q7c5');
    const started = { ...queued, status: 'running' as const, queue_position: null, resolved: { optimizer: 'adamw', lr0: 0.000833 } as Run['resolved'] };
    let snap: Run = queued;
    hub.api = {
      mock: false,
      run: () => Promise.resolve(snap),
      runMetrics: () => Promise.resolve({ steps: [], epochs: [] }),
      runLogs: () => Promise.resolve({ lines: [] }),
      runEvents: (_id: string, h: RunEventHandlers) => {
        handlers = h;
        return () => {};
      },
    } as unknown as Api;
    hub.mode = 'live';
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_q7c5');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.queue_position).toBe(2);
    handlers.queue?.({ position: 1, waiting: 2 });
    expect(live.run?.queue_position).toBe(1);
    expect(live.waiting).toBe(2);
    snap = started;
    handlers.status?.({ status: 'running', error: null });
    expect(live.run?.queue_position).toBeNull();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.resolved?.lr0).toBe(0.000833);
    stop();
  });
  /** A live run whose snapshots resolve only when the test says so (each `run()` call queues one). */
  async function waitingRun() {
    vi.useFakeTimers();
    const queued = await new MockApi('normal', 0).run('run_q7c5');
    const snaps: ReturnType<typeof deferred<Run>>[] = [];
    hub.api = {
      mock: false,
      run: () => {
        const d = deferred<Run>();
        snaps.push(d);
        return d.promise;
      },
      runMetrics: () => Promise.resolve({ steps: [], epochs: [] }),
      runLogs: () => Promise.resolve({ lines: [] }),
      runEvents: (_id: string, h: RunEventHandlers) => {
        handlers = h;
        return () => {};
      },
    } as unknown as Api;
    hub.mode = 'live';
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_q7c5');
    });
    flushSync();
    return { queued, snaps, live, stop };
  }

  it('keeps a queue event that arrives before the first snapshot, over the older place it holds', async () => {
    const { queued, snaps, live, stop } = await waitingRun();
    expect(queued.queue_position).toBe(2);
    // The stream reports the new place while no run is loaded yet; the snapshot was taken before it.
    handlers.queue?.({ position: 1, waiting: 2 });
    expect(live.run).toBeUndefined();
    snaps[0]!.resolve(queued);
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('queued');
    expect(live.run?.queue_position).toBe(1);
    expect(live.waiting).toBe(2);
    stop();
  });

  it('keeps a queue event that arrives while a reconnect reloads, over the place its snapshot captured', async () => {
    const { queued, snaps, live, stop } = await waitingRun();
    snaps[0]!.resolve(queued);
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.queue_position).toBe(2);
    // The reconnect's snapshot is read at place 2; the run moves up before it lands.
    handlers.reconnect?.();
    expect(snaps).toHaveLength(2);
    handlers.queue?.({ position: 1, waiting: 2 });
    expect(live.run?.queue_position).toBe(1);
    snaps[1]!.resolve(queued);
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.queue_position).toBe(1);
    stop();
  });

  it('drops the place a snapshot holds once a newer status says the run started', async () => {
    const { queued, snaps, live, stop } = await waitingRun();
    handlers.status?.({ status: 'running', error: null });
    snaps[0]!.resolve(queued);
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('running');
    expect(live.run?.queue_position).toBeNull();
    stop();
  });
});

describe('LiveRun: a cloud run (docs/API.md, SSE cloud)', () => {
  const saved = hub.api;
  afterEach(() => {
    hub.api = saved;
    vi.useRealTimers();
  });

  it('keeps the latest cloud event over a snapshot requested before it', async () => {
    vi.useFakeTimers();
    const run = await new MockApi('cloud-upload', 0).run('run_c9e4');
    let handlers: RunEventHandlers = {};
    const snaps: ReturnType<typeof deferred<Run>>[] = [];
    hub.api = {
      mock: false,
      run: () => {
        const d = deferred<Run>();
        snaps.push(d);
        return d.promise;
      },
      runMetrics: () => Promise.resolve({ steps: [], epochs: [] }),
      runLogs: () => Promise.resolve({ lines: [] }),
      runEvents: (_id: string, h: RunEventHandlers) => {
        handlers = h;
        return () => {};
      },
    } as unknown as Api;
    hub.mode = 'live';
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_c9e4');
    });
    flushSync();
    // The upload ends while the first snapshot (still uploading) is on its way.
    const next = { ...run.cloud!, phase: 'training', progress: null };
    handlers.cloud?.(next);
    snaps[0]!.resolve(run);
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.cloud?.phase).toBe('training');
    // State, not a change: a later event replaces it whole.
    handlers.cloud?.({ ...next, phase: 'collecting', progress: { done_bytes: 1, total_bytes: 2 } });
    expect(live.run?.cloud).toMatchObject({ phase: 'collecting', progress: { done_bytes: 1, total_bytes: 2 } });
    stop();
  });
});

describe('LiveRun: a long log', () => {
  const saved = hub.api;
  afterEach(() => {
    hub.api = saved;
    vi.useRealTimers();
  });

  it('asks for as many lines as it keeps, and counts the older ones the server has (the panel says them)', async () => {
    vi.useFakeTimers();
    const done = { ...(await new MockApi('normal', 0).run('run_q7c5')), status: 'completed' as const };
    const limits: (number | undefined)[] = [];
    const lines = Array.from({ length: 2000 }, (_, i) => ({ t: `2026-11-12T10:${String(Math.floor(i / 60)).padStart(2, '0')}:${String(i % 60).padStart(2, '0')}Z`, level: 'info' as const, message: `epoch ${i + 705}` }));
    hub.api = {
      mock: false,
      run: () => Promise.resolve(done),
      runMetrics: () => Promise.resolve({ steps: [], epochs: [] }),
      runLogs: (_id: string, limit?: number) => (limits.push(limit), Promise.resolve({ lines, total: 2704 })),
      runEvents: () => () => {},
    } as unknown as Api;
    hub.mode = 'live';
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_q7c5');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(limits).toEqual([2000]);
    expect(live.logs).toHaveLength(2000);
    expect(live.logsEarlier).toBe(704);
    stop();
  });

  it('adds the lines streamed after the server counted them, and keeps its count when a later load has none', async () => {
    vi.useFakeTimers();
    const running = { ...(await new MockApi('normal', 0).run('run_q7c5')), status: 'running' as const };
    const at = (i: number) => `2026-11-12T10:${String(Math.floor(i / 60)).padStart(2, '0')}:${String(i % 60).padStart(2, '0')}Z`;
    const lines = Array.from({ length: 2000 }, (_, i) => ({ t: at(i), level: 'info' as const, message: `epoch ${i + 705}` }));
    let history = deferred<RunLogs>();
    let handlers: RunEventHandlers = {};
    hub.api = {
      mock: false,
      run: () => Promise.resolve(running),
      runMetrics: () => Promise.resolve({ steps: [], epochs: [] }),
      runLogs: () => history.promise,
      runEvents: (_id: string, h: RunEventHandlers) => {
        handlers = h;
        return () => {};
      },
    } as unknown as Api;
    hub.mode = 'live';
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_q7c5');
    });
    flushSync();
    // Ten lines stream in after the server answered with 2,704, before its answer is applied; a cloud worker's keep
    // their own (earlier) times.
    for (let i = 0; i < 10; i++) handlers.log?.({ t: at(i % 2 ? 2000 + i : 100 + i), level: 'info', message: `late ${i}` });
    history.resolve({ lines, total: 2704 });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.logs).toHaveLength(2000);
    expect(live.logsEarlier).toBe(714);
    // Two loads asked before those lines came, answered one after the other: the second sees them already merged
    // (and sorted among the history by their times) and counts them once.
    const first = deferred<RunLogs>();
    const second = deferred<RunLogs>();
    history = first;
    handlers.reconnect?.();
    history = second;
    handlers.reconnect?.();
    // The server, asked after the first ten, counts and returns them; then ten more stream in before it answers.
    const now = [...live.logs];
    for (let i = 0; i < 10; i++) handlers.log?.({ t: at(200 + i), level: 'info', message: `cloud ${i}` });
    first.resolve({ lines: now, total: 2714 });
    await vi.advanceTimersByTimeAsync(0);
    second.resolve({ lines: now, total: 2714 });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.logsEarlier).toBe(724);
    // A reload of the history from a server that does not count (or a failed one) leaves the count as it was.
    history = deferred();
    handlers.reconnect?.();
    history.resolve({ lines: lines.slice(-500) });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.logsEarlier).toBe(724);
    stop();
  });
});
