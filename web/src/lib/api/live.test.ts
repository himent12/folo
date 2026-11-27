// @vitest-environment happy-dom
import { createRoot, effect, flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, type Api, type RunEventHandlers } from './client';
import { hub } from './index';
import { keepStepsPerEpoch, LiveRun, mergeBy, STREAM_GRACE_MS, streamedRun, streamedStatus } from './live';
import { MockApi } from './mock';
import type { Run, RunMetrics, Step } from './types';

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

  it('exposes streamedStatus without notifying subscribers on step ticks, only on status or liveness changes', async () => {
    let live!: LiveRun;
    const seen: (string | undefined)[] = [];
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
      effect(() => {
        seen.push(streamedStatus('run_7d1e'));
      });
    });
    flushSync();
    expect(seen).toEqual([undefined]);
    metrics.resolve({ steps: [], epochs: [] });
    handlers.connection?.('open');
    await vi.advanceTimersByTimeAsync(0);
    flushSync();
    expect(seen).toEqual([undefined, 'running']);
    // Step ticks update `live.run` at frame cadence but must not re-run `streamedStatus` subscribers.
    handlers.step?.(step(3));
    await vi.advanceTimersByTimeAsync(20);
    handlers.step?.(step(4));
    await vi.advanceTimersByTimeAsync(20);
    flushSync();
    expect(seen).toEqual([undefined, 'running']);
    // A status change notifies immediately.
    handlers.status?.({ status: 'completed', error: null });
    flushSync();
    expect(seen).toEqual([undefined, 'running', 'completed']);
    stop();
    expect(streamedStatus('run_7d1e')).toBeUndefined();
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

  it('skips runLogs requests and log buffering when created with { logs: false } (#309)', async () => {
    const runLogs = vi.fn(() => Promise.resolve({ lines: [{ t: '2026-11-08T14:00:00Z', level: 'info' as const, message: 'hi' }] }));
    hub.api = { ...hub.api, runLogs } as Api;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e', { logs: false });
    });
    flushSync();
    metrics.resolve({ steps: [], epochs: [] });
    handlers.connection?.('open');
    await vi.advanceTimersByTimeAsync(0);
    handlers.log?.({ t: '2026-11-08T14:00:01Z', level: 'info', message: 'streamed' });
    await vi.advanceTimersByTimeAsync(20);
    expect(runLogs).not.toHaveBeenCalled();
    expect(live.logs).toEqual([]);
    expect(live.logsEarlier).toBe(0);
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

  it('does not re-fetch an already-completed run on stream open or initial terminal status (#309)', async () => {
    const doneSnap: Run = { ...snapshot, status: 'completed', epoch: 2, epochs: 2, step: 4, best: { epoch: 2, map50_95: 0.3 } };
    const ep = (epoch: number) => ({ epoch, train_loss: { box: 1, cls: 1, dfl: 1, total: 3 }, val: { map50: 0.5, map50_95: 0.3, precision: 0.5, recall: 0.5, per_class: [] }, lr: 0.01, seconds: 1 });
    const runSpy = vi.fn(async () => doneSnap);
    const metricsSpy = vi.fn(async () => ({ steps: [step(1), step(2)], epochs: [ep(1), ep(2)] as never }));
    const logsSpy = vi.fn(async () => ({ lines: [] }));
    hub.api = { ...hub.api, run: runSpy, runMetrics: metricsSpy, runLogs: logsSpy } as Api;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    // Initial load settled the completed run; SSE opening and sending its initial `completed` status must not refetch.
    handlers.connection?.('open');
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('completed');
    expect(runSpy).toHaveBeenCalledTimes(1);
    expect(metricsSpy).toHaveBeenCalledTimes(1);
    expect(logsSpy).toHaveBeenCalledTimes(1);
    stop();
  });

  it('does not cascade stale-generation syncs when a completed run opens while initial load is in flight (#309)', async () => {
    const doneSnap: Run = { ...snapshot, status: 'completed', epoch: 2, epochs: 2, step: 4, best: { epoch: 2, map50_95: 0.3 } };
    const ep = (epoch: number) => ({ epoch, train_loss: { box: 1, cls: 1, dfl: 1, total: 3 }, val: { map50: 0.5, map50_95: 0.3, precision: 0.5, recall: 0.5, per_class: [] }, lr: 0.01, seconds: 1 });
    const first = deferred<RunMetrics>();
    const second = deferred<RunMetrics>();
    let calls = 0;
    const runSpy = vi.fn(async () => doneSnap);
    const metricsSpy = vi.fn(() => {
      calls++;
      if (calls === 1) return first.promise;
      if (calls === 2) return second.promise;
      return Promise.resolve({ steps: [step(1), step(2)], epochs: [ep(1), ep(2)] as never });
    });
    const logsSpy = vi.fn(async () => ({ lines: [] }));
    hub.api = { ...hub.api, run: runSpy, runMetrics: metricsSpy, runLogs: logsSpy } as Api;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    // Stream opens and immediately sends its initial `completed` status before the first load resolves.
    handlers.connection?.('open');
    handlers.status?.({ status: 'completed', error: null });
    first.resolve({ steps: [step(1), step(2)], epochs: [ep(1), ep(2)] as never });
    second.resolve({ steps: [step(1), step(2)], epochs: [ep(1), ep(2)] as never });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('completed');
    expect(runSpy).toHaveBeenCalledTimes(2);
    expect(metricsSpy).toHaveBeenCalledTimes(2);
    expect(logsSpy).toHaveBeenCalledTimes(2);
    stop();
  });

  it('shares the terminal snapshot request with sync when an active run finishes (#309)', async () => {
    const ep = (epoch: number) => ({ epoch, train_loss: { box: 1, cls: 1, dfl: 1, total: 3 }, val: { map50: 0.5, map50_95: 0.3, precision: 0.5, recall: 0.5, per_class: [] }, lr: 0.01, seconds: 1 });
    let ended = false;
    const runSpy = vi.fn(async () => ({
      ...snapshot,
      status: ended ? ('completed' as const) : ('running' as const),
      epoch: 2,
      epochs: 2,
      best: { epoch: 2, map50_95: 0.3 },
      model_id: ended ? 'm_final' : null,
    }));
    const metricsSpy = vi.fn(async () => ({ steps: ended ? [step(1), step(2), step(3)] : [step(1), step(2)], epochs: [ep(1), ep(2)] as never }));
    const logsSpy = vi.fn(async () => ({ lines: [] }));
    hub.api = { ...hub.api, run: runSpy, runMetrics: metricsSpy, runLogs: logsSpy } as Api;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    handlers.connection?.('open');
    await vi.advanceTimersByTimeAsync(0);
    expect(runSpy).toHaveBeenCalledTimes(2);
    handlers.step?.(step(3));
    ended = true;
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('completed');
    expect(live.run?.model_id).toBe('m_final');
    expect(live.steps.map((s) => s.step)).toEqual([1, 2, 3]);
    // Terminal event issues a single sync load (1 /run, 1 /metrics, 1 /logs) — no duplicate snapshot() call.
    expect(runSpy).toHaveBeenCalledTimes(3);
    expect(metricsSpy).toHaveBeenCalledTimes(3);
    expect(logsSpy).toHaveBeenCalledTimes(3);
    stop();
  });

  it('still resyncs when a stopped run completes before SSE delivers status (#309 review)', async () => {
    const ep = (epoch: number) => ({ epoch, train_loss: { box: 1, cls: 1, dfl: 1, total: 3 }, val: { map50: 0.5, map50_95: 0.3, precision: 0.5, recall: 0.5, per_class: [] }, lr: 0.01, seconds: 1 });
    let completed = false;
    const runSpy = vi.fn(async () => ({
      ...snapshot,
      status: completed ? ('completed' as const) : ('stopped' as const),
      epoch: completed ? 2 : 1,
      epochs: 2,
      best: { epoch: completed ? 2 : 1, map50_95: 0.3 },
      model_id: completed ? 'm_resumed' : null,
    }));
    const metricsSpy = vi.fn(async () => ({ steps: completed ? [step(1), step(2)] : [step(1)], epochs: completed ? ([ep(1), ep(2)] as never) : ([ep(1)] as never) }));
    const logsSpy = vi.fn(async () => ({ lines: [] }));
    hub.api = { ...hub.api, run: runSpy, runMetrics: metricsSpy, runLogs: logsSpy } as Api;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('stopped');
    expect(live.run?.model_id).toBeNull();
    // Run resumed and finished on the server before SSE delivered its initial status.
    completed = true;
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('completed');
    expect(live.run?.model_id).toBe('m_resumed');
    expect(live.epochs.map((e) => e.epoch)).toEqual([1, 2]);
    stop();
  });

  it('still applies a completed run snapshot when /metrics rejects before terminal status arrives (#309 review)', async () => {
    const doneSnap: Run = { ...snapshot, status: 'completed', epoch: 2, epochs: 2, best: { epoch: 2, map50_95: 0.3 }, model_id: 'm_done' };
    hub.api = {
      ...hub.api,
      run: vi.fn(async () => doneSnap),
      runMetrics: vi.fn(async () => {
        throw new Error('metrics unavailable');
      }),
      runLogs: vi.fn(async () => ({ lines: [] })),
    } as Api;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    handlers.connection?.('open');
    // Both initial and post-open loads reject on /metrics before the terminal status event arrives.
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run).toBeUndefined();
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('completed');
    expect(live.run?.model_id).toBe('m_done');
    stop();
  });

  it('falls back to a full terminal sync when an in-flight post-open snapshot resolves as running (#309 review)', async () => {
    const ep = (epoch: number) => ({ epoch, train_loss: { box: 1, cls: 1, dfl: 1, total: 3 }, val: { map50: 0.5, map50_95: 0.3, precision: 0.5, recall: 0.5, per_class: [] }, lr: 0.01, seconds: 1 });
    const firstSnap = deferred<Run>();
    const openSnap = deferred<Run>();
    const firstMetrics = deferred<RunMetrics>();
    const openMetrics = deferred<RunMetrics>();
    let runCalls = 0;
    let metricsCalls = 0;
    const runningSnap: Run = { ...snapshot, status: 'running', epoch: 1, epochs: 2, best: { epoch: 1, map50_95: 0.2 }, model_id: null };
    const finalSnap: Run = { ...snapshot, status: 'completed', epoch: 2, epochs: 2, best: { epoch: 2, map50_95: 0.4 }, model_id: 'm_final' };
    const runSpy = vi.fn(() => {
      runCalls++;
      if (runCalls === 1) return firstSnap.promise;
      if (runCalls === 2) return openSnap.promise;
      return Promise.resolve(finalSnap);
    });
    const metricsSpy = vi.fn(() => {
      metricsCalls++;
      if (metricsCalls === 1) return firstMetrics.promise;
      if (metricsCalls === 2) return openMetrics.promise;
      return Promise.resolve({ steps: [step(1), step(2)], epochs: [ep(1), ep(2)] as never });
    });
    hub.api = { ...hub.api, run: runSpy, runMetrics: metricsSpy, runLogs: vi.fn(async () => ({ lines: [] })) } as Api;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    handlers.connection?.('open');
    // Terminal status arrives while the post-open load is still in flight; its snapshot predated completion.
    handlers.status?.({ status: 'completed', error: null });
    firstSnap.resolve(runningSnap);
    openSnap.resolve(runningSnap);
    firstMetrics.resolve({ steps: [step(1)], epochs: [ep(1)] as never });
    openMetrics.resolve({ steps: [step(1)], epochs: [ep(1)] as never });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('completed');
    expect(live.run?.model_id).toBe('m_final');
    expect(live.epochs.map((e) => e.epoch)).toEqual([1, 2]);
    stop();
  });

  it('closes the SSE stream immediately when the initial load returns 404', async () => {
    const closes = vi.fn();
    const gone = new ApiError(404, 'run_not_found', 'No run with id run_missing.');
    const runSpy = vi.fn(() => Promise.reject(gone));
    hub.api = {
      ...hub.api,
      run: runSpy,
      runMetrics: vi.fn(() => Promise.reject(gone)),
      runLogs: vi.fn(() => Promise.reject(gone)),
      runEvents: (_id, h) => {
        handlers = h;
        return closes;
      },
    } as Api;
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_missing');
    });
    try {
      flushSync();
      await vi.advanceTimersByTimeAsync(0);
      expect(live.error).toBe(gone);
      expect(live.loading).toBe(false);
      expect(live.connection).toBe('idle');
      expect(live.lost).toBeNull();
      expect(closes).toHaveBeenCalledTimes(1);
      // Subsequent stream errors or timers must not trigger another fetch.
      const calls = runSpy.mock.calls.length;
      handlers.connection?.('reconnecting');
      await vi.advanceTimersByTimeAsync(60_000);
      expect(runSpy.mock.calls.length).toBe(calls);
    } finally {
      stop();
    }
  });
});
