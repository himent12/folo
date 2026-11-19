// @vitest-environment happy-dom
// A run's end while a view follows it (issue #327): the epochs it validated reach the view whatever order the end, the
// snapshot, the history and the log come in. Each answer is held until the test releases it.
import { createRoot, flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Api, RunEventHandlers } from './client';
import { hub } from './index';
import { LiveRun } from './live';
import { MockApi } from './mock';
import { epochsValidated } from './progress';
import { RunModel } from '../../routes/run/runModel';
import type { Epoch, Run, RunLogs, RunMetrics, Step } from './types';

const root = (fn: () => void) => createRoot((dispose) => (fn(), dispose));

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => ((resolve = res), (reject = rej)));
  return { promise, resolve, reject };
}

const steps: Step[] = [1, 2, 3, 4].map((n) => ({ epoch: 1, step: n, lr: 0.01, loss: { box: 1, cls: 1, dfl: 1, total: 3 }, imgs_per_s: 100, eta_s: 4 - n }));
/** The run page's readings over the view, as Run.tsx builds them: the "final" wording and the short-history note. */
const readings = (live: LiveRun) =>
  new RunModel({
    get run() {
      return live.run;
    },
    steps: live.stepsLog,
    epochs: live.epochsLog,
    get version() {
      return live.version;
    },
    get server() {
      return live.server;
    },
    get shortAnswers() {
      return live.shortAnswers;
    },
  });

const ep: Epoch = { epoch: 1, train_loss: { box: 1, cls: 1, dfl: 1, total: 3 }, val: { map50: 0.5, map50_95: 0.3, precision: 0.5, recall: 0.5, per_class: [] }, lr: 0.01, seconds: 1 } as never;

describe('LiveRun at a run’s end (issue #327)', () => {
  let handlers: RunEventHandlers = {};
  let running: Run;
  let completed: Run;
  /** What each request answers, set per test; `calls` counts the history requests. */
  let run: () => Promise<Run>;
  let metrics: () => Promise<RunMetrics>;
  let logs: () => Promise<RunLogs>;
  let calls = 0;
  /** The abort signal of every request the view made. */
  let signals: (AbortSignal | undefined)[] = [];
  let live!: LiveRun;
  let stop = () => {};
  const saved = hub.api;

  beforeEach(async () => {
    vi.useFakeTimers();
    const base = await new MockApi('normal', 0).run('run_7d1e');
    running = { ...base, status: 'running', epochs: 1, epoch: 1, step: 4, steps_per_epoch: 4, best: null };
    completed = { ...running, status: 'completed', best: { epoch: 1, map50_95: 0.3 }, eta_s: null };
    run = () => Promise.resolve(running);
    metrics = () => Promise.resolve({ steps, epochs: [] });
    logs = () => Promise.resolve({ lines: [] });
    calls = 0;
    signals = [];
    const fake: Partial<Api> = {
      mock: false,
      run: (_id, s) => (signals.push(s), run()),
      runMetrics: (_id, s) => (calls++, signals.push(s), metrics()),
      runLogs: (_id, _n, s) => (signals.push(s), logs()),
      runEvents: (_id, h) => {
        handlers = h;
        return () => {};
      },
    };
    hub.api = fake as Api;
    hub.mode = 'live';
    // Open while it trains its last steps, before its validation.
    stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    handlers.connection?.('open');
    await vi.advanceTimersByTimeAsync(0);
    expect([live.run?.status, live.steps.length, live.epochs.length]).toEqual(['running', 4, 0]);
  });
  afterEach(() => {
    stop();
    hub.api = saved;
    vi.useRealTimers();
  });

  it('asks again when a reconnect read the history before the validation and the snapshot after the end', async () => {
    // A reconnect at the end: /metrics answers from before epoch 1 validated, /run from after the run completed, and
    // the terminal event is late. That history is not the final one.
    const before = deferred<RunMetrics>();
    const after = deferred<Run>();
    metrics = () => before.promise;
    run = () => after.promise;
    handlers.reconnect?.();
    metrics = () => Promise.resolve({ steps, epochs: [ep] });
    run = () => Promise.resolve(completed);
    before.resolve({ steps, epochs: [] });
    after.resolve(completed);
    await vi.advanceTimersByTimeAsync(0);
    expect([live.run?.status, live.epochs.length]).toEqual(['completed', 0]);
    expect(epochsValidated(live.server!)).toBe(1); // the page holds the panels as loading: one is on its way
    await vi.advanceTimersByTimeAsync(1000);
    expect(live.epochs.map((e) => e.epoch)).toEqual([1]);
    // The late end changes nothing.
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.epochs.map((e) => e.epoch)).toEqual([1]);
  });

  it('applies the final metrics while its log request stalls', async () => {
    // The stream missed `epoch 1`; the final /logs never answers, /run and /metrics do.
    logs = () => new Promise(() => {});
    run = () => Promise.resolve(completed);
    metrics = () => Promise.resolve({ steps, epochs: [ep] });
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.epochs.map((e) => e.epoch)).toEqual([1]);
    expect(live.error).toBeUndefined();
  });

  it('lets a slow final history download for as long as it keeps arriving, and applies it once', async () => {
    // A long run's history over a slow link: its bytes arrive for a minute (30.5 MB at 160 KiB/s takes about 54 s).
    const slow = deferred<RunMetrics>();
    run = () => Promise.resolve(completed);
    metrics = () => slow.promise;
    handlers.status?.({ status: 'completed', error: null });
    const asked = calls;
    await vi.advanceTimersByTimeAsync(55_000);
    expect([live.error, calls]).toEqual([undefined, asked]);
    slow.resolve({ steps, epochs: [ep] });
    await vi.advanceTimersByTimeAsync(60_000);
    expect([live.error, calls, live.epochs.map((e) => e.epoch)]).toEqual([undefined, asked, [1]]);
  });

  it('aborts its requests and clears its timers when the view goes, with requests stalled', async () => {
    run = () => new Promise(() => {});
    metrics = () => new Promise(() => {});
    logs = () => new Promise(() => {});
    const before = signals.length;
    // A reconnect's snapshot, history and log, and the stream's check whether the run is gone: none answers.
    handlers.reconnect?.();
    handlers.connection?.('reconnecting');
    await vi.advanceTimersByTimeAsync(0);
    expect(signals.length - before).toBe(4);
    stop();
    expect(signals.every((s) => s?.aborted)).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it('ignores a resync that fails after the end started the final history request, and its retry', async () => {
    const stale = deferred<RunMetrics>();
    const final = deferred<RunMetrics>();
    metrics = () => stale.promise;
    handlers.reconnect?.();
    metrics = () => final.promise;
    run = () => Promise.resolve(completed);
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    const asked = calls;
    stale.reject(new Error('Boom.'));
    await vi.advanceTimersByTimeAsync(5000);
    // The current request is healthy: no error over the loading panels, and no retry of the obsolete one.
    expect(live.error).toBeUndefined();
    expect(calls).toBe(asked);
    final.resolve({ steps, epochs: [ep] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.epochs.map((e) => e.epoch)).toEqual([1]);
  });

  it('publishes the epoch the stream delivered with the end, not a frame later', async () => {
    const final = deferred<RunMetrics>();
    metrics = () => final.promise;
    run = () => Promise.resolve(completed);
    handlers.epoch?.(ep);
    const v = live.version;
    handlers.status?.({ status: 'completed', error: null });
    expect(live.version).toBeGreaterThan(v);
    expect(live.epochs).toHaveLength(1);
    final.resolve({ steps, epochs: [ep] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.epochs).toHaveLength(1);
  });

  it('calls nothing final after a lost stream until the late ended snapshot and the last epoch are in', async () => {
    // A two-epoch run: the stream delivers epoch 1 and drops; it comes back after the run completed, with the server's
    // snapshot and history late. The view holds epoch 1 of a completed run, and the server has validated epoch 2.
    const two: Run = { ...completed, epochs: 2, epoch: 2, best: { epoch: 2, map50_95: 0.4 } };
    const m = readings(live);
    handlers.epoch?.(ep);
    handlers.connection?.('reconnecting');
    const snap = deferred<Run>();
    const history = deferred<RunMetrics>();
    run = () => snap.promise;
    metrics = () => history.promise;
    handlers.connection?.('open');
    handlers.reconnect?.();
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    expect([live.run?.status, live.epochs.map((e) => e.epoch)]).toEqual(['completed', [1]]);
    expect([live.results?.final, m.final.get(), m.pending.get()]).toEqual([false, false, false]);
    snap.resolve(two);
    history.resolve({ steps, epochs: [ep, { ...ep, epoch: 2 }] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.epochs.map((e) => e.epoch)).toEqual([1, 2]);
    expect([live.results?.final, m.final.get()]).toEqual([true, true]);
  });

  it('takes a history that stays short of the ended snapshot as all there is after three answers, and stops asking', async () => {
    // A completed three-epoch run whose metrics file recorded only epoch 1 (a write failed during training): /metrics
    // answers 200 with epoch 1 every time.
    const three: Run = { ...completed, epochs: 3, epoch: 3, best: { epoch: 3, map50_95: 0.4 } };
    const m = readings(live);
    run = () => Promise.resolve(three);
    metrics = () => Promise.resolve({ steps, epochs: [ep] });
    const asked = calls;
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    expect([m.pending.get(), m.shortNote.get()]).toEqual([true, null]);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(calls - asked).toBe(3);
    expect(live.epochs.map((e) => e.epoch)).toEqual([1]);
    expect([m.pending.get(), m.final.get(), live.error]).toEqual([false, false, undefined]);
    expect(m.shortNote.get()).toEqual({
      title: 'Epochs 2–3 are missing from its history',
      why: "The run's record names 3 validated epochs; its history ends at epoch 1. The readings show what was recorded.",
    });
    await vi.advanceTimersByTimeAsync(120_000);
    expect(calls - asked).toBe(3);
    expect(vi.getTimerCount()).toBe(0);
  });
});
