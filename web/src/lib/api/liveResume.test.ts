// @vitest-environment happy-dom
// LiveRun tests for resume rewinding/reconciliation, training queue events, cloud phase events, and long-log accounting.
import { createRoot, flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Api, RunEventHandlers } from './client';
import { hub } from './index';
import { LiveRun } from './live';
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
