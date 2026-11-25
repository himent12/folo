// @vitest-environment happy-dom
import { createRoot, effect, flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Api, RunEventHandlers } from './client';
import { hub } from './index';
import { LiveRun, LOG_TRIM_BATCH, MAX_LOGS } from './live';
import { MockApi } from './mock';
import type { Run, Step } from './types';

const root = (fn: () => void) => createRoot((dispose) => (fn(), dispose));
const step = (n: number): Step => ({ epoch: 0, step: n, lr: 0.01, loss: { box: 1, cls: 1, dfl: 1, total: 3 }, imgs_per_s: 100, eta_s: 10 });

describe('LiveRun frame batching and chunked log trimming', () => {
  const saved = hub.api;
  let snapshot: Run;

  beforeEach(async () => {
    vi.useFakeTimers();
    snapshot = await new MockApi('normal', 0).run('run_7d1e');
  });
  afterEach(() => {
    hub.api = saved;
    vi.useRealTimers();
  });

  it('coalesces a burst of step events into a single run and version update on the next animation frame', async () => {
    let handlers: RunEventHandlers = {};
    const fake: Partial<Api> = {
      mock: false,
      run: () => Promise.resolve({ ...snapshot, status: 'running', step: 2, steps_per_epoch: 0 }),
      runMetrics: () => Promise.resolve({ steps: [step(1), step(2)], epochs: [] }),
      runLogs: () => Promise.resolve({ lines: [] }),
      runEvents: (_id, h) => {
        handlers = h;
        return () => {};
      },
    };
    hub.api = fake as Api;
    hub.mode = 'live';
    let live!: LiveRun;
    let runUpdates = 0;
    let versionUpdates = 0;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
      effect(() => {
        void live.run;
        runUpdates++;
      });
      effect(() => {
        void live.version;
        versionUpdates++;
      });
    });
    try {
      flushSync();
      await vi.advanceTimersByTimeAsync(20);
      runUpdates = 0;
      versionUpdates = 0;
      for (let i = 3; i <= 22; i++) {
        handlers.step?.({ ...step(i), epoch: i, eta_s: 100 - i, ...(i === 5 ? { steps_per_epoch: 42 } : {}) });
      }
      flushSync();
      expect(runUpdates).toBe(0);
      expect(versionUpdates).toBe(0);
      expect(live.steps.at(-1)?.step).toBe(22);
      await vi.advanceTimersByTimeAsync(20);
      expect(runUpdates).toBe(1);
      expect(versionUpdates).toBe(1);
      expect(live.run?.step).toBe(22);
      expect(live.run?.epoch).toBe(22);
      expect(live.run?.eta_s).toBe(78);
      expect(live.run?.steps_per_epoch).toBe(42);
    } finally {
      stop();
    }
  });

  it('batches live log trimming in chunks of LOG_TRIM_BATCH while keeping total line counts exact', async () => {
    const running = { ...snapshot, status: 'running' as const };
    const at = (i: number) => `2026-11-12T10:${String(Math.floor(i / 60)).padStart(2, '0')}:${String(i % 60).padStart(2, '0')}Z`;
    const initial = Array.from({ length: MAX_LOGS }, (_, i) => ({ t: at(i), level: 'info' as const, message: `init ${i}` }));
    let handlers: RunEventHandlers = {};
    const fake: Partial<Api> = {
      mock: false,
      run: () => Promise.resolve(running),
      runMetrics: () => Promise.resolve({ steps: [], epochs: [] }),
      runLogs: () => Promise.resolve({ lines: initial, total: MAX_LOGS }),
      runEvents: (_id, h) => {
        handlers = h;
        return () => {};
      },
    };
    hub.api = fake as Api;
    hub.mode = 'live';
    let live!: LiveRun;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    try {
      flushSync();
      await vi.advanceTimersByTimeAsync(0);
      expect(live.logs.length).toBe(MAX_LOGS);
      expect(live.logsEarlier).toBe(0);
      for (let i = 0; i < LOG_TRIM_BATCH - 1; i++) {
        handlers.log?.({ t: at(MAX_LOGS + i), level: 'info', message: `stream ${i}` });
      }
      expect(live.logs.length).toBe(MAX_LOGS + LOG_TRIM_BATCH - 1);
      expect(live.logsEarlier).toBe(0);
      handlers.log?.({ t: at(MAX_LOGS + LOG_TRIM_BATCH - 1), level: 'info', message: `stream ${LOG_TRIM_BATCH - 1}` });
      expect(live.logs.length).toBe(MAX_LOGS);
      expect(live.logsEarlier).toBe(LOG_TRIM_BATCH);
      // A no-total history response that slices buffered lines back down to MAX_LOGS still accounts for them.
      for (let i = 0; i < 10; i++) {
        handlers.log?.({ t: at(MAX_LOGS + LOG_TRIM_BATCH + i), level: 'info', message: `buf ${i}` });
      }
      expect(live.logs.length).toBe(MAX_LOGS + 10);
      hub.api.runLogs = () => Promise.resolve({ lines: live.logs.slice(-500) });
      handlers.reconnect?.();
      await vi.advanceTimersByTimeAsync(0);
      expect(live.logs.length).toBe(MAX_LOGS);
      expect(live.logsEarlier).toBe(LOG_TRIM_BATCH + 10);
    } finally {
      stop();
    }
  });

  it('preserves pending step progress when status arrives mid-frame without double-updating on the next frame', async () => {
    let handlers: RunEventHandlers = {};
    const fake: Partial<Api> = {
      mock: false,
      run: () => Promise.resolve({ ...snapshot, status: 'running', step: 2, steps_per_epoch: 10 }),
      runMetrics: () => Promise.resolve({ steps: [step(1), step(2)], epochs: [] }),
      runLogs: () => Promise.resolve({ lines: [] }),
      runEvents: (_id, h) => {
        handlers = h;
        return () => {};
      },
    };
    hub.api = fake as Api;
    hub.mode = 'live';
    let live!: LiveRun;
    let runUpdates = 0;
    const stop = root(() => {
      live = new LiveRun(() => 'run_7d1e');
      effect(() => {
        void live.run;
        runUpdates++;
      });
    });
    try {
      flushSync();
      await vi.advanceTimersByTimeAsync(20);
      runUpdates = 0;
      handlers.step?.({ ...step(9), epoch: 3, eta_s: 42 });
      handlers.status?.({ status: 'stopping', error: null });
      flushSync();
      expect(live.run?.status).toBe('stopping');
      expect(live.run?.step).toBe(9);
      expect(live.run?.epoch).toBe(3);
      expect(runUpdates).toBe(1);
      await vi.advanceTimersByTimeAsync(20);
      expect(runUpdates).toBe(1);
      expect(live.run?.step).toBe(9);
    } finally {
      stop();
    }
  });
});
