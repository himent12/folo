// @vitest-environment happy-dom
// A late Stop answer (issue #382): the stream may say the run ended while the stop was on its way.
import { createRoot, flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { stopDescription, stoppedToast } from '../../routes/run/StopRun';
import { toast } from '$lib/ui/toast';
import type { Api, RunEventHandlers } from './client';
import { hub } from './index';
import { LiveRun } from './live';
import { MockApi } from './mock';
import type { Run, RunMetrics } from './types';

const root = (fn: () => void) => createRoot((dispose) => (fn(), dispose));

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

describe('LiveRun.stop', () => {
  let handlers: RunEventHandlers = {};
  let snapshot: Run;
  let serverStatus: Run['status'] = 'running';
  const saved = hub.api;

  beforeEach(async () => {
    vi.useFakeTimers();
    serverStatus = 'running';
    snapshot = await new MockApi('normal', 0).run('run_7d1e');
    const fake: Partial<Api> = {
      mock: false,
      run: () => Promise.resolve({ ...snapshot, status: serverStatus, step: 2 }),
      runMetrics: () => Promise.resolve<RunMetrics>({ steps: [], epochs: [] }),
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
    for (const t of toast.items) toast.dismiss(t.id);
  });

  async function started() {
    let live!: LiveRun;
    const dispose = root(() => {
      live = new LiveRun(() => 'run_7d1e');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('running');
    return { live, dispose };
  }

  it('keeps a Completed the stream reported while the stop was on its way', async () => {
    const { live, dispose } = await started();
    const answer = deferred<Run>();
    const pending = live.stop(() => answer.promise);
    // The stop came after the last batch: the run completes, and the stream says so before the answer lands.
    serverStatus = 'completed';
    handlers.status?.({ status: 'completed', error: null });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.run?.status).toBe('completed');
    answer.resolve({ ...snapshot, status: 'stopping', step: 2 });
    const now = await pending;
    expect(now?.status).toBe('completed');
    expect(live.run?.status).toBe('completed');
    stoppedToast(now!, 0);
    expect(toast.items.map((t) => [t.title, t.detail])).toEqual([[`${snapshot.name} completed`, 'It finished before the stop took effect, with all its results.']]);
    dispose();
  });

  it('keeps an end a snapshot brought while the stop was on its way (no status event)', async () => {
    const { live, dispose } = await started();
    const answer = deferred<Run>();
    const pending = live.stop(() => answer.promise);
    live.run = { ...live.run!, status: 'completed' };
    answer.resolve({ ...snapshot, status: 'stopping', step: 2 });
    expect((await pending)?.status).toBe('completed');
    expect(live.run?.status).toBe('completed');
    dispose();
  });

  it('keeps a newer attempt: stopped, then resumed elsewhere, before the answer came (review of #389, round 2)', async () => {
    const { live, dispose } = await started();
    const answer = deferred<Run>();
    const pending = live.stop(() => answer.promise);
    serverStatus = 'stopped';
    handlers.status?.({ status: 'stopped', error: null });
    await vi.advanceTimersByTimeAsync(0);
    serverStatus = 'running';
    handlers.status?.({ status: 'running', error: null });
    await vi.advanceTimersByTimeAsync(0);
    answer.resolve({ ...snapshot, status: 'stopping', step: 2 });
    expect((await pending)?.status).toBe('running');
    expect(live.run?.status).toBe('running');
    dispose();
  });

  it('applies the answer when no status event came meanwhile', async () => {
    const { live, dispose } = await started();
    const now = await live.stop(() => Promise.resolve({ ...snapshot, status: 'stopping', step: 2 }));
    expect(now?.status).toBe('stopping');
    expect(live.run?.status).toBe('stopping');
    dispose();
  });

  it("applies the answer over an active state the stream reported meanwhile (a reconnect's running), keeping its progress", async () => {
    const { live, dispose } = await started();
    const answer = deferred<Run>();
    const pending = live.stop(() => answer.promise);
    // The stream reconnects while the stop is out: its snapshot still says running, a step further on.
    handlers.status?.({ status: 'running', error: null });
    live.run = { ...live.run!, step: 5 };
    answer.resolve({ ...snapshot, status: 'stopping', step: 2 });
    const now = await pending;
    expect([now?.status, now?.step]).toEqual(['stopping', 5]);
    expect(live.run?.status).toBe('stopping');
    dispose();
  });
});

describe('stoppedToast', () => {
  afterEach(() => {
    for (const t of toast.items) toast.dismiss(t.id);
  });

  it('does not claim nothing is saved before the run has stopped', () => {
    // A stop during the first epoch's validation keeps that epoch (docs/API.md); only the stop's outcome settles it.
    stoppedToast({ name: 'r', epoch: 1, status: 'stopping' }, 0);
    expect(toast.items.map((t) => t.detail)).toEqual(['No epoch had finished yet. Unless one finishes before it stops, nothing is saved.']);
  });

  it('tells a run that ended before the answer came as it ended, claiming only the epochs known finished', () => {
    // A `stopped` event leaves the epoch its last step was in (review of #389, round 2): epoch 1, not finished.
    stoppedToast({ name: 'r', epoch: 1, status: 'stopped' }, 0);
    stoppedToast({ name: 's', epoch: 3, status: 'stopped' }, 2);
    stoppedToast({ name: 'f', epoch: 1, status: 'failed' }, 0);
    expect(toast.items.map((t) => [t.title, t.detail])).toEqual([
      ['Stopped r', undefined],
      ['Stopped s', 'The checkpoint from epoch 2 is kept.'],
      ['f failed', 'Its page has the error and the log.'],
    ]);
  });

  it('the confirm does not promise nothing is saved either', () => {
    expect(stopDescription({ status: 'running' }, 0)).toBe('Training stops after the current step. No epoch has finished yet: unless one finishes first, nothing is saved.');
  });
});
