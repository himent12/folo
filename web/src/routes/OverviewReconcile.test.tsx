// @vitest-environment happy-dom
// SSE-to-run-list reconciliation across Overview, SidebarLive, RunWatch, and Run detail (playtest R15 + R1/R2 review).
import { flush } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { ApiError, type RunEventHandlers } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { streamedStatus } from '$lib/api/live';
import { MockApi } from '$lib/api/mock';
import { runsChanged } from '$lib/api/runs';
import type { Run } from '$lib/api/types';
import { parse, router } from '$lib/router';
import RunWatch from '$lib/shell/RunWatch.tsx';
import SidebarLive from '$lib/shell/SidebarLive';
import { activeRuns, effectiveRuns, noteRunEnded, watched } from '$lib/shell/runList';
import Overview from './Overview';
import RunPage from './Run';

let target: HTMLElement;
let c: { unmount(): void } | undefined;

afterEach(() => {
  vi.useRealTimers();
  c?.unmount();
  c = undefined;
  target?.remove();
  watched.runs = undefined;
  watched.since = null;
});

function open(api: MockApi) {
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse('#/');
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <Overview />, target);
  flush();
}

async function idleRuns(): Promise<Run[]> {
  const list = await new MockApi('normal', 0).runs();
  return list.map((r) => (r.status === 'running' || r.status === 'stopping' || r.status === 'queued' ? { ...r, status: 'completed', queue_position: null } : r));
}

describe('Overview & shell live reconciliation', () => {
  it('reconciles Overview, sidebar, and aria-live without remount oscillation when RunWatch commits after SSE completion (playtest R15 + R2 P1)', async () => {
    const api = new MockApi('normal', 0);
    const liveRun = (await api.runs()).find((r) => r.status === 'running')!;
    const runs = await idleRuns();
    const initialList = runs.map((r) => (r.id === liveRun.id ? liveRun : r));
    let handlers: RunEventHandlers = {};
    let eventOpens = 0;
    api.runEvents = (_id, h) => {
      eventOpens++;
      handlers = h;
      return () => {};
    };
    api.run = async () => liveRun;
    let resolveNextList: ((v: Run[]) => void) | null = null;
    let runsCalls = 0;
    api.runs = () => {
      runsCalls++;
      if (runsCalls === 1) return Promise.resolve(initialList);
      return new Promise<Run[]>((res) => (resolveNextList = res));
    };

    // Mount SidebarLive and Overview BEFORE RunWatch (matching App.tsx tree order) so any renderQueue/userQueue torn
    // read when RunWatch's Resource commits would immediately trigger a LiveRunCard remount and a false announcement.
    hub.api = api;
    hub.mode = 'mock';
    router.route = parse('#/');
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(
      () => (
        <>
          <SidebarLive collapsed={false} />
          <Overview />
          <RunWatch />
        </>
      ),
      target,
    );
    flush();

    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
      expect(target.querySelector('.live .card h2')?.textContent).toBe(liveRun.name);
      expect(target.querySelector('a.live-run')).not.toBeNull();
      expect(activeRuns().training).toBe(1);
      expect(streamedStatus(liveRun.id)).toBe('running');
    });
    expect(eventOpens).toBe(1);
    const seqBeforeEnd = watched.source!.committedSeq;

    // Observe every mutation to SidebarLive's polite aria-live region after completion.
    const liveRegion = target.querySelector('[aria-live="polite"]')!;
    const announcements: string[] = [];
    const obs = new MutationObserver(() => {
      const text = liveRegion.textContent?.trim();
      if (text) announcements.push(text);
    });
    obs.observe(liveRegion, { childList: true, characterData: true, subtree: true });
    onTestFinished(() => obs.disconnect());

    // SSE delivers completion while the refreshed GET /api/runs is still in flight.
    handlers.status?.({ status: 'completed', error: null });
    flush();

    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(target.querySelector('.live')?.textContent).toContain('Nothing is training');
    expect(target.querySelector('#recent-runs tbody')?.textContent).toContain(liveRun.name);
    expect(target.querySelector('#recent-runs tbody')?.textContent).toContain('Completed');
    expect(target.querySelector('a.live-run')).toBeNull();
    expect(activeRuns()).toMatchObject({ training: 0, live: false });
    expect(eventOpens).toBe(1);
    expect(runsCalls).toBe(2);
    expect(resolveNextList).not.toBeNull();

    // Authoritative list lands: wait until Resource actually commits AND RunWatch flushes, then verify zero remounts
    // and zero false "started training" announcements.
    resolveNextList!(initialList.map((r) => (r.id === liveRun.id ? { ...r, status: 'completed' } : r)));
    await vi.waitFor(() => {
      flush();
      expect(watched.source!.committedSeq).toBeGreaterThan(seqBeforeEnd);
    });
    flush();
    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(target.querySelector('#recent-runs tbody')?.textContent).toContain('Completed');
    expect(target.querySelector('a.live-run')).toBeNull();
    expect(eventOpens).toBe(1);
    expect(announcements.some((a) => a.includes('started training'))).toBe(false);
  });

  it('advances the live slot to a queued run and moves the stopped run to Recent runs on SSE stop (playtest R15)', async () => {
    const api = new MockApi('normal', 0);
    const first = (await api.runs()).find((r) => r.status === 'running')!;
    const runs = await idleRuns();
    const nextBase = runs.find((r) => r.id !== first.id)!;
    const next = { ...nextBase, status: 'queued' as const, queue_position: 1, epoch: 0, step: 0, best: null, cloud: undefined };
    const list = runs.map((r) => (r.id === first.id ? first : r.id === next.id ? next : r));
    const byId = new Map<string, RunEventHandlers>();
    api.runEvents = (id, h) => {
      byId.set(id, h);
      return () => {};
    };
    api.run = async (id) => list.find((r) => r.id === id)!;
    api.runs = () => new Promise<Run[]>(() => {});
    watched.runs = list;
    open(api);

    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
      expect(streamedStatus(first.id)).toBe('running');
    });

    byId.get(first.id)?.status?.({ status: 'stopped', error: null });
    flush();

    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(target.querySelector('.live .card h2')?.textContent).toBe(next.name);
    expect(target.querySelector('.live .card')?.textContent).toContain('Starting');
    expect(target.querySelector('#recent-runs tbody')?.textContent).toContain(first.name);
    expect(target.querySelector('#recent-runs tbody')?.textContent).toContain('Stopped');
    expect(activeRuns()).toMatchObject({ training: 0, queued: 1, live: false });
  });

  it('keeps the streamed terminal state if the post-end refresh fails, and clears it on server epoch switch (playtest R15)', async () => {
    const api = new MockApi('normal', 0);
    const liveRun = (await api.runs()).find((r) => r.status === 'running')!;
    const runs = await idleRuns();
    const initialList = runs.map((r) => (r.id === liveRun.id ? liveRun : r));
    let handlers: RunEventHandlers = {};
    api.runEvents = (_id, h) => {
      handlers = h;
      return () => {};
    };
    api.run = async () => liveRun;
    let calls = 0;
    api.runs = () => (++calls === 1 ? Promise.resolve(initialList) : Promise.reject(new ApiError(0, 'offline', 'down')));

    hub.api = api;
    hub.mode = 'mock';
    const shell = document.body.appendChild(document.createElement('div'));
    const shellView = render(() => <RunWatch />, shell);
    onTestFinished(() => {
      shellView.unmount();
      shell.remove();
    });
    open(api);

    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
      expect(streamedStatus(liveRun.id)).toBe('running');
    });

    handlers.status?.({ status: 'completed', error: null });
    flush();
    await vi.waitFor(() => expect(calls).toBeGreaterThanOrEqual(2));
    await Promise.resolve();
    flush();
    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(target.querySelector('.live')?.textContent).toContain('Nothing is training');

    hub.epoch++;
    flush();
    watched.runs = initialList;
    flush();
    expect(activeRuns().training).toBe(1);
  });

  it('retires a stale terminal override once a later healthy poll commits a resumed run, and never lets a stale override beat a live stream (R1 review)', async () => {
    const api = new MockApi('normal', 0);
    const liveRun = (await api.runs()).find((r) => r.status === 'running')!;
    const runs = await idleRuns();
    const runningList = runs.map((r) => (r.id === liveRun.id ? liveRun : r));
    let handlers: RunEventHandlers = {};
    api.runEvents = (_id, h) => {
      handlers = h;
      return () => {};
    };
    api.run = async () => liveRun;
    let calls = 0;
    api.runs = () => {
      calls++;
      if (calls === 1) return Promise.resolve(runningList);
      if (calls === 2) return Promise.reject(new ApiError(0, 'offline', 'down'));
      return Promise.resolve(runningList);
    };

    hub.api = api;
    hub.mode = 'mock';
    const shell = document.body.appendChild(document.createElement('div'));
    const shellView = render(
      () => (
        <>
          <SidebarLive collapsed={false} />
          <RunWatch />
        </>
      ),
      shell,
    );
    onTestFinished(() => {
      shellView.unmount();
      shell.remove();
    });
    open(api);

    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
      expect(streamedStatus(liveRun.id)).toBe('running');
    });

    handlers.status?.({ status: 'stopped', error: null });
    flush();
    await vi.waitFor(() => expect(calls).toBe(2));
    await Promise.resolve();
    flush();
    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(shell.querySelector('a.live-run')).toBeNull();

    // Subsequent healthy poll succeeds after recovery with the run resumed to `running`: override must retire.
    await watched.source!.refresh();
    flush();
    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
      expect(target.querySelector('.live .card h2')?.textContent).toBe(liveRun.name);
      expect(shell.querySelector('a.live-run')).not.toBeNull();
      expect(streamedStatus(liveRun.id)).toBe('running');
    });
    // Even if a stale terminal note exists while the live stream is actively `running`, effectiveRuns prefers the stream.
    api.runs = () => Promise.reject(new ApiError(0, 'offline', 'down'));
    noteRunEnded({ ...liveRun, status: 'stopped' });
    flush();
    expect(effectiveRuns(runningList)?.find((r) => r.id === liveRun.id)?.status).toBe('running');
  });

  it('keeps the terminal override when a superseded invalidate() aborts before the replacement refresh commits (R1/R2 review)', async () => {
    const api = new MockApi('normal', 0);
    const liveRun = (await api.runs()).find((r) => r.status === 'running')!;
    const runs = await idleRuns();
    const initialList = runs.map((r) => (r.id === liveRun.id ? liveRun : r));
    let handlers: RunEventHandlers = {};
    let eventOpens = 0;
    api.runEvents = (_id, h) => {
      eventOpens++;
      handlers = h;
      return () => {};
    };
    api.run = async () => liveRun;
    let resolveThird: ((v: Run[]) => void) | null = null;
    let calls = 0;
    api.runs = (signal?: AbortSignal) => {
      calls++;
      if (calls === 1) return Promise.resolve(initialList);
      if (calls === 2) {
        return new Promise<Run[]>((_, reject) => {
          signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true });
        });
      }
      return new Promise<Run[]>((res) => (resolveThird = res));
    };

    hub.api = api;
    hub.mode = 'mock';
    router.route = parse('#/');
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(
      () => (
        <>
          <SidebarLive collapsed={false} />
          <Overview />
          <RunWatch />
        </>
      ),
      target,
    );
    flush();

    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
      expect(streamedStatus(liveRun.id)).toBe('running');
    });
    expect(eventOpens).toBe(1);

    // SSE completion starts Request #2, then a concurrent runsChanged() aborts Request #2 and starts Request #3.
    handlers.status?.({ status: 'completed', error: null });
    flush();
    expect(calls).toBe(2);
    await Promise.resolve();
    runsChanged();
    const thirdSeq = watched.source!.requestSeq;
    expect(calls).toBe(3);
    await Promise.resolve();
    await Promise.resolve();
    flush();

    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(target.querySelector('.live')?.textContent).toContain('Nothing is training');
    expect(target.querySelector('a.live-run')).toBeNull();
    expect(eventOpens).toBe(1);

    // Request #3 commits the completed list: wait until Resource commits + RunWatch flushes and confirm zero remounts.
    resolveThird!(initialList.map((r) => (r.id === liveRun.id ? { ...r, status: 'completed' } : r)));
    await vi.waitFor(() => {
      flush();
      expect(watched.source!.committedSeq).toBeGreaterThanOrEqual(thirdSeq);
    });
    flush();
    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(eventOpens).toBe(1);
  });

  it('does not let an older running SSE stream override a fresh terminal poll, and reconciles terminal SSE during initial load (R1 review)', async () => {
    const api = new MockApi('normal', 0);
    const liveRun = (await api.runs()).find((r) => r.status === 'running')!;
    const runs = await idleRuns();
    const initialList = runs.map((r) => (r.id === liveRun.id ? liveRun : r));
    let handlers: RunEventHandlers = {};
    api.runEvents = (_id, h) => {
      handlers = h;
      return () => {};
    };
    api.run = async () => liveRun;
    api.runMetrics = () => new Promise(() => {});
    api.runs = () => new Promise<Run[]>(() => {});
    watched.runs = initialList;
    open(api);

    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
      expect(target.querySelector('.live .card h2')?.textContent).toBe(liveRun.name);
    });

    handlers.status?.({ status: 'completed', error: null });
    flush();
    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(target.querySelector('.live')?.textContent).toContain('Nothing is training');
    expect(target.querySelector('#recent-runs tbody')?.textContent).toContain('Completed');

    // Fresh terminal poll beats a stale running SSE stream.
    c!.unmount();
    target.remove();
    watched.runs = undefined;
    const api2 = new MockApi('normal', 0);
    api2.runEvents = () => () => {};
    api2.run = async () => liveRun;
    watched.runs = initialList;
    open(api2);
    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
      expect(streamedStatus(liveRun.id)).toBe('running');
    });
    watched.runs = initialList.map((r) => (r.id === liveRun.id ? { ...r, status: 'completed' } : r));
    flush();
    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(target.querySelector('.live')?.textContent).toContain('Nothing is training');
  });

  it('reconciles shell badge and triggers refresh when Run detail receives terminal SSE during initial load, without firing on historical runs (R2 Sol P2 #1)', async () => {
    const api = new MockApi('normal', 0);
    const liveRun = (await api.runs()).find((r) => r.status === 'running')!;
    const runs = await idleRuns();
    const doneRun = runs.find((r) => r.id !== liveRun.id)!;
    const initialList = runs.map((r) => (r.id === liveRun.id ? liveRun : r));
    const byId = new Map<string, RunEventHandlers>();
    api.runEvents = (id, h) => {
      byId.set(id, h);
      return () => {};
    };
    // Hold `/run` and `/metrics` open indefinitely for the live run so initial load stays pending.
    api.run = (id) => (id === liveRun.id ? new Promise<Run>(() => {}) : Promise.resolve(doneRun));
    api.runMetrics = (id) => (id === liveRun.id ? new Promise(() => {}) : Promise.resolve({ steps: [], epochs: [] }));
    let runsCalls = 0;
    api.runs = () => {
      runsCalls++;
      if (runsCalls === 1) return Promise.resolve(initialList);
      return new Promise<Run[]>(() => {});
    };

    hub.api = api;
    hub.mode = 'mock';
    router.route = parse(`#/runs/${liveRun.id}`);
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(
      () => (
        <>
          <SidebarLive collapsed={false} />
          <RunPage />
          <RunWatch />
        </>
      ),
      target,
    );
    flush();

    await vi.waitFor(() => {
      flush();
      expect(activeRuns().training).toBe(1);
      expect(target.querySelector('a.live-run')).not.toBeNull();
      expect(byId.has(liveRun.id)).toBe(true);
    });
    expect(runsCalls).toBe(1);

    // Terminal SSE arrives while Run detail's `/run` + `/metrics` GETs are still pending.
    byId.get(liveRun.id)?.status?.({ status: 'completed', error: null });
    flush();

    expect(activeRuns()).toMatchObject({ training: 0, live: false });
    expect(target.querySelector('a.live-run')).toBeNull();
    expect(runsCalls).toBe(2);

    // Navigating to an already-completed historical run must NOT trigger an extra run-list refresh.
    router.route = parse(`#/runs/${doneRun.id}`);
    flush();
    await vi.waitFor(() => {
      flush();
      expect(streamedStatus(doneRun.id)).toBe('completed');
    });
    flush();
    expect(runsCalls).toBe(2);
  });

  it('retires a stale terminal override on Overview when Overview falls back to its own runs Resource and gets a fresh resumed list (R2 Sol P2 #2)', async () => {
    vi.useFakeTimers();
    const api = new MockApi('normal', 0);
    const liveRun = (await api.runs()).find((r) => r.status === 'running')!;
    const runs = await idleRuns();
    const runningList = runs.map((r) => (r.id === liveRun.id ? liveRun : r));
    let handlers: RunEventHandlers = {};
    api.runEvents = (_id, h) => {
      handlers = h;
      return () => {};
    };
    api.run = async () => liveRun;
    let calls = 0;
    let failShell = false;
    api.runs = () => {
      calls++;
      if (calls === 1) return Promise.resolve(runningList);
      // Shell's post-terminal refresh (call 2) fails offline, leaving `watched.since !== null` so Overview switches to
      // its own `runs` Resource; the next 2 s shell poll stays pending while Overview's 5 s poll returns the resumed run.
      if (calls === 2) return Promise.reject(new ApiError(0, 'offline', 'down'));
      if (failShell) return new Promise<Run[]>(() => {});
      return Promise.resolve(runningList);
    };

    hub.api = api;
    hub.mode = 'mock';
    router.route = parse('#/');
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(
      () => (
        <>
          <SidebarLive collapsed={false} />
          <Overview />
          <RunWatch />
        </>
      ),
      target,
    );
    flush();

    await vi.advanceTimersByTimeAsync(10);
    flush();
    expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
    expect(streamedStatus(liveRun.id)).toBe('running');

    // SSE reports `stopped`; shell's immediate refresh (call 2) fails offline -> `watched.since !== null`.
    handlers.status?.({ status: 'stopped', error: null });
    flush();
    await vi.advanceTimersByTimeAsync(10);
    flush();
    expect(calls).toBe(2);
    expect(watched.since).not.toBeNull();
    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(target.querySelector('.live')?.textContent).toContain('Nothing is training');

    // Keep the shell's 2 s poll pending (`failShell = true`) so only Overview's 5 s poll commits fresh data.
    failShell = true;
    await vi.advanceTimersByTimeAsync(2000);
    flush();
    // Shell's 2 s poll is in flight and uncommitted: override still holds.
    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');

    // Now allow Overview's 5 s poll to succeed with the resumed `running` run while the shell's poll is still pending.
    failShell = false;
    await vi.advanceTimersByTimeAsync(3000);
    flush();
    await vi.advanceTimersByTimeAsync(10);
    flush();
    expect(watched.since).not.toBeNull();
    expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
    expect(target.querySelector('.live .card h2')?.textContent).toBe(liveRun.name);
  });
});
