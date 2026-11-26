// @vitest-environment happy-dom
// Sprint 13 Studio regression tests:
// 1. Router navigation & guard edge cases (empty hash root no-op, replace no-op, synchronous proceed() in guard).
// 2. Chart domain/hover/line/histogram edge cases (extreme overflow, NaN/Inf, single-point series, log bands, empty histogram marker).
// 3. Playground ZoomStage non-finite guards.
// 4. Queue optimistic-update rollback does not clobber newer poll/live state when a move fails.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createRoot, flush, signal } from 'fw';
import { render } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { Resource } from '$lib/api/resource';
import { ApiError } from '$lib/api/client';
import type { LiveRun } from '$lib/api/live';
import type { Run, SystemInfo } from '$lib/api/types';
import { logDomain, niceDomain } from '$lib/charts/domain';
import { pickNearest } from '$lib/charts/lineHover';
import LineChart from '$lib/charts/LineChart';
import Histogram from '$lib/charts/Histogram.tsx';
import { parse, router } from '$lib/router';
import { clampPan, zoomToward } from './playground/ZoomStage';
import { queueState } from './run/queueState';

describe('Sprint 13 router navigation & guard edge cases', () => {
  it('treats go("/", push) from empty hash and no-op replace calls without firing guards or adding entries', () => {
    history.replaceState(null, '', '/');
    router.route = parse('');
    const seen: string[] = [];
    const unguard = router.guard((to) => {
      seen.push(to.path);
      return true;
    });
    const pushSpy = vi.spyOn(history, 'pushState');
    const replaceSpy = vi.spyOn(history, 'replaceState');
    try {
      router.go('/');
      expect(seen).toEqual([]);
      expect(pushSpy).not.toHaveBeenCalled();

      router.go('/runs', { replace: true });
      expect(seen).toEqual(['/runs']);
      seen.length = 0;
      replaceSpy.mockClear();

      // Exact same path with replace:true is a no-op and must not re-run guards.
      router.go('/runs', { replace: true });
      expect(seen).toEqual([]);
      expect(replaceSpy).not.toHaveBeenCalled();
    } finally {
      unguard();
      pushSpy.mockRestore();
      replaceSpy.mockRestore();
      router.go('/', { replace: true });
    }
  });

  it('completes hashchange navigation when a guard calls proceed() synchronously before returning false', () => {
    router.go('/runs', { replace: true });
    const goSpy = vi.spyOn(history, 'go').mockImplementation(() => {});
    const unguard = router.guard((_to, proceed) => {
      proceed();
      return false;
    });
    try {
      history.replaceState(null, '', '#/datasets');
      window.dispatchEvent(new HashChangeEvent('hashchange'));
      expect(router.route.name).toBe('datasets');
      expect(goSpy).not.toHaveBeenCalled();
    } finally {
      unguard();
      goSpy.mockRestore();
      router.go('/', { replace: true });
    }
  });
});

describe('Sprint 13 charts edge cases', () => {
  it('handles extreme overflow in niceDomain and non-finite/reversed bounds in logDomain', () => {
    const ext = niceDomain(-1e308, 1e308);
    expect(Number.isFinite(ext.domain[0])).toBe(true);
    expect(Number.isFinite(ext.domain[1])).toBe(true);
    expect(ext.ticks.every(Number.isFinite)).toBe(true);

    expect(logDomain(0.1, Number.POSITIVE_INFINITY)).toEqual([0.1, 1]);
    expect(logDomain(Number.NaN, 10)).toEqual([0.1, 1]);
    const reversed = logDomain(100, 1);
    expect(reversed[0]).toBeLessThan(reversed[1]);
  });

  it('pickNearest selects single-point series and skips NaN/Infinity points', () => {
    const xScale = (v: number) => v * 10;
    const yScale = (v: number) => 100 - v * 100;
    const one = pickNearest(
      () => true,
      () => true,
      [{ id: 'solo', label: 'solo', color: '#f00', points: [{ x: 3, y: 0.7 }] }],
      xScale,
      yScale,
      30,
      30,
      12,
    );
    expect(one).toBe('solo');

    const mixed = pickNearest(
      () => true,
      () => true,
      [
        {
          id: 's',
          label: 's',
          color: '#0f0',
          points: [
            { x: Number.NaN, y: 1 },
            { x: 2, y: Number.POSITIVE_INFINITY },
            { x: 5, y: 0.4 },
          ],
        },
      ],
      xScale,
      yScale,
      50,
      60,
      12,
    );
    expect(mixed).toBe('s');
  });

  it('LineChart and Histogram render SVGs and tables cleanly with NaN/Inf/single-point/log-band/empty-marker inputs', () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    const view = render(
      () => (
        <div>
          <LineChart
            label="Loss"
            yLog
            series={[
              {
                id: 'a',
                label: 'train',
                color: 'var(--chart-1)',
                points: [
                  { x: Number.NaN, y: 1 },
                  { x: 1, y: 0.5 },
                  { x: 2, y: 0.25 },
                ],
                lower: [
                  { x: 1, y: 0 },
                  { x: 2, y: 0.1 },
                ],
              },
            ]}
            markers={[{ x: 1, y: 0, label: 'invalid-log-y' }]}
            xDomain={[1, 2]}
          />
          <Histogram label="Empty hist" edges={[]} counts={[]} marker={{ x: 0.5, label: 'median' }} />
          <Histogram
            label="Dirty hist"
            edges={[0, 0.5, 1, 1.5]}
            counts={[Number.NaN, -3, 4]}
            marker={{ x: Number.NaN, label: 'bad' }}
          />
        </div>
      ),
      host,
    );
    try {
      for (const el of host.querySelectorAll('.plot, .hist')) {
        Object.defineProperty(el, 'clientWidth', { value: 400, configurable: true });
        el.dispatchEvent(new Event('resize'));
      }
      flush();
      expect(host.querySelectorAll('svg')).toHaveLength(3);
      expect(host.innerHTML).not.toContain('NaN');
      expect(host.innerHTML).not.toContain('Infinity');
    } finally {
      view.unmount();
      host.remove();
    }
  });
});

describe('Sprint 13 ZoomStage non-finite guards', () => {
  it('clampPan and zoomToward never produce NaN/Infinity for invalid inputs', () => {
    expect(clampPan(Number.NaN, 400, 2)).toBe(0);
    expect(clampPan(10, Number.POSITIVE_INFINITY, 2)).toBe(0);
    expect(clampPan(10, 400, Number.NaN)).toBe(0);

    const z1 = zoomToward({ z: 2, tx: 10, ty: -5 }, Number.NaN, 100, 100, 400, 300);
    expect(Number.isFinite(z1.z) && Number.isFinite(z1.tx) && Number.isFinite(z1.ty)).toBe(true);
    const z2 = zoomToward({ z: Number.NaN, tx: Number.NaN, ty: Number.NaN }, 2, Number.NaN, 50, 400, 300);
    expect(z2).toEqual({ z: 2, tx: 0, ty: 0 });
  });
});

describe('Sprint 13 queueState optimistic rollback safety', () => {
  const prevApi = hub.api;
  afterEach(() => {
    hub.api = prevApi;
    vi.restoreAllMocks();
  });

  it('does not overwrite newer poll data with stale pre-move state when move fails', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    const sample = (await api.runs())[0]!;
    let rejectMove!: (e: Error) => void;
    vi.spyOn(api, 'moveRun').mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          rejectMove = reject;
        }),
    );

    const makeQueued = (id: string, pos: number): Run => ({
      ...sample,
      id,
      name: id,
      status: 'queued',
      queue_position: pos,
    });

    const initial = [makeQueued('r1', 1), makeQueued('r2', 2)];
    const newerPoll = [makeQueued('r1', 1), makeQueued('r2', 2), makeQueued('r3', 3)];
    vi.spyOn(api, 'runs').mockResolvedValue(newerPoll);
    vi.spyOn(api, 'run').mockResolvedValue(makeQueued('r2', 3));
    vi.spyOn(api, 'system').mockResolvedValue({ features: ['queue_control'] } as unknown as SystemInfo);

    const cur = signal<Run | undefined>(initial[1]);
    const live = { run: { ...initial[1]! }, waiting: 2 } as unknown as LiveRun;
    let dispose!: () => void;
    let state!: ReturnType<typeof queueState>;
    let others!: Resource<Run[]>;

    createRoot((d) => {
      dispose = d;
      others = new Resource(() => api.runs());
      others.set(initial);
      const sys = new Resource(() => Promise.resolve({ features: ['queue_control'] } as unknown as SystemInfo));
      sys.set({ features: ['queue_control'] } as unknown as SystemInfo);
      state = queueState({ run: () => cur.get(), live, others, system: sys });
    });

    try {
      const pending = state.move({ action: 'up' });
      // Optimistic update applied immediately.
      expect(live.run?.queue_position).toBe(1);
      expect(others.data?.find((r) => r.id === 'r2')?.queue_position).toBe(1);

      // A concurrent poll arrives while the move request is still in flight.
      others.set(newerPoll);
      live.run = { ...live.run!, queue_position: 3 };

      // Now the move request fails: rollback must NOT clobber `newerPoll` back to `initial`!
      rejectMove(new ApiError(500, 'err', 'server failed'));
      await pending;
      await Promise.resolve();

      expect(others.data).not.toBe(initial);
      expect(others.data?.length).toBe(3);
      expect(live.run?.queue_position).toBe(3);
    } finally {
      dispose();
    }
  });
});
