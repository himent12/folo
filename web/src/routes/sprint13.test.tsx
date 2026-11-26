// @vitest-environment happy-dom
// Sprint 13 Studio regression tests:
// 1. Router navigation & guard edge cases (empty hash root no-op, replace no-op without effect subscription leak, synchronous proceed() in guard).
// 2. Chart domain/hover/line/histogram edge cases (extreme overflow, NaN/Inf, single-point series, log bands, empty histogram marker, dirty hover readout).
// 3. Playground ZoomStage non-finite guards.
// 4. Queue optimistic-update rollback & SSE sequencing when a detail refresh races with a live queue event.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { createRoot, effect, flush, signal } from 'fw';
import { render } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { LiveRun } from '$lib/api/live';
import { Resource } from '$lib/api/resource';
import { ApiError } from '$lib/api/client';
import type { Run, SystemInfo } from '$lib/api/types';
import { logDomain, niceDomain } from '$lib/charts/domain';
import { pickNearest } from '$lib/charts/lineHover';
import LineChart from '$lib/charts/LineChart';
import Histogram from '$lib/charts/Histogram.tsx';
import { parse, router } from '$lib/router';
import { clampPan, zoomToward } from './playground/ZoomStage';
import { queueState } from './run/queueState';

describe('Sprint 13 router navigation & guard edge cases', () => {
  it('treats go("/", push) from empty hash and no-op replace calls without firing guards or leaking subscriptions', () => {
    history.replaceState(null, '', '/');
    router.route = parse('');
    const seen: string[] = [];
    const unguard = router.guard((to) => {
      seen.push(to.path);
      return true;
    });
    const pushSpy = vi.spyOn(history, 'pushState');
    const replaceSpy = vi.spyOn(history, 'replaceState');
    let effectRuns = 0;
    const trigger = signal(0);
    let disposeEffect!: () => void;
    try {
      router.go('/');
      expect(seen).toEqual([]);
      expect(pushSpy).not.toHaveBeenCalled();

      router.go('/runs', { replace: true });
      expect(seen).toEqual(['/runs']);
      seen.length = 0;
      replaceSpy.mockClear();

      // Calling no-op router.go(..., { replace: true }) inside an effect must NOT subscribe it to router.location.
      createRoot((d) => {
        disposeEffect = d;
        effect(() => {
          trigger.get();
          effectRuns++;
          router.go('/runs', { replace: true });
        });
      });
      flush();
      expect(effectRuns).toBe(1);
      expect(seen).toEqual([]);
      expect(replaceSpy).not.toHaveBeenCalled();

      // Navigating elsewhere must NOT re-trigger the effect that called no-op router.go().
      router.go('/datasets', { replace: true });
      flush();
      expect(effectRuns).toBe(1);
    } finally {
      disposeEffect?.();
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

  it('LineChart and Histogram render SVGs, tables, and hover readouts cleanly with NaN/Inf/negative inputs', () => {
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
          <div class="dirty-hist">
            <Histogram
              label="Dirty hist"
              edges={[0, 0.5, 1, 1.5]}
              counts={[Number.NaN, -3, 4]}
              marker={{ x: Number.NaN, label: 'bad' }}
            />
          </div>
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

      // Focus + navigate onto dirty bins (bin 0 = NaN, bin 1 = -3) and verify visible readout stays sanitized.
      const dirtySvg = host.querySelector<SVGSVGElement>('.dirty-hist svg')!;
      dirtySvg.dispatchEvent(new FocusEvent('focus'));
      dirtySvg.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true, cancelable: true }));
      flush();
      const readout = host.querySelector('.dirty-hist .readout')!;
      expect(readout.textContent).toContain('0 boxes');
      expect(readout.textContent).not.toContain('NaN');
      dirtySvg.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
      flush();
      expect(readout.textContent).toContain('0 boxes');
      expect(readout.textContent).not.toContain('-3');
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

describe('Sprint 13 queueState optimistic rollback & SSE sequencing safety', () => {
  const prevApi = hub.api;
  const prevMode = hub.mode;
  afterEach(() => {
    hub.api = prevApi;
    hub.mode = prevMode;
    vi.restoreAllMocks();
  });

  it('does not overwrite a newer SSE queue event when a rollback detail refresh resolves late', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    hub.mode = 'mock';
    const sample = (await api.runs())[0]!;
    const makeQueued = (id: string, pos: number): Run => ({
      ...sample,
      id,
      name: id,
      status: 'queued',
      queue_position: pos,
    });

    const initial = [makeQueued('r1', 1), makeQueued('r2', 2)];
    const newerPoll = [makeQueued('r1', 1), makeQueued('r2', 2), makeQueued('r3', 3)];

    let rejectMove!: (e: Error) => void;
    vi.spyOn(api, 'moveRun').mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          rejectMove = reject;
        }),
    );

    let emitQueue!: (ev: { position: number; waiting: number }) => void;
    vi.spyOn(api, 'runEvents').mockImplementation((_id, handlers) => {
      emitQueue = (ev) => handlers.queue?.(ev);
      return () => {};
    });
    vi.spyOn(api, 'runMetrics').mockResolvedValue({ steps: [], epochs: [] });
    vi.spyOn(api, 'runLogs').mockResolvedValue({ lines: [], total: 0 });
    vi.spyOn(api, 'runs').mockResolvedValue(newerPoll);
    vi.spyOn(api, 'system').mockResolvedValue({ features: ['queue_control'] } as unknown as SystemInfo);

    let runCall = 0;
    let resolveDetailRefresh!: (r: Run) => void;
    vi.spyOn(api, 'run').mockImplementation((id) => {
      runCall++;
      if (runCall === 1) return Promise.resolve(makeQueued(id, 2));
      return new Promise((resolve) => {
        resolveDetailRefresh = resolve;
      });
    });

    let dispose!: () => void;
    let live!: LiveRun;
    let state!: ReturnType<typeof queueState>;
    let others!: Resource<Run[]>;

    createRoot((d) => {
      dispose = d;
      live = new LiveRun(() => 'r2');
      others = new Resource(() => api.runs());
      others.set(initial);
      const sys = new Resource(() => Promise.resolve({ features: ['queue_control'] } as unknown as SystemInfo));
      sys.set({ features: ['queue_control'] } as unknown as SystemInfo);
      state = queueState({ run: () => live.run, live, others, system: sys });
    });

    try {
      await vi.waitFor(() => {
        flush();
        expect(live.run?.id).toBe('r2');
        expect(live.run?.queue_position).toBe(2);
      });

      const pending = state.move({ action: 'up' });
      // Optimistic update applied immediately.
      expect(live.run?.queue_position).toBe(1);
      expect(others.data?.find((r) => r.id === 'r2')?.queue_position).toBe(1);

      // While the move is in flight, a newer poll and SSE queue event (pos 3) arrive.
      others.set(newerPoll);
      emitQueue({ position: 3, waiting: 3 });
      expect(live.run?.queue_position).toBe(3);

      // Move fails: rollback triggers detail refresh (snapping old pos 3), does NOT revert `others` to `initial`.
      rejectMove(new ApiError(500, 'err', 'server failed'));
      await pending;
      await vi.waitFor(() => expect(resolveDetailRefresh).toBeTypeOf('function'));

      // Before the detail refresh returns, another SSE queue event advances position to 4!
      emitQueue({ position: 4, waiting: 4 });
      expect(live.run?.queue_position).toBe(4);

      // Now the stale detail refresh resolves with position 3: LiveRun sequencing MUST keep position 4!
      resolveDetailRefresh(makeQueued('r2', 3));
      await Promise.resolve();
      await Promise.resolve();

      expect(others.data).not.toBe(initial);
      expect(others.data?.length).toBe(3);
      expect(live.run?.queue_position).toBe(4);
    } finally {
      dispose();
    }
  });
});
