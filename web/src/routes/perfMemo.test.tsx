// @vitest-environment happy-dom
// Memoization and render-work invariants for high-frequency chart transforms and large table pipelines.
import { createRoot, effect, flush, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AppendLog } from '$lib/api/appendLog';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Epoch, Run, Step } from '$lib/api/types';
import XYChart from '$lib/charts/XYChart';
import type { XYSeries } from '$lib/charts/xy';
import { router } from '$lib/router';
import { render } from '../../framework/testing';
import { compareState } from './compare/state';
import { IPS_WINDOW, RunModel } from './run/runModel';
import Runs from './Runs';
import { sortRuns } from './runs/listState';

vi.mock('./runs/listState', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./runs/listState')>();
  return { ...actual, sortRuns: vi.fn(actual.sortRuns) };
});

let target: HTMLElement;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  hub.epoch++;
  window.matchMedia ??= (() => ({ matches: false, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  target.remove();
});

function makeStep(step: number, epoch: number, lr: number, ips: number, total = 1.5): Step {
  return {
    step,
    epoch,
    loss: { total, box: total * 0.4, cls: total * 0.3, dfl: total * 0.3 },
    lr,
    imgs_per_s: ips,
    eta_s: 120,
  };
}

describe('RunModel per-step chart transform memoization', () => {
  it('computes lrFit, ipsFit, and ips window accurately while keeping lossLogStatus stable across SSE steps', async () => {
    const stepsLog = new AppendLog<Step>();
    const epochsLog = new AppendLog<Epoch>();
    const version = signal(0);
    const baseRun = (await hub.api.runs())[0]!;
    const run = signal<Run>({
      ...baseRun,
      id: 'run_perf',
      name: 'perf-run',
      status: 'running',
      finished: null,
      epoch: 1,
      epochs: 50,
      step: 1,
      steps_per_epoch: 100,
      best: null,
      eta_s: 120,
    });

    for (let i = 1; i <= 150; i++) {
      stepsLog.push(makeStep(i, i <= 100 ? 1 : 2, 0.001 + i * 0.00001, 100 + (i % 20), 2.0 - i * 0.005));
    }
    version.set(1);

    let logStatusRuns = 0;
    let model!: RunModel;
    const stop = createRoot((d) => {
      model = new RunModel({
        get run() {
          return run.get();
        },
        steps: stepsLog,
        epochs: epochsLog,
        get version() {
          return version.get();
        },
      });
      effect(() => {
        void model.lossLogStatus.get();
        logStatusRuns++;
      });
      return d;
    });
    flush();
    expect(logStatusRuns).toBe(1);

    const lrFit = model.lrFit(() => 200);
    const ipsFit = model.ipsFit(() => 200);
    const lrDom1 = lrFit.get();
    const ipsDom1 = ipsFit.get();
    expect(lrDom1).not.toBeNull();
    expect(lrDom1!.domain[0]).toBeLessThanOrEqual(0.00101);
    expect(lrDom1!.domain[1]).toBeGreaterThanOrEqual(0.0025);
    expect(ipsDom1).not.toBeNull();
    expect(ipsDom1!.domain[0]).toBe(0);
    expect(ipsDom1!.domain[1]).toBeGreaterThanOrEqual(100);

    // ips window matches last IPS_WINDOW steps without slice/reduce drift.
    const ips1 = model.ips.get();
    expect(ips1.n).toBe(IPS_WINDOW);
    expect(ips1.trend).toHaveLength(IPS_WINDOW);
    const expectedAvg = stepsLog.items.slice(-IPS_WINDOW).reduce((s, x) => s + x.imgs_per_s, 0) / IPS_WINDOW;
    expect(ips1.avg).toBeCloseTo(expectedAvg, 10);

    // Appending 25 positive SSE steps does not re-trigger lossLogStatus subscribers.
    for (let i = 151; i <= 175; i++) {
      stepsLog.push(makeStep(i, 2, 0.0025, 110, 1.1));
      version.set(i);
      flush();
    }
    expect(logStatusRuns).toBe(1);

    // A non-positive loss value flips skippedNonPositive and notifies subscribers once.
    stepsLog.push(makeStep(176, 2, 0.0025, 110, 0));
    version.set(176);
    flush();
    expect(logStatusRuns).toBe(2);
    expect(model.lossLogStatus.get().nonPositive).toBeGreaterThan(0);

    stop();
  });
});

describe('compareState per-run series memoization', () => {
  it('reuses unchanged run point arrays and rev across real metrics refreshes when another compared run updates', async () => {
    const runs = await hub.api.runs();
    const pair = runs.filter((r) => r.status === 'completed').slice(0, 2);
    expect(pair).toHaveLength(2);
    const [rA, rB] = pair as [Run, Run];
    const mA0 = await hub.api.runMetrics(rA.id);
    const mB0 = await hub.api.runMetrics(rB.id);
    let bumpB = false;
    vi.spyOn(hub.api, 'runMetrics').mockImplementation(async (id) => {
      if (id === rA.id) return structuredClone(mA0);
      const b = structuredClone(mB0);
      if (bumpB && b.epochs.length > 0) {
        const last = b.epochs[b.epochs.length - 1]!;
        b.epochs.push({ ...last, epoch: last.epoch + 1, val: { ...last.val, map50_95: last.val.map50_95 + 0.01 } });
      }
      return b;
    });

    router.go(`/runs/compare?ids=${rA.id},${rB.id}`, { replace: true });
    let st!: ReturnType<typeof compareState>;
    const stop = createRoot((d) => {
      st = compareState();
      return d;
    });
    try {
      await vi.waitFor(() => {
        flush();
        expect(st.series.get()).toHaveLength(2);
        expect(st.series.get()[0]!.points.length).toBeGreaterThan(0);
        expect(st.series.get()[1]!.points.length).toBeGreaterThan(0);
      });

      const firstA = st.series.get()[0]!;
      const firstB = st.series.get()[1]!;

      // Real full metrics refresh returning fresh cloned JSON for both runs, with only rB advancing.
      bumpB = true;
      await st.metrics.refresh();
      flush();

      const secondA = st.series.get()[0]!;
      const secondB = st.series.get()[1]!;
      expect(secondA.points).toBe(firstA.points);
      expect(secondA.rev).toBe(firstA.rev);
      expect(secondB.points).not.toBe(firstB.points);
      expect(secondB.rev).not.toBe(firstB.rev);
      expect(st.valueOf(rA.id)).toBe(firstA.points.at(-1)?.y ?? null);

      // Switching metric invalidates cached point arrays for both runs.
      st.metricId.set('m50');
      flush();
      expect(st.series.get()[0]!.points).not.toBe(firstA.points);
    } finally {
      stop();
    }
  });
});

describe('XYChart scale/path memoization', () => {
  it('reuses cached path strings across equivalent domain arrays and single-series updates', () => {
    const cwSpy = vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(480);
    const ptsA = [
      { x: 0, y: 0.2 },
      { x: 1, y: 0.8 },
    ];
    const ptsB = signal([
      { x: 0, y: 0.3 },
      { x: 1, y: 0.6 },
    ]);
    const xDom = signal<[number, number]>([0, 1]);
    const yDom = signal<[number, number]>([0, 1]);
    const hl = signal<string | null>(null);

    const view = render(
      () => (
        <XYChart
          label="test xy"
          series={[
            { id: 'a', label: 'A', color: 'var(--chart-1)', points: ptsA } as XYSeries,
            { id: 'b', label: 'B', color: 'var(--chart-2)', points: ptsB.get() } as XYSeries,
          ]}
          xDomain={xDom.get()}
          yDomain={yDom.get()}
          xTicks={[0, 0.5, 1]}
          yTicks={[0, 0.5, 1]}
          highlight={hl.get()}
          readout={() => <span>legend</span>}
          table={{ caption: 't', head: ['x', 'y'], rows: [] }}
        />
      ),
      target,
    );
    try {
      flush();
      const pathsBefore = [...target.querySelectorAll<SVGPathElement>('path.line')].map((p) => p.getAttribute('d'));
      expect(pathsBefore).toHaveLength(2);
      expect(pathsBefore[0]).toBeTruthy();

      // New equivalent domain tuple references keep the rendered paths identical.
      xDom.set([0, 1]);
      yDom.set([0, 1]);
      flush();
      expect([...target.querySelectorAll<SVGPathElement>('path.line')].map((p) => p.getAttribute('d'))).toEqual(pathsBefore);

      // Updating only series B updates B's path while keeping A's path identical.
      ptsB.set([
        { x: 0, y: 0.1 },
        { x: 1, y: 0.9 },
      ]);
      flush();
      const pathsAfter = [...target.querySelectorAll<SVGPathElement>('path.line')].map((p) => p.getAttribute('d'));
      expect(pathsAfter[0]).toBe(pathsBefore[0]);
      expect(pathsAfter[1]).not.toBe(pathsBefore[1]);

      // Highlighting series A marks B as context (receded) via memoized activeHighlight.
      hl.set('a');
      flush();
      expect(target.querySelectorAll('path.line.context')).toHaveLength(1);
    } finally {
      cwSpy.mockRestore();
      view.unmount();
    }
  });
});

describe('Runs table clock/filter/sort pipeline', () => {
  it('skips re-sorting on clock ticks unless sorting by duration with a live run', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] });
    const seed = await hub.api.runs();
    const completed = seed.filter((r) => r.status === 'completed' && r.started && r.finished);
    const liveRun = seed.find((r) => r.status === 'running' && r.started && !r.finished);
    expect(completed.length).toBeGreaterThan(0);
    expect(liveRun).toBeDefined();

    const runsList: Run[] = [
      liveRun!,
      ...Array.from({ length: 10 }, (_, i) => ({
        ...completed[i % completed.length]!,
        id: `run_done_${i}`,
        name: `done-${i}`,
      })),
    ];
    vi.spyOn(hub.api, 'runs').mockResolvedValue(runsList);
    router.go('/runs', { replace: true });
    const sortSpy = vi.mocked(sortRuns);

    const view = render(() => <Runs />, target);
    try {
      await vi.waitFor(() => {
        flush();
        expect(target.querySelectorAll('tbody tr').length).toBe(11);
      });

      sortSpy.mockClear();
      // On default `created` sort, advancing the 30s clock tick must NOT re-sort the runs table.
      vi.advanceTimersByTime(65_000);
      flush();
      expect(sortSpy).not.toHaveBeenCalled();

      // Switching to `duration` sort sorts immediately, and subsequent 30s clock ticks re-sort for the live run.
      const durBtn = [...target.querySelectorAll<HTMLButtonElement>('th button')].find((b) => b.textContent?.includes('Duration'))!;
      expect(durBtn).toBeDefined();
      durBtn.click();
      flush();
      expect(sortSpy).toHaveBeenCalledTimes(1);

      vi.advanceTimersByTime(30_000);
      flush();
      expect(sortSpy).toHaveBeenCalledTimes(2);
    } finally {
      view.unmount();
    }
  });
});
