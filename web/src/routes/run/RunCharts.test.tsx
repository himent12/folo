// @vitest-environment happy-dom
// A run's KPI strip and its four charts (design/UNIFIED.md §4 Run): the live strip in its first minute (runs F14: the
// epoch in progress, the start's phase, no time left until it has settled), a history that failed to load (one error,
// one Retry), a run that detected nothing yet (said in words), and the one cursor the four charts share.
import { flush as flushSync, tick } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { AppendLog } from '$lib/api/appendLog';
import { MockApi } from '$lib/api/mock';
import { ETA_PENDING } from '$lib/api/progress';
import type { Epoch, Run, Step } from '$lib/api/types';
import type { EvalSource } from '$lib/evaluated';
import RunCharts from './RunCharts';
import { RunModel } from './runModel';

let target: HTMLElement;
let real: typeof window.matchMedia;
let base: Run;

beforeEach(async () => {
  real = window.matchMedia;
  window.matchMedia = ((q: string) => ({
    matches: false,
    media: q,
    addEventListener() {},
    removeEventListener() {},
  })) as unknown as typeof window.matchMedia;
  target = document.createElement('div');
  document.body.appendChild(target);
  base = await new MockApi('normal', 0).run('run_7d1e');
});
afterEach(() => {
  window.matchMedia = real;
  target.remove();
});

const step = (n: number, spe = 50): Step => ({
  epoch: Math.ceil(n / spe),
  step: n,
  lr: 1e-3,
  loss: { box: 1, cls: 1, dfl: 1, total: 3 },
  imgs_per_s: 40,
  eta_s: 200,
  steps_per_epoch: spe,
  epochs: 5,
});
const epoch = (n: number, map = 0.3): Epoch => ({
  epoch: n,
  train_loss: { box: 1, cls: 1, dfl: 1, total: 3 },
  val: {
    map50: map * 1.5,
    map50_95: map,
    precision: 0.5,
    recall: 0.4,
    best_f1_conf: 0.3,
    per_class: [],
  },
  duration_s: 10,
});

function show(
  run: Run,
  o: {
    steps?: Step[];
    epochs?: Epoch[];
    error?: unknown;
    retry?: () => void;
    evalOn?: EvalSource;
  } = {},
) {
  const steps = new AppendLog<Step>();
  const epochs = new AppendLog<Epoch>();
  for (const s of o.steps ?? []) steps.push(s);
  for (const e of o.epochs ?? []) epochs.push(e);
  let model!: RunModel;
  const c = render(() => {
    model = new RunModel({ run, steps, epochs, version: 1 });
    return <RunCharts model={model} loading={false} evalOn={o.evalOn ?? null} error={o.error} retry={o.retry ?? (() => {})} never={false} phone={false} />;
  }, target);
  flushSync();
  return { c, model };
}

/** A tile's label, value (with its unit) and caption. */
const tile = (label: string) => {
  const t = [...target.querySelectorAll('.kpis .tile')].find((x) => x.querySelector('.label')?.textContent?.startsWith(label));
  return t
    ? {
        value: t.querySelector('.value')?.textContent?.replace(/\s+/g, ' ').trim(),
        sub: t.querySelector('.sub')?.textContent?.trim(),
      }
    : undefined;
};

describe('the live KPI strip', () => {
  it('says "1 / 5", and the time left only once the first epoch is done', () => {
    const run = {
      ...base,
      status: 'running',
      epoch: 1,
      epochs: 5,
      step: 3,
      steps_per_epoch: 50,
      eta_s: 240,
      best: null,
    } as const;
    let v = show(run, { steps: [step(3)] });
    expect(tile('Epoch')).toEqual({ value: '1/ 5', sub: '1%' });
    expect(tile('Time left')).toEqual({ value: '—', sub: ETA_PENDING });
    v.c.unmount();
    v = show({ ...run, step: 50 }, { steps: [step(50)] });
    expect(tile('Time left')?.value).toBe('4m');
    v.c.unmount();
  });

  it('names the start while no step has come yet', () => {
    const { c } = show({
      ...base,
      status: 'running',
      epoch: 0,
      epochs: 5,
      step: 0,
      eta_s: null,
      best: null,
    });
    expect(tile('Epoch')?.sub).toBe('Starting the first epoch…');
    c.unmount();
  });

  it('gives the change since the last epoch beside the best while it trains, and Duration once it ended', () => {
    const run = {
      ...base,
      status: 'running',
      epoch: 2,
      epochs: 5,
      step: 100,
      steps_per_epoch: 50,
      eta_s: 240,
      best: { epoch: 1, map50_95: 0.312 },
    } as const;
    let v = show(run, {
      steps: [step(100)],
      epochs: [epoch(1, 0.312), epoch(2, 0.3)],
    });
    expect(tile('mAP50-95 · latest')).toEqual({
      value: '0.300',
      sub: `▼ 0.012 vs last · best 0.312`,
    });
    expect(tile('Precision / Recall')?.value).toBe('0.500 / 0.400');
    v.c.unmount();
    v = show({ ...run, status: 'completed', epoch: 5 }, { steps: [step(250)], epochs: [epoch(1, 0.312), epoch(2, 0.3)] });
    expect(tile('mAP50-95 · final')?.sub).toBe('best 0.312 · ep 1');
    expect(tile('Duration')).toBeDefined();
    expect(tile('Time left')).toBeUndefined();
    expect(tile('Final throughput')).toBeDefined();
    v.c.unmount();
  });
});

describe('the run charts when the history failed to load', () => {
  it('says so with Retry, never "Waiting for the first validation…"', () => {
    let retried = 0;
    const { c } = show(base, {
      error: new Error('HTTP 500'),
      retry: () => retried++,
    });
    expect(target.textContent).toContain("Could not load this run's metrics");
    expect(target.textContent).not.toMatch(/Waiting for the first|after the first epoch/);
    [...target.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Retry')!.click();
    expect(retried).toBe(1);
    c.unmount();
  });
});

describe('the run charts before anything is detected', () => {
  it('says "no detections yet" instead of "best 0.000 · ep 1" and "conf 1.00"', () => {
    const val = {
      map50: 0,
      map50_95: 0,
      precision: 0,
      recall: 0,
      best_f1_conf: 1,
      per_class: [],
    };
    const run = {
      ...base,
      status: 'running' as const,
      best: { epoch: 1, map50_95: 0 },
    };
    const { c } = show(run, {
      epochs: [
        {
          epoch: 1,
          train_loss: { box: 1, cls: 1, dfl: 1, total: 3 },
          val,
          duration_s: 10,
        },
      ],
    });
    expect(tile('mAP50-95')?.sub).toBe('no detections yet');
    expect(tile('Precision / Recall')?.sub).toBe('no detections yet');
    expect(target.textContent).not.toMatch(/best 0\.000|conf 1\.00/);
    c.unmount();
  });
});

describe('the four run charts', () => {
  it('are Validation mAP and Training loss, then Precision / Recall and Learning rate, with no metric tabs', () => {
    const evalOn: EvalSource = {
      datasetId: base.config.dataset_id,
      datasetName: 'shapes',
      split: 'val',
      fallback: null,
    };
    const { c } = show({ ...base, status: 'completed' }, { steps: [step(50), step(100)], epochs: [epoch(1), epoch(2)], evalOn });
    const title = (t: Element) => t.textContent!.replace(t.querySelector('.meta')?.textContent ?? '', '').trim();
    expect([...target.querySelectorAll('.run-chart .title')].map(title)).toEqual(['Validation mAP', 'Training loss', 'Precision / Recall', 'Learning rate']);
    expect(target.querySelector('[role=tablist], select[aria-label="Metric to chart"]')).toBeNull();
    // The series hues (design/UNIFIED.md §3): mAP50-95 sky with mAP50 amber dashed beside it, precision magenta, recall teal.
    const legend = (id: string) => [...target.querySelectorAll(`#${id} .legend li .name`)].map((n) => n.textContent?.trim());
    expect(legend('chart-map')).toEqual(['mAP50-95', 'mAP50', 'best']);
    expect(legend('chart-pr')).toEqual(['precision', 'recall']);
    c.unmount();
  });

  it("show the throughput's whole history in the fourth chart, picked in its header", () => {
    const steps = Array.from({ length: 300 }, (_, i) => ({ ...step(i + 1), imgs_per_s: i < 100 ? 20 : 40 }));
    const { c, model } = show({ ...base, status: 'completed', epochs: 6, epoch: 6, step: 300 }, { steps, epochs: [epoch(1), epoch(2)] });
    const chart = () => target.querySelector('#chart-lr')!;
    const title = () => chart().querySelector('.title')!.firstChild?.textContent?.trim();
    expect(title()).toBe('Learning rate');
    [...chart().querySelectorAll<HTMLElement>('[role=radio], button')].find((b) => b.textContent?.trim() === 'Throughput')!.click();
    flushSync();
    expect(model.lrView.get()).toBe('ips');
    expect(title()).toBe('Throughput');
    expect([...chart().querySelectorAll('.legend li .name')].map((n) => n.textContent?.trim())).toEqual(['img/s']);
    // Every step from the first, not the tile's last 120: the slow start is in it.
    const pts = model.ipsSeries.get()[0]!.points;
    expect(pts[0]!.y).toBe(20);
    expect(pts.length).toBe(300);
    expect(model.ips.get().n).toBe(120);
    c.unmount();
  });

  it("end each chart's first line in the run's mark, the loss and the learning rate too (design/UNIFIED.md §3, Marks)", () => {
    const first = (m: RunModel) => [m.mapSeries.get()[0]!, m.prSeries.get()[0]!, m.lossSeries.get().find((s) => s.id === 'total')!, m.lrSeries.get()[0]!, m.ipsSeries.get()[0]!];
    const live = show({ ...base, status: 'running' }, { steps: [step(50), step(100)], epochs: [epoch(1), epoch(2)] });
    expect(first(live.model).map((s) => s.end)).toEqual(['live', 'live', 'live', 'live', 'live']);
    // Every other series ends in a plain dot of its own hue.
    expect(live.model.lossSeries.get().filter((s) => s.id !== 'total').map((s) => s.end)).toEqual([undefined, undefined, undefined, undefined]);
    live.c.unmount();
    const stopped = show({ ...base, status: 'stopped' }, { steps: [step(50), step(100)], epochs: [epoch(1), epoch(2)] });
    expect(first(stopped.model).map((s) => s.end)).toEqual(['stopped', 'stopped', 'stopped', 'stopped', 'stopped']);
    stopped.c.unmount();
    const done = show({ ...base, status: 'completed' }, { steps: [step(50), step(100)], epochs: [epoch(1), epoch(2)] });
    expect(first(done.model).map((s) => s.end)).toEqual([undefined, undefined, undefined, undefined, undefined]);
    done.c.unmount();
  });

  it('share one cursor: arrowing through one reads every one', async () => {
    // happy-dom lays nothing out: every plot is 600 px wide from its first read.
    const proto = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth');
    Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
      configurable: true,
      get: () => 600,
    });
    try {
      const { c, model } = show(
        {
          ...base,
          status: 'completed',
          epoch: 3,
          epochs: 3,
          step: 150,
          steps_per_epoch: 50,
        },
        {
          steps: [step(50), step(100), step(150)],
          epochs: [epoch(1), epoch(2), epoch(3)],
        },
      );
      const svg = target.querySelector<SVGSVGElement>('#chart-map svg[aria-roledescription=chart]')!;
      svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
      flushSync();
      await tick();
      expect(model.cursor.get()).not.toBeNull();
      for (const id of ['chart-map', 'chart-loss', 'chart-pr', 'chart-lr']) expect(target.querySelector(`#${id} line.cursor`), id).not.toBeNull();
      c.unmount();
    } finally {
      if (proto) Object.defineProperty(HTMLElement.prototype, 'clientWidth', proto);
    }
  });
});
