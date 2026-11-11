// @vitest-environment happy-dom
// A run's chart on a phone is open or folded by one remembered state (round 26: open on Samples, folded on Classes and
// Config): every section, and every mount of the chart this session, reads the same `chartFold`.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../../framework/testing';
import { AppendLog } from '$lib/api/appendLog';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { MockApi } from '$lib/api/mock';
import type { Epoch, Run, Step } from '$lib/api/types';
import RunMetrics, { setChartOpen } from './RunMetrics';

const unmount = (c: Rendered) => c.unmount();

let target: HTMLElement;
let real: typeof window.matchMedia;
let run: Run;

beforeEach(async () => {
  real = window.matchMedia;
  window.matchMedia = ((q: string) => ({ matches: q.includes('max-width: 767px'), media: q, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
  target = document.createElement('div');
  document.body.appendChild(target);
  run = await new MockApi('normal', 0).run('run_7d1e');
  setChartOpen(true);
});
afterEach(() => {
  window.matchMedia = real;
  target.remove();
});

// The steps and epochs are the run's append logs (`live.stepsLog`, `live.epochsLog`), empty here.
const show = () => render(() => <RunMetrics run={run} steps={new AppendLog<Step>()} epochs={new AppendLog<Epoch>()} version={0} loading={false} />, target);
const readout = () => target.querySelector<HTMLButtonElement>('button.readout');
const hide = () => [...target.querySelectorAll<HTMLButtonElement>('header button')].find((b) => b.textContent?.trim() === 'Hide chart');

describe('the run chart on phones', () => {
  it('opens and folds by one remembered state, the same wherever it is shown again', () => {
    let c = show();
    flushSync();
    // Open: the header's "Hide chart" folds it; there is no readout row above the picker.
    expect(readout()).toBeNull();
    expect(hide()?.getAttribute('aria-expanded')).toBe('true');
    expect(target.querySelector('#run-chart-body')).not.toBeNull();
    hide()!.click();
    flushSync();
    expect(readout()?.getAttribute('aria-expanded')).toBe('false');
    expect(target.querySelector('#run-chart-body')).toBeNull();
    // Another section, another mount: still folded.
    unmount(c);
    c = show();
    flushSync();
    expect(readout()?.getAttribute('aria-expanded')).toBe('false');
    expect(sessionStorage.getItem('folo.run.chartOpen')).toBe('0');
    readout()!.click();
    flushSync();
    expect(readout()).toBeNull();
    expect(sessionStorage.getItem('folo.run.chartOpen')).toBe('1');
    unmount(c);
  });
});

describe('the run chart when the history failed to load', () => {
  it('says so with Retry, never "Waiting for the first validation…"', () => {
    let retried = 0;
    const c = render(
      () => (
        <RunMetrics run={run} steps={new AppendLog<Step>()} epochs={new AppendLog<Epoch>()} version={0} loading={false} error={new Error('HTTP 500')} retry={() => retried++} />
      ),
      target,
    );
    flushSync();
    expect(target.textContent).toContain("Could not load this run's metrics");
    expect(target.textContent).not.toMatch(/Waiting for the first|after the first epoch/);
    [...target.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Retry')!.click();
    expect(retried).toBe(1);
    unmount(c);
  });
});

describe('the run chart before anything is detected', () => {
  it('says "no detections yet" instead of "best 0.000 · ep 1" and "conf 1.00"', () => {
    window.matchMedia = ((q: string) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
    const val = { map50: 0, map50_95: 0, precision: 0, recall: 0, best_f1_conf: 1, per_class: [] };
    const epochs = new AppendLog<Epoch>();
    epochs.push({ epoch: 1, train_loss: { box: 1, cls: 1, dfl: 1, total: 3 }, val, duration_s: 10 });
    const live = { ...run, status: 'running' as const, best: { epoch: 1, map50_95: 0 } };
    const c = render(() => <RunMetrics run={live} steps={new AppendLog<Step>()} epochs={epochs} version={1} loading={false} />, target);
    flushSync();
    const caps = [...target.querySelectorAll('.tab .cap .full')].map((e) => e.textContent?.trim());
    expect(caps[0]).toBe('no detections yet');
    expect(caps[2]).toBe('no detections yet');
    expect(target.textContent).not.toMatch(/best 0\.000|conf 1\.00/);
    unmount(c);
  });
});
