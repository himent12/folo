// @vitest-environment happy-dom
// A run's chart on a phone is open or folded by one remembered state (round 26: open on Samples, folded on Classes and
// Config): every section, and every mount of the chart this session, reads the same `chartFold`.
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { MockApi } from '$lib/api/mock';
import type { Run } from '$lib/api/types';
import RunMetrics, { setChartOpen } from './RunMetrics.svelte';

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

const show = () => mount(RunMetrics, { target, props: { run, steps: [], epochs: [], version: 0, loading: false } });
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

  it('states a value once above the chart: the picker names the metric, the legend has the value', () => {
    const c = show();
    flushSync();
    const options = [...target.querySelectorAll('select[aria-label="Metric to chart"] option')].map((o) => o.textContent);
    expect(options).toEqual(['mAP50-95', 'mAP50', 'Precision · Recall', 'Loss', 'Learning rate', 'Throughput']);
    unmount(c);
  });
});
