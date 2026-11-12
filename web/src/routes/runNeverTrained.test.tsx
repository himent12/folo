// @vitest-environment happy-dom
// A run that ended without training (v-shell N7, v-data-runs N3): one plain line where the metrics would be, no
// "Waiting for the first validation…", no tiles "after the first epoch", no samples or per-class panels to wait for;
// a data error before the first step never says "during training".
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Run as RunT } from '$lib/api/types';
import { parse, router } from '$lib/router';
import Run from './Run';

let target: HTMLElement;
let c: Rendered | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
});

async function open(change: Partial<RunT>) {
  const api = new MockApi('normal', 0);
  const done = (await api.runs()).find((r) => r.status === 'completed')!;
  const run: RunT = { ...done, epoch: 0, step: 0, best: null, model_id: null, ...change };
  api.run = async () => run;
  api.runMetrics = async () => ({ steps: [], epochs: [] });
  api.runLogs = async () => ({ lines: [{ t: '2026-11-11T10:00:00Z', level: 'error', message: 'dataset folder not found' }], total: 1 });
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse(`#/runs/${run.id}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <Run />, target);
  await vi.waitFor(
    () => {
      flushSync();
      if (!target.querySelector('.metrics .never')) throw new Error('not loaded');
    },
    { timeout: 4000, interval: 10 },
  );
  return run;
}

describe('a run that ended without training', () => {
  it('failed before its first step: one line for the metrics, no panels that wait, and no "during training"', async () => {
    const run = await open({ status: 'failed', error_code: 'data_error', error: "dataset folder 'D:/gone' not found" });
    const text = target.textContent ?? '';
    expect(target.querySelector('.metrics')?.textContent?.trim()).toBe('No metrics: it ended before its first training step.');
    expect(text).not.toContain('Waiting for the first');
    expect(text).not.toContain('after the first epoch');
    expect(target.querySelector('.metrics')?.textContent).not.toMatch(/mosaic/i);
    expect(target.querySelector('#samples')).toBeNull();
    expect(target.querySelector('#per-class')).toBeNull();
    expect(target.querySelector('#logs')).not.toBeNull();
    expect(text).toContain('Failed before training started');
    expect(text).toContain(`The files of dataset ${run.dataset?.name ?? 'this run'} are missing or could not be read.`);
    expect(text).not.toContain('during training');
  });

  it('interrupted before it started: the same', async () => {
    await open({ status: 'stopped', error_code: 'interrupted', error: null, started: null, finished: '2026-11-11T10:00:00Z' });
    const text = target.textContent ?? '';
    expect(text).toContain('Interrupted: the server stopped before it started');
    expect(text).not.toContain('Waiting for the first');
    expect(target.querySelector('#samples')).toBeNull();
  });
});
