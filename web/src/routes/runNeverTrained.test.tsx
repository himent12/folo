// @vitest-environment happy-dom
// A run that ended without training (v-shell N7, v-data-runs N3): one plain line where the metrics would be, no
// "Waiting for the first validation…", no tiles "after the first epoch", no samples or per-class panels to wait for;
// a data error before the first step never says "during training".
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { ApiError } from '$lib/api/client';
import { MockApi } from '$lib/api/mock';
import type { Run as RunT, Step } from '$lib/api/types';
import { parse, router } from '$lib/router';
import Run from './Run';

let target: HTMLElement;
let c: Rendered | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
});

async function open(
  change: Partial<RunT>,
  steps: Step[] = [],
  loaded = (t: HTMLElement) => !!t.querySelector('.metrics .never'),
  prep: (api: MockApi) => void = () => {},
) {
  const api = new MockApi('normal', 0);
  const done = (await api.runs()).find((r) => r.status === 'completed')!;
  const run: RunT = { ...done, epoch: 0, step: 0, best: null, model_id: null, ...change };
  api.run = async () => run;
  api.runMetrics = async () => ({ steps, epochs: [] });
  api.runLogs = async () => ({ lines: [{ t: '2026-11-11T10:00:00Z', level: 'error', message: 'dataset folder not found' }], total: 1 });
  prep(api);
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse(`#/runs/${run.id}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <Run />, target);
  await vi.waitFor(
    () => {
      flushSync();
      if (!loaded(target)) throw new Error('not loaded');
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

  it('stopped during its first epoch: the readings that will never come say so, nothing waits for a validation', async () => {
    const step: Step = { epoch: 1, step: 1, lr: 0, loss: { box: 4, cls: 6, dfl: 2.5, total: 12.5 }, imgs_per_s: 2, eta_s: 0, steps_per_epoch: 100, epochs: 10 };
    await open({ status: 'stopped', error: null, epoch: 0, step: 1, steps_per_epoch: 100, epochs: 10 }, [step], (t) => (t.querySelector('.metrics')?.textContent ?? '').includes('12.5'));
    const text = target.textContent ?? '';
    expect(text).toContain('Stopped during its first epoch');
    expect(text).not.toContain('Waiting for the first');
    expect(text).not.toContain('after the first epoch');
    expect(text).not.toContain('appear after the first validation');
    expect(text).toContain('not validated');
    expect(text).toContain('No samples: it ended before its first validation.');
  });
});

// #335 and its reviews: Check dataset is withdrawn once the datasets list says the run's dataset was removed (its page
// is "Dataset not found"); while that is not known (the list is delayed, or cannot be read) it is offered as on main.
describe("a failed run's Check dataset", () => {
  const failed = { status: 'failed', error_code: 'data_error', error: "dataset folder 'D:/gone' not found" } as const;
  const check = () => [...target.querySelectorAll<HTMLElement>('.c-acts a')].find((a) => a.textContent?.trim() === 'Check dataset');
  const shown = () => !!target.querySelector('.c-acts');

  it('is offered while the datasets list is delayed, and stays for a registered dataset', async () => {
    let release!: () => void;
    const held = new Promise<void>((r) => (release = r));
    let reads = 0;
    const run = await open(failed, [], shown, (api) => {
      const datasets = api.datasets.bind(api);
      api.datasets = async () => (reads++, await held, datasets());
    });
    expect(check()!.getAttribute('href')).toBe(`#/datasets/${run.config.dataset_id}`);
    release();
    await vi.waitFor(() => {
      if (reads < 1) throw new Error('not read');
    });
    await held;
    flushSync();
    expect(check()!.getAttribute('href')).toBe(`#/datasets/${run.config.dataset_id}`);
  });

  it('is withdrawn once the list says the dataset was removed', async () => {
    await open(failed, [], shown, (api) => {
      api.datasets = async () => [];
    });
    await vi.waitFor(() => {
      flushSync();
      if (check()) throw new Error('still offered');
    });
  });

  it('is offered when the datasets cannot be read', async () => {
    let reads = 0;
    await open(failed, [], shown, (api) => {
      api.datasets = async () => {
        reads++;
        throw new ApiError(503, 'unavailable', 'The server is busy.');
      };
    });
    await vi.waitFor(() => {
      if (reads < 1) throw new Error('not read');
    });
    flushSync();
    expect(check()).toBeDefined();
  });
});
