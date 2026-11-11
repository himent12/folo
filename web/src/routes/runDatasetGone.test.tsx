// @vitest-environment happy-dom
// A run whose dataset was removed (datasets F2): its page still names the dataset and its classes as recorded with
// the run (Run.dataset), without a link to a page that is gone, and its sample tiles say why there is no image.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Run from './Run';

let target: HTMLElement;
let c: Rendered | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
});

describe('a run whose dataset was removed', () => {
  it('keeps the recorded name and classes, links nowhere, and its samples say the dataset was removed', async () => {
    const api = new MockApi('normal', 0);
    const runs = await api.runs();
    const done = runs.find((r) => r.status === 'completed' && r.config.dataset_id === 'ds_shapes')!;
    expect(done.dataset).toEqual({ name: 'shapes', classes: ['circle', 'square', 'triangle'] });
    await api.deleteDataset('ds_shapes');
    hub.api = api;
    hub.mode = 'mock';
    router.route = parse(`#/runs/${done.id}`);
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <Run />, target);
    await vi.waitFor(
      () => {
        flushSync();
        if (!target.querySelector('#samples .gone') || !target.querySelector('#per-class tbody tr')) throw new Error('not loaded');
      },
      { timeout: 4000, interval: 10 },
    );
    expect(target.querySelector('.run-meta')?.textContent).toContain('shapes (removed)');
    expect(target.querySelector('.run-meta a[href^="#/datasets/"]')).toBeNull();
    expect(target.querySelector('#samples .gone')?.textContent).toContain('The dataset was removed');
    expect([...target.querySelectorAll('#samples button')].some((b) => b.textContent?.trim() === 'Retry')).toBe(false);
    const names = [...target.querySelectorAll('#per-class tbody tr')].map((r) => r.querySelector('td')?.textContent?.trim());
    expect(names.sort()).toEqual(['circle', 'square', 'triangle']);
  });
});
