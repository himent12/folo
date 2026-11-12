// @vitest-environment happy-dom
// A run whose metrics fail to load (v-data-runs N9, v-shell N4): one error with one Retry, on the metrics panel; the
// panels derived from the metrics wait for it quietly, and the log, a request of its own, shows its lines.
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

describe('a run whose metrics fail to load', () => {
  it('says so once, with one Retry, and still shows its log', async () => {
    const api = new MockApi('normal', 0);
    const done = (await api.runs()).find((r) => r.status === 'completed')!;
    let fail = true;
    const metrics = api.runMetrics.bind(api);
    api.runMetrics = (id, ...rest) => (fail ? Promise.reject(new Error('Boom.')) : metrics(id, ...rest));
    hub.api = api;
    hub.mode = 'mock';
    router.route = parse(`#/runs/${done.id}`);
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <Run />, target);
    const retries = () => [...target.querySelectorAll('button')].filter((b) => b.textContent?.trim() === 'Retry');
    await vi.waitFor(
      () => {
        flushSync();
        if (!retries().length || !target.querySelector('#logs .line')) throw new Error('not loaded');
      },
      { timeout: 4000, interval: 10 },
    );
    expect(retries()).toHaveLength(1);
    expect(target.textContent?.match(/Boom\./g)).toHaveLength(1);
    expect(target.textContent).not.toContain('Could not load the log');
    expect(target.querySelector('#samples')?.textContent).toContain("Samples show once the run's metrics load.");
    expect(target.querySelector('#per-class')?.textContent).toContain("Per-class results show once the run's metrics load.");

    fail = false;
    retries()[0]!.click();
    await vi.waitFor(
      () => {
        flushSync();
        if (!target.querySelector('#per-class tbody tr')) throw new Error('not reloaded');
      },
      { timeout: 4000, interval: 10 },
    );
    expect(retries()).toHaveLength(0);
  });
});
