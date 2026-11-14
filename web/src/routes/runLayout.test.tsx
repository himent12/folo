// @vitest-environment happy-dom
// A run page's layout (design/UNIFIED.md §4 Run): no completed banner (Open model is the header's primary, Try in
// Playground beside Compare), the status badge on the title's line, the run's id in Config's header, the four charts in
// two rows, and on phones Precision / Recall and Learning rate as sections of the picker.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Run as RunT } from '$lib/api/types';
import { parse, router } from '$lib/router';
import Run from './Run';

let target: HTMLElement;
let c: Rendered | undefined;
let real: typeof window.matchMedia;

beforeEach(() => {
  real = window.matchMedia;
});
afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  window.matchMedia = real;
});

/** Opens a run's page at a width (`phone`: under 768 px), the run changed by `change`. */
async function open(pick: (rs: RunT[]) => RunT, change: Partial<RunT> = {}, phone = false, nothing = false) {
  window.matchMedia = ((q: string) => ({ matches: phone && q.includes('max-width: 767px') ? true : phone && q.includes('max-width: 1023px'), media: q, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
  const api = new MockApi('normal', 0);
  const run = { ...pick(await api.runs()), ...change };
  const get = api.run.bind(api);
  api.run = (id, ...rest) => (id === run.id ? Promise.resolve(run) : get(id, ...rest));
  // A model that detected nothing: every epoch scored 0 (the page takes the best from the epochs it loads).
  const metrics = api.runMetrics.bind(api);
  if (nothing)
    api.runMetrics = async (id, ...rest) => {
      const m = await metrics(id, ...rest);
      return { ...m, epochs: m.epochs.map((e) => ({ ...e, val: { ...e.val, map50_95: 0 } })) };
    };
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse(`#/runs/${run.id}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <Run />, target);
  await vi.waitFor(
    () => {
      flushSync();
      if (!target.querySelector('#chart-map .legend')) throw new Error('not loaded');
    },
    { timeout: 4000, interval: 10 },
  );
  return run;
}

const headerButtons = () => [...target.querySelectorAll('header .actions a, header .actions button')].map((b) => b.textContent?.trim()).filter(Boolean);

describe('a completed run', () => {
  it('has no completed banner: Open model is the primary, Try in Playground beside Compare, the status on the title line', async () => {
    const run = await open((rs) => rs.find((r) => r.status === 'completed' && r.model_id && (r.best?.map50_95 ?? 0) > 0)!);
    expect(target.textContent).not.toMatch(/Completed in/);
    const primary = target.querySelector<HTMLAnchorElement>(`header a[href="#/models/${run.model_id}"]`);
    expect(primary?.textContent?.trim()).toBe('Open model');
    expect(primary?.className).toContain('primary');
    expect(headerButtons()).toContain('Try in Playground');
    expect(headerButtons()).toContain('Compare');
    expect(target.querySelector('.title-line h1')?.textContent?.trim()).toBe(run.name);
    expect(target.querySelector('.title-line .title-badge')?.textContent).toContain('Completed');
    // Its id moved from the header to Config's.
    expect(target.querySelector('#config .run-id')?.textContent).toBe(run.id);
    // The four charts, in two rows.
    expect([...target.querySelectorAll('.run-chart')].map((p) => p.id)).toEqual(['chart-map', 'chart-loss', 'chart-pr', 'chart-lr']);
  });

  it('leaves Try in Playground out when the model detected nothing', async () => {
    await open((rs) => rs.find((r) => r.status === 'completed' && r.model_id)!, { best: { epoch: 1, map50_95: 0 } }, false, true);
    expect(headerButtons()).toContain('Open model');
    expect(headerButtons()).not.toContain('Try in Playground');
    expect(target.textContent).not.toContain('Try in Playground');
  });
});

describe('a run on a phone', () => {
  it('puts Precision / Recall and Learning rate in the section picker, under the KPI tiles, mAP and loss', async () => {
    await open((rs) => rs.find((r) => r.status === 'completed')!, {}, true);
    const tabs = [...target.querySelectorAll('.phone-pick [role=tab]')].map((t) => t.textContent?.trim());
    expect(tabs.slice(0, 4)).toEqual(['P / R', 'LR', 'Samples', 'Classes']);
    expect(tabs).toContain('Logs');
    expect(tabs).toContain('Config');
    // The row of four: mAP and loss stay in the metrics, P/R and LR are parts of the picked section.
    expect([...target.querySelectorAll('.metrics .run-chart')].map((p) => p.id)).toEqual(['chart-map', 'chart-loss']);
    expect(target.querySelector('[data-part="pr"] #chart-pr')).not.toBeNull();
    expect(target.querySelector('[data-part="lr"] #chart-lr')).not.toBeNull();
    // Samples is the section it opens on.
    expect(target.querySelector('.detail')?.getAttribute('data-show')).toBe('samples');
    [...target.querySelectorAll<HTMLElement>('.phone-pick [role=tab]')].find((t) => t.textContent?.trim() === 'P / R')!.click();
    flushSync();
    expect(target.querySelector('.detail')?.getAttribute('data-show')).toBe('pr');
  });
});
