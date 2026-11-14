// @vitest-environment happy-dom
// A run page's layout (design/UNIFIED.md §4 Run): no completed banner (Open model is the header's primary, Try in
// Playground beside Compare), the status badge on the title's line, the run's id in Config's header, the four charts in
// two rows, on a tablet the sections stacked, and on phones Precision / Recall and Learning rate as sections of the
// picker; a queued run's Config has the id too.
/// <reference types="node" />
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
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

type Tier = 'wide' | 'tablet' | 'phone';

/** Opens a run's page at a tier (a tablet: 768–1023 px; a phone: under 768), the run changed by `change`. */
async function open(pick: (rs: RunT[]) => RunT, { change = {}, tier = 'wide', nothing = false, ready = '#chart-map .legend' }: { change?: Partial<RunT>; tier?: Tier; nothing?: boolean; ready?: string } = {}) {
  const narrow = (q: string) => (q.includes('max-width: 767px') ? tier === 'phone' : q.includes('max-width: 1023px') ? tier !== 'wide' : false);
  window.matchMedia = ((q: string) => ({ matches: narrow(q), media: q, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
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
      if (!target.querySelector(ready)) throw new Error('not loaded');
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
    expect(target.querySelector('.title-line .badge-slot')?.textContent).toContain('Completed');
    // Its id moved from the header to Config's.
    expect(target.querySelector('#config .run-id')?.textContent).toBe(run.id);
    // The four charts, in two rows.
    expect([...target.querySelectorAll('.run-chart')].map((p) => p.id)).toEqual(['chart-map', 'chart-loss', 'chart-pr', 'chart-lr']);
  });

  it('moves Try in Playground to the ⋯ when the model detected nothing', async () => {
    await open((rs) => rs.find((r) => r.status === 'completed' && r.model_id)!, { change: { best: { epoch: 1, map50_95: 0 } }, nothing: true });
    expect(headerButtons()).toContain('Open model');
    expect(headerButtons()).not.toContain('Try in Playground');
    // Still one click away: the header leaves it out, the menu keeps it.
    target.querySelector<HTMLButtonElement>('header button[aria-label="More run actions"]')!.click();
    flushSync();
    expect([...document.querySelectorAll('[role="menuitem"]')].map((e) => e.textContent?.trim())).toContain('Try in Playground');
  });
});

describe('a run on a phone', () => {
  it('puts Precision / Recall and Learning rate in the section picker, under the KPI tiles, mAP and loss', async () => {
    await open((rs) => rs.find((r) => r.status === 'completed')!, { tier: 'phone' });
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

describe('a run on a tablet', () => {
  it('stacks its sections rather than show one at a time (design/UNIFIED.md §4 Run: rows 2 to 5 stack below 1024 px)', async () => {
    await open((rs) => rs.find((r) => r.status === 'completed')!, { tier: 'tablet' });
    // The picker is phones' only: here the detail is no tab panel, and every section is in it.
    const detail = target.querySelector('.detail')!;
    expect(detail.getAttribute('role')).toBeNull();
    expect(detail.getAttribute('aria-labelledby')).toBeNull();
    expect([...detail.querySelectorAll('[data-part]')].map((p) => p.getAttribute('data-part'))).toEqual(['samples', 'classes', 'logs', 'config']);
    // All four charts stay in the metrics (stacked by its CSS), none in the sections.
    expect([...target.querySelectorAll('.metrics .run-chart')].map((p) => p.id)).toEqual(['chart-map', 'chart-loss', 'chart-pr', 'chart-lr']);
  });
});

describe("a run's sections, in the order they are seen", () => {
  // Reading order at every tier: from 1024 px two to a row (Samples | Per class, Logs | Cloud over Config), on a tablet
  // stacked, on a phone the picker's order. Tab follows the DOM, so the DOM is that order.
  const READ = ['samples', 'classes', 'logs', 'cloud', 'config'];
  const parts = () => [...target.querySelectorAll('.detail > [data-part]')].map((p) => p.getAttribute('data-part'));

  it('puts them in the DOM in reading order, which the CSS keeps (no `order`, no explicit grid placement)', () => {
    const src = readFileSync(resolve(process.cwd(), 'src/routes/Run.tsx'), 'utf8');
    const css = src.slice(src.lastIndexOf('css`'));
    for (const rule of css.matchAll(/\[data-part='(\w+)'\]\s*\{([^}]*)\}/g)) {
      expect(rule[2], `[data-part=${rule[1]}]`).not.toMatch(/(^|[\s;])order\s*:|grid-(area|column|row-start)\s*:/);
    }
    // From 1024 px two columns filled row by row: what the reading order above assumes.
    expect(css).toMatch(/@media \(min-width: 1024px\) \{\s*\.detail \{\s*grid-template-columns: minmax\(0, 2fr\) minmax\(0, 1fr\);/);
    expect(css).not.toMatch(/grid-auto-flow/);
  });

  for (const tier of ['wide', 'tablet', 'phone'] as const) {
    it(`follows it on a ${tier === 'wide' ? 'laptop or desktop' : tier}, for a local and a cloud run`, async () => {
      for (const cloud of [false, true]) {
        await open((rs) => rs.find((r) => (cloud ? !!r.cloud && r.status === 'completed' : !r.cloud && r.status === 'completed'))!, { tier });
        await vi.waitFor(() => {
          flushSync();
          if (cloud && !target.querySelector('[data-part="cloud"]')) throw new Error('no cloud panel yet');
        });
        const seen = parts();
        const want = [...(tier === 'phone' ? ['pr', 'lr'] : []), ...READ.filter((p) => cloud || p !== 'cloud')];
        expect(seen, `${cloud ? 'cloud' : 'local'} run`).toEqual(want);
        if (tier === 'phone') {
          // The picker lists the sections in the same order.
          expect([...target.querySelectorAll('.phone-pick [role=tab]')].map((t) => t.id.replace('run-sec-', ''))).toEqual(seen);
        }
        // Per class and Config stay named regions of their own (what the old rail's label grouped).
        for (const id of ['per-class', 'config']) {
          const region = target.querySelector(`.detail section#${id}`)!;
          expect(document.getElementById(region.getAttribute('aria-labelledby')!)?.textContent).toBeTruthy();
        }
        c?.unmount();
        c = undefined;
        target.remove();
      }
    });
  }
});

describe('a queued run', () => {
  it("has the run's id in its Config's header", async () => {
    const run = await open((rs) => rs.find((r) => r.status === 'queued' && !r.cloud)!, { ready: '.waiting #config' });
    expect(target.querySelector('.waiting #config .run-id')?.textContent).toBe(run.id);
  });
});
