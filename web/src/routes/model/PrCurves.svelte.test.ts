// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { fromModelEval } from '$lib/api/evaluation';
import { MockApi } from '$lib/api/mock';
import PrCurves from './PrCurves.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;
let phone = false;

beforeEach(() => {
  window.matchMedia = ((q: string) => ({
    matches: phone && q.includes('max-width: 767px'),
    media: q,
    addEventListener() {},
    removeEventListener() {},
  })) as unknown as typeof window.matchMedia;
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

async function render(view: 'all' | 'curves' | 'classes', more: { onviewcurve?: (cls: number) => void; onback?: () => void } = {}) {
  const api = new MockApi('normal', 0);
  const model = await api.model('m_fast_n');
  const ev = fromModelEval(await api.modelEval('m_fast_n'));
  c = mount(PrCurves, { target, props: { classes: model.classes, ev, map50: model.metrics?.map50 ?? null, view, ...more } });
  flushSync();
}

describe('PrCurves on phones', () => {
  it('gives the class table one column definition: a col per visible column, header and rows alike', async () => {
    phone = true;
    await render('classes');
    const cols = target.querySelectorAll('table.table colgroup col');
    expect(cols).toHaveLength(4);
    expect(target.querySelector('table.table')?.classList.contains('fixed')).toBe(true);
    // Numbers right-aligned, headings too.
    const heads = [...target.querySelectorAll('table.table thead th:not(.hide-sm)')];
    expect(heads.map((h) => h.textContent)).toEqual(['Class', 'AP50', 'AP50-95', 'Instances']);
    expect(heads.slice(1).every((h) => h.classList.contains('r'))).toBe(true);
  });

  it('tells a phone how to trace a curve from the Curves section', async () => {
    phone = true;
    await render('curves');
    expect(target.querySelector('.legend .hint')?.textContent).toBe('Tap a class in Classes to trace its curve.');
  });

  it('keeps the automatic layout (no colgroup) on wide screens', async () => {
    phone = false;
    await render('all');
    expect(target.querySelector('table.table colgroup')).toBeNull();
  });

  it('keeps tracing (aria-pressed) and the details row (aria-expanded) on two separate controls', async () => {
    phone = true;
    await render('classes');
    const row = target.querySelector('table.table tbody tr')!;
    const trace = row.querySelector<HTMLButtonElement>('button.cls')!;
    const disc = row.querySelector<HTMLButtonElement>('button.disc')!;
    expect(trace.getAttribute('aria-label')).toMatch(/^Trace /);
    expect(trace.getAttribute('aria-pressed')).toBe('false');
    expect(trace.hasAttribute('aria-expanded')).toBe(false);
    expect(disc.getAttribute('aria-expanded')).toBe('false');
    expect(disc.hasAttribute('aria-pressed')).toBe(false);

    disc.click();
    flushSync();
    expect(disc.getAttribute('aria-expanded')).toBe('true');
    expect(trace.getAttribute('aria-pressed')).toBe('false');
    expect(target.querySelector(`#${disc.getAttribute('aria-controls')}`)).not.toBeNull();

    trace.click();
    flushSync();
    expect(trace.getAttribute('aria-pressed')).toBe('true');
    expect(disc.getAttribute('aria-expanded')).toBe('true');
    disc.click();
    flushSync();
    expect(disc.getAttribute('aria-expanded')).toBe('false');
    expect(trace.getAttribute('aria-pressed')).toBe('true');
  });

  it("shows a traced class's curve from the selection bar alone, and offers the way back in the curves' legend", async () => {
    phone = true;
    const viewed: number[] = [];
    await render('classes', { onviewcurve: (cls) => viewed.push(cls) });
    const disc = target.querySelector<HTMLButtonElement>('table.table tbody tr button.disc')!;
    const cls = Number(disc.dataset.cls);
    target.querySelector<HTMLButtonElement>(`button.cls[data-cls="${cls}"]`)!.click();
    disc.click();
    flushSync();
    // One action in one place: the opened details row does not repeat the selection bar's View curve.
    expect([...target.querySelectorAll('tr.detail button')].some((b) => b.textContent?.includes('View curve'))).toBe(false);
    const view = target.querySelector<HTMLButtonElement>('.tools .to-curve')!;
    expect(view.textContent?.trim()).toBe('View curve');
    view.click();
    flushSync();
    expect(viewed).toEqual([cls]);
    expect(target.querySelector<HTMLButtonElement>(`button.cls[data-cls="${cls}"]`)!.getAttribute('aria-pressed')).toBe('true');
    unmount(c!);
    let back = 0;
    await render('curves', { onback: () => back++ });
    const b = target.querySelector<HTMLButtonElement>('.legend button.back')!;
    expect(b.textContent?.trim()).toBe('Back to Classes');
    b.click();
    expect(back).toBe(1);
  });
});

describe('PrCurves: the operating point (docs/API.md, best_f1_conf)', () => {
  it('names the threshold the evaluation read precision and recall at, under the table and in a class’s details', async () => {
    phone = true;
    await render('classes');
    const conf = (await new MockApi('normal', 0).modelEval('m_fast_n')).best_f1_conf!;
    const at = `at conf ${conf.toFixed(2)} (best F1)`;
    expect(target.querySelector('.foot')?.textContent).toContain(`Precision and recall ${at};`);
    target.querySelector<HTMLButtonElement>('table.table tbody tr button.disc')!.click();
    flushSync();
    expect(target.querySelector('tr.detail .ctx')?.textContent).toBe(`${at}, IoU 0.50`);
  });

  it('keeps the plain wording when the evaluation does not say (older servers)', async () => {
    phone = false;
    const api = new MockApi('normal', 0);
    const ev = fromModelEval({ ...(await api.modelEval('m_fast_n')), best_f1_conf: null });
    expect(ev.bestF1Conf).toBeNull();
    const model = await api.model('m_fast_n');
    c = mount(PrCurves, { target, props: { classes: model.classes, ev, map50: null, view: 'all' } });
    flushSync();
    expect(target.querySelector('.foot')?.textContent).toContain('Precision and recall at the confidence that maximises mean F1;');
  });
});
