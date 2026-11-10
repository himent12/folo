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

async function render(view: 'all' | 'curves' | 'classes', more: { onviewcurve?: (cls: number) => void; onback?: () => void; selected?: number[] } = {}) {
  const api = new MockApi('normal', 0);
  const model = await api.model('m_fast_n');
  const ev = fromModelEval(await api.modelEval('m_fast_n'));
  c = mount(PrCurves, { target, props: { classes: model.classes, ev, map50: model.metrics?.map50 ?? null, view, ...more } });
  flushSync();
}

describe('PrCurves on phones', () => {
  it('traces classes from the Curves section itself: chips beside the graph, traced first, each a tap to remove', async () => {
    phone = true;
    await render('curves');
    const trace = target.querySelector<HTMLButtonElement>('.tools.bar .trace')!;
    const tracer = target.querySelector<HTMLElement>(`#${trace.getAttribute('aria-controls')}`)!;
    expect(tracer.hidden).toBe(true);
    trace.click();
    flushSync();
    expect(trace.getAttribute('aria-expanded')).toBe('true');
    expect(tracer.hidden).toBe(false);
    const chips = () => [...tracer.querySelectorAll<HTMLButtonElement>('.t-chip')];
    const second = chips()[1]!.textContent;
    chips()[1]!.click();
    flushSync();
    // Traced: first in the list, pressed, named for removal, and in the legend over the graph.
    expect(chips()[0]!.textContent).toBe(second);
    expect(chips()[0]!.getAttribute('aria-pressed')).toBe('true');
    expect(chips()[0]!.getAttribute('aria-label')).toBe(`Stop tracing ${second}`);
    expect([...target.querySelectorAll('.legend .item')].some((i) => i.textContent?.startsWith(`${second} `))).toBe(true);
    chips()[0]!.click();
    flushSync();
    expect(chips().some((b) => b.getAttribute('aria-pressed') === 'true')).toBe(false);
  });

  it('keeps the legend a readout: navigation and Clear sit in the selection bar, never among the values', async () => {
    phone = true;
    await render('curves', { onback: () => {}, selected: [0] });
    expect(target.querySelectorAll('.legend button, .legend .hint')).toHaveLength(0);
    expect(target.querySelector('.tools.bar button.back')).not.toBeNull();
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

  it("shows a traced class's curve from the selection bar alone, and offers the way back in the curves' selection bar", async () => {
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
    view.click();
    flushSync();
    expect(viewed).toEqual([cls]);
    expect(target.querySelector<HTMLButtonElement>(`button.cls[data-cls="${cls}"]`)!.getAttribute('aria-pressed')).toBe('true');
    unmount(c!);
    let back = 0;
    await render('curves', { onback: () => back++ });
    target.querySelector<HTMLButtonElement>('.tools.bar button.back')!.click();
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
});
