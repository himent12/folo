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

async function render(view: 'all' | 'curves' | 'classes') {
  const api = new MockApi('normal', 0);
  const model = await api.model('m_fast_n');
  const ev = fromModelEval(await api.modelEval('m_fast_n'));
  c = mount(PrCurves, { target, props: { classes: model.classes, ev, map50: model.metrics?.map50 ?? null, view } });
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
});
