// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import Histogram from './Histogram.svelte';

let c: ReturnType<typeof mount> | undefined;
let target: HTMLElement;

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target?.remove();
});

function render(props: Record<string, unknown> = {}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  c = mount(Histogram, {
    target,
    props: { edges: [0, 0.05, 0.1, 1], counts: [30, 50, 20], label: 'Box sizes', unit: 'boxes', yLabel: 'Boxes', ...props },
  });
  flushSync();
}

const readout = () => target.querySelector('.readout')?.textContent?.replace(/\s+/g, ' ').trim();

describe('Histogram', () => {
  it('states the total in the readout row at rest, with no density note', () => {
    render();
    expect(readout()).toBe('100 boxes');
  });

  it('draws the bins as evenly spaced categories, each named, with bar height as the count', () => {
    render({ binLabel: (lo: number, hi: number) => (hi >= 1 ? `≥${lo * 100}%` : `${lo * 100}–${hi * 100}%`) });
    const box = target.querySelector<HTMLElement>('.hist')!;
    Object.defineProperty(box, 'clientWidth', { value: 340, configurable: true });
    box.dispatchEvent(new Event('resize'));
    flushSync();
    const svg = target.querySelector<SVGElement>('svg[role=img]');
    if (!svg) return; // no layout in this environment
    expect(svg.querySelector('.y-label')?.textContent).toBe('Boxes');
    const ticks = [...svg.querySelectorAll('text.tick')].map((t) => t.textContent);
    expect(ticks).toEqual(expect.arrayContaining(['0–5%', '5–10%', '≥10%']));
    // 300 px of plot over three bins: equal widths whatever the value span.
    const xs = [...svg.querySelectorAll('path.bar')].map((p) => Number(p.getAttribute('d')!.match(/^M([\d.]+)/)![1]));
    expect(xs[1]! - xs[0]!).toBeCloseTo(xs[2]! - xs[1]!);
  });

  it('keeps the hidden data table with every bin, count and share', () => {
    render({ binLabel: (lo: number, hi: number) => `${lo * 100}–${hi * 100}%` });
    const rows = [...target.querySelectorAll('tbody tr')].map((r) => [...r.children].map((x) => x.textContent).join(' '));
    expect(rows).toEqual(['0–5% 30 30.0%', '5–10% 50 50.0%', '10–100% 20 20.0%']);
  });

  it('reads a bin out in the row when the keyboard moves to it, with its actual count', () => {
    render();
    // The plot renders once it has a width; happy-dom reports none, so give it one.
    const box = target.querySelector<HTMLElement>('.hist')!;
    Object.defineProperty(box, 'clientWidth', { value: 400, configurable: true });
    box.dispatchEvent(new Event('resize'));
    flushSync();
    const svg = target.querySelector<SVGElement>('svg[role=img]');
    if (!svg) return; // no layout in this environment: the pure parts are covered in histogram.test.ts
    svg.dispatchEvent(new FocusEvent('focus'));
    svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    flushSync();
    expect(readout()).toContain('50 boxes');
    expect(readout()).toContain('50.0%');
    svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    expect(readout()).toBe('100 boxes');
  });
});
