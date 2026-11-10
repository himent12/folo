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

/** Mounts with a layout width: happy-dom has none, so every element reports `px` while the chart measures itself. */
function renderWide(px: number, props: Record<string, unknown> = {}) {
  const proto = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth');
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => px });
  try {
    render(props);
    window.dispatchEvent(new Event('resize'));
    target.querySelector('.hist')?.dispatchEvent(new Event('resize'));
    flushSync();
  } finally {
    if (proto) Object.defineProperty(HTMLElement.prototype, 'clientWidth', proto);
    else delete (HTMLElement.prototype as { clientWidth?: number }).clientWidth;
  }
}

const readout = () => target.querySelector('.readout')?.textContent?.replace(/\s+/g, ' ').trim();

describe('Histogram', () => {
  it('draws the bins as evenly spaced categories, each named, with bar height as the count', () => {
    renderWide(340, { binLabel: (lo: number, hi: number) => (hi >= 1 ? `≥${lo * 100}%` : `${lo * 100}–${hi * 100}%`) });
    const svg = target.querySelector<SVGElement>('svg[aria-roledescription=chart]')!;
    expect(svg).not.toBeNull();
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
    renderWide(400);
    const svg = target.querySelector<SVGElement>('svg[aria-roledescription=chart]')!;
    expect(readout()).toBe('100 boxes');
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
