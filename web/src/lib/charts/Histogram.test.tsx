// @vitest-environment happy-dom
import { flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it } from 'vitest';
import { render as mountFw, type Rendered } from '../../../framework/testing';
import Histogram, { type HistogramProps } from './Histogram.tsx';

let c: Rendered | undefined;
let target: HTMLElement;

afterEach(() => {
  if (c) c.unmount();
  c = undefined;
  target?.remove();
});

function render(props: Record<string, unknown> = {}) {
  target = document.createElement('div');
  document.body.appendChild(target);
  const all = { edges: [0, 0.05, 0.1, 1], counts: [30, 50, 20], label: 'Box sizes', unit: 'boxes', yLabel: 'Boxes', ...props } as HistogramProps;
  c = mountFw(() => <Histogram {...all} />, target);
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

  it('consumes Escape only while active, defers to open popovers, steps 5 bins on Shift+Arrow, and stays idle when shrunk to empty', () => {
    const counts = signal([0, 30, 50, 20, 10, 5, 15]);
    const edges = signal([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1]);
    const outOfBoundsLabels: number[] = [];
    const proto = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth');
    Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 400 });
    try {
      target = document.createElement('div');
      document.body.appendChild(target);
      c = mountFw(
        () => (
          <Histogram
            edges={edges.get()}
            counts={counts.get()}
            binLabel={(lo, hi, i) => {
              if (i >= counts.peek().length) outOfBoundsLabels.push(i);
              return `${lo}–${hi}`;
            }}
            label="Box sizes"
            unit="boxes"
          />
        ),
        target,
      );
      window.dispatchEvent(new Event('resize'));
      target.querySelector('.hist')?.dispatchEvent(new Event('resize'));
      flushSync();
    } finally {
      if (proto) Object.defineProperty(HTMLElement.prototype, 'clientWidth', proto);
      else delete (HTMLElement.prototype as { clientWidth?: number }).clientWidth;
    }

    const svg = target.querySelector<SVGElement>('svg[aria-roledescription=chart]')!;
    // Idle Escape does not preventDefault.
    const idleEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    svg.dispatchEvent(idleEsc);
    flushSync();
    expect(idleEsc.defaultPrevented).toBe(false);

    // Modifier-held arrow is ignored.
    const altLeft = new KeyboardEvent('keydown', { key: 'ArrowLeft', altKey: true, bubbles: true, cancelable: true });
    svg.dispatchEvent(altLeft);
    flushSync();
    expect(altLeft.defaultPrevented).toBe(false);
    expect(readout()).toBe('130 boxes');

    // ArrowRight from idle lands on the first non-empty bin (30 boxes), then Shift+ArrowRight jumps 5 bins to bin 6 (15 boxes).
    svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
    flushSync();
    expect(readout()).toContain('30 boxes');
    svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', shiftKey: true, bubbles: true, cancelable: true }));
    flushSync();
    expect(readout()).toContain('15 boxes');

    // Open info popover takes Escape priority over dismissing the focused histogram readout.
    const pop = document.createElement('div');
    pop.className = 'info-pop open';
    document.body.appendChild(pop);
    const blockedEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    svg.dispatchEvent(blockedEsc);
    flushSync();
    expect(blockedEsc.defaultPrevented).toBe(false);
    expect(readout()).toContain('15 boxes');
    pop.remove();

    // Escape while active consumes the event.
    const activeEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    svg.dispatchEvent(activeEsc);
    flushSync();
    expect(activeEsc.defaultPrevented).toBe(true);
    expect(readout()).toBe('130 boxes');

    // ArrowLeft from idle lands on the last bin (15 boxes).
    svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true, cancelable: true }));
    flushSync();
    expect(readout()).toContain('15 boxes');

    // Shrinking the bin count below the hovered index clears selection without ever calling binLabel out of bounds.
    edges.set([0, 0.5, 1]);
    counts.set([10, 15]);
    flushSync();
    expect(readout()).toBe('25 boxes');
    expect(outOfBoundsLabels).toEqual([]);

    // Shrinking to empty and focusing the SVG keeps selection idle and tabindex=-1.
    edges.set([]);
    counts.set([]);
    flushSync();
    expect(svg.getAttribute('tabindex')).toBe('-1');
    svg.dispatchEvent(new FocusEvent('focus'));
    flushSync();
    expect(readout()).toBe('0 boxes');
    expect(outOfBoundsLabels).toEqual([]);
  });

  it('self-heals cached SVG rect on pointermove after an external vertical layout shift', () => {
    renderWide(400, { edges: [0, 0.25, 0.5, 0.75, 1], counts: [10, 20, 30, 40] });
    const svg = target.querySelector<SVGElement>('svg[aria-roledescription=chart]')!;
    let top = 0;
    svg.getBoundingClientRect = () =>
      ({ left: 0, top, width: 400, height: 160, right: 400, bottom: top + 160, x: 0, y: top, toJSON: () => ({}) }) as DOMRect;

    svg.dispatchEvent(new PointerEvent('pointerenter', { clientX: 80, clientY: 80, pointerType: 'mouse', bubbles: true }));
    svg.dispatchEvent(new PointerEvent('pointermove', { clientX: 80, clientY: 80, pointerType: 'mouse', bubbles: true }));
    flushSync();
    expect(readout()).toContain('10 boxes');

    // External banner shifts SVG down by 250px within the 200ms cache TTL; hover must still resolve the bar.
    top = 250;
    svg.dispatchEvent(new PointerEvent('pointermove', { clientX: 350, clientY: 330, pointerType: 'mouse', bubbles: true }));
    flushSync();
    expect(readout()).toContain('40 boxes');
  });
});
