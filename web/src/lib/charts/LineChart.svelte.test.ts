// @vitest-environment happy-dom
import { flushSync, mount, tick, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import LineChart from './LineChart.svelte';

describe('LineChart announcements', () => {
  it('speaks keyboard-scrubbed values and goes quiet on pointer movement', async () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const series = [{ id: 'a', label: 'mAP', color: 'red', points: [0, 1, 2, 3].map((x) => ({ x, y: x / 10 })) }];
    const c = mount(LineChart, { target, props: { series, label: 'test chart', xTitle: (x: number) => `Epoch ${x}` } });
    // The plot renders once it has a width; happy-dom reports 0, so fake one.
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 400 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    const live = target.querySelector('[aria-live]')!;
    const svg = target.querySelector('svg[role=img]');
    expect(svg).toBeTruthy();
    if (svg) {
      svg.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
      flushSync();
      await tick();
      await tick();
      expect(live.textContent).toContain('Epoch 3');
      svg.dispatchEvent(new PointerEvent('pointermove', { clientX: 10, bubbles: true }));
      flushSync();
      expect(live.textContent).toBe('');
      // End → Escape → End announces again (Escape clears the region, so the same text is new).
      const key = (k: string) => svg.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true }));
      key('End');
      flushSync();
      await tick();
      await tick();
      key('Escape');
      flushSync();
      expect(live.textContent).toBe('');
      key('End');
      flushSync();
      await tick();
      await tick();
      expect(live.textContent).toContain('Epoch 3');
    }
    unmount(c);
  });
});

describe('LineChart x ticks', () => {
  it('labels a short domain with decimals instead of repeating integers', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const series = [{ id: 'a', label: 'loss', color: 'red', points: [0, 0.25, 0.5].map((x) => ({ x, y: 1 - x })) }];
    const c = mount(LineChart, { target, props: { series, label: 'short', xDomain: [0, 1] as [number, number] } });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 600 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    const labels = [...target.querySelectorAll('.ticks text')].map((t) => t.textContent);
    const xs = labels.slice(-6);
    expect(new Set(xs).size).toBe(xs.length);
    unmount(c);
  });
});

describe('LineChart touch readout', () => {
  it('pins the cursor on a tap, keeps it when the finger lifts, and lets go on a tap outside', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const series = [
      { id: 'a', label: 'long', color: 'red', points: [0, 1, 2, 3, 4].map((x) => ({ x, y: x / 10 })) },
      { id: 'b', label: 'short', color: 'blue', points: [0, 1].map((x) => ({ x, y: x / 5 })) },
    ];
    const c = mount(LineChart, { target, props: { series, label: 'touch', legendNote: 'Latest', xTitle: (x: number) => `Epoch ${x}` } });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 400 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    const svg = target.querySelector('svg[role=img]')!;
    const touch = (type: string, clientX: number, clientY = 50) =>
      svg.dispatchEvent(new PointerEvent(type, { pointerType: 'touch', clientX, clientY, bubbles: true }));
    // Narrow plot: no leading note at rest, so the legend is one line of whole items.
    expect(target.querySelector('.legend .note')).toBeNull();
    touch('pointerdown', 300);
    touch('pointerup', 300);
    touch('pointerleave', 300);
    flushSync();
    expect(target.querySelector('.cursor.pinned')).toBeTruthy();
    // The cursor's x reads out on the axis.
    expect(target.querySelector('.pill text')!.textContent).toMatch(/^Epoch \d$/);
    // The shorter series is painted last (on top).
    const lines = [...target.querySelectorAll('path.line')].map((p) => (p as SVGElement).style.stroke);
    expect(lines).toEqual(['red', 'blue']);
    document.body.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
    flushSync();
    expect(target.querySelector('.cursor')).toBeNull();
    unmount(c);
  });

  it('lets go when the pinned cursor is tapped again, or on Escape', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const series = [{ id: 'a', label: 'a', color: 'red', points: [0, 1, 2, 3, 4].map((x) => ({ x, y: x / 10 })) }];
    const c = mount(LineChart, { target, props: { series, label: 'touch' } });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: 600 });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    const svg = target.querySelector('svg[role=img]')!;
    const tap = (clientX: number) => {
      svg.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', clientX, clientY: 50, bubbles: true }));
      svg.dispatchEvent(new PointerEvent('pointerup', { pointerType: 'touch', clientX, clientY: 50, bubbles: true }));
      flushSync();
    };
    tap(300);
    const at = Number(target.querySelector('.cursor')!.getAttribute('x1'));
    tap(at + 4);
    expect(target.querySelector('.cursor')).toBeNull();
    tap(300);
    expect(target.querySelector('.cursor.pinned')).toBeTruthy();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    flushSync();
    expect(target.querySelector('.cursor')).toBeNull();
    unmount(c);
  });
});

describe('LineChart layering and ticks', () => {
  const mountAt = (props: Record<string, unknown>, w = 640) => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(LineChart, { target, props: props as never });
    Object.defineProperty(target.querySelector('.plot')!, 'clientWidth', { value: w });
    target.querySelector('.plot')!.dispatchEvent(new Event('resize'));
    flushSync();
    return { target, c };
  };
  const series = [
    { id: 'sky', label: 'sky', color: 'red', points: [0, 1, 2, 3, 4].map((x) => ({ x, y: x / 10 })) },
    { id: 'teal', label: 'teal', color: 'blue', pattern: 'dot' as const, points: [0, 1, 2, 3, 4, 5, 6].map((x) => ({ x, y: x / 10 })) },
  ];

  it('paints the highlighted series last, with its knockout, and keeps its dot rhythm', () => {
    const { target, c } = mountAt({ series, label: 'cmp', highlight: 'teal' });
    const lines = [...target.querySelectorAll('path.line')].map((p) => (p as SVGElement).style.stroke);
    expect(lines).toEqual(['red', 'blue']);
    // Drawn once (no duplicate copy on top), emphasised, the dots spaced for the heavier stroke.
    const teal = target.querySelectorAll('path.line')[1] as SVGElement;
    expect(teal.classList.contains('emphasis')).toBe(true);
    expect(teal.style.strokeDasharray).toMatch(/^0\.014,?\s*6\.3/);
    // The knockout sits directly under its line.
    expect(teal.previousElementSibling?.classList.contains('knockout')).toBe(true);
    unmount(c);
  });

  it('draws legend swatches in the pattern of their lines', () => {
    const { target, c } = mountAt({ series, label: 'cmp' });
    const sw = [...target.querySelectorAll('.legend .swatch line')].map((l) => (l as SVGElement).style.strokeDasharray);
    expect(sw[0]).toBe('');
    expect(sw[1]).toMatch(/^0\.01,?\s*4\.5/);
    unmount(c);
  });

  it('uses the y ticks the caller fitted', () => {
    const { target, c } = mountAt({ series, label: 'fit', yDomain: [0, 0.7], yTickValues: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] });
    const ys = [...target.querySelectorAll('.ticks text[text-anchor="end"]')].map((t) => t.textContent);
    expect(ys).toEqual(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7']);
    unmount(c);
  });
});
