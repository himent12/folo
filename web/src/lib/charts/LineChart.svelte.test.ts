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
