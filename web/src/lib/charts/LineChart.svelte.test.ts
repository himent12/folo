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
    }
    unmount(c);
  });
});
