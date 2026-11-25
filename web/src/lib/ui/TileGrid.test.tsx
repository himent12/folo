// @vitest-environment happy-dom
import { flush } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import TileGrid from './TileGrid';

describe('TileGrid scroll scheduling and tabindex stability', () => {
  let gridTop = 0;

  beforeEach(() => {
    vi.useFakeTimers();
    gridTop = 0;
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function (this: HTMLElement) {
      return this.classList.contains('grid') ? 800 : 0;
    });
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      const top = this.classList.contains('grid') ? gridTop : 0;
      return { top, bottom: top, left: 0, right: 0, width: 800, height: 0, x: 0, y: top, toJSON: () => ({}) } as DOMRect;
    });
    vi.spyOn(window, 'innerHeight', 'get').mockReturnValue(600);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    document.body.innerHTML = '';
  });

  it('does not starve rAF measurement during continuous high-frequency scroll events', async () => {
    const ranges: [number, number][] = [];
    const cancelSpy = vi.spyOn(window, 'cancelAnimationFrame');
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame');
    const { unmount } = render(() => (
      <TileGrid
        total={400}
        density="m"
        aspect={4 / 3}
        label={(i) => `tile ${i}`}
        onopen={() => {}}
        onrange={(lo, hi) => ranges.push([lo, hi])}
        toolbar=".toolbar"
        tile={(i) => <span>{i()}</span>}
      />
    ));
    try {
      flush();
      ranges.length = 0;
      cancelSpy.mockClear();
      rafSpy.mockClear();
      // Continuous 120 Hz scroll events (every 8 ms, faster than a 16 ms frame): must not cancel the pending rAF.
      for (let k = 1; k <= 6; k++) {
        gridTop = -k * 160;
        window.dispatchEvent(new Event('scroll'));
        await vi.advanceTimersByTimeAsync(8);
        flush();
      }
      await vi.advanceTimersByTimeAsync(20);
      flush();
      expect(cancelSpy).not.toHaveBeenCalled();
      expect(ranges.length).toBeGreaterThanOrEqual(2);
      expect(ranges.at(-1)![0]).toBeGreaterThan(0);
    } finally {
      unmount();
    }
  });

  it('keeps tabIdx on the first visible row without churning on sub-row pixel scrolls', async () => {
    const { container, unmount } = render(() => (
      <TileGrid
        total={400}
        density="m"
        aspect={4 / 3}
        label={(i) => `tile ${i}`}
        onopen={() => {}}
        toolbar=".toolbar"
        tile={(i) => <span>{i()}</span>}
      />
    ));
    try {
      flush();
      // Scroll far enough that focusIdx (0) leaves the rendered window (past overscan).
      gridTop = -1000;
      window.dispatchEvent(new Event('scroll'));
      await vi.advanceTimersByTimeAsync(20);
      flush();
      const tabCell = () => container.querySelector<HTMLElement>('.cell[tabindex="0"]')?.dataset.i;
      const before = tabCell();
      expect(Number(before)).toBeGreaterThan(0);
      // Small sub-row scroll (5 px): same first visible row, same tab stop.
      gridTop = -1005;
      window.dispatchEvent(new Event('scroll'));
      await vi.advanceTimersByTimeAsync(20);
      flush();
      expect(tabCell()).toBe(before);
    } finally {
      unmount();
    }
  });
});
