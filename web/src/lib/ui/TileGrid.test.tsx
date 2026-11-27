// @vitest-environment happy-dom
import { flush, signal } from 'fw';
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

  it('clamps onrange bounds to [0, total] and keeps a valid Tab stop when a focused partial row shrinks', async () => {
    const total = signal(10);
    const ranges: [number, number, number, number][] = [];
    const { container, unmount } = render(() => (
      <TileGrid
        total={total.get()}
        density="m"
        aspect={4 / 3}
        label={(i) => `tile ${i}`}
        onopen={() => {}}
        onrange={(lo, hi, visLo, visHi) => ranges.push([lo, hi, visLo, visHi])}
        toolbar=".toolbar"
        directional
        tile={(i) => <span>{i()}</span>}
      />
    ));
    try {
      flush();
      const lastRange = ranges.at(-1)!;
      expect(lastRange[0]).toBeGreaterThanOrEqual(0);
      expect(lastRange[1]).toBeLessThanOrEqual(10);
      expect(lastRange[2]).toBeGreaterThanOrEqual(lastRange[0]);
      expect(lastRange[3]).toBeGreaterThanOrEqual(lastRange[2]);
      expect(lastRange[3]).toBeLessThanOrEqual(lastRange[1]);
      // Focus the last item (index 9, on partial row 2), then shrink total to 9.
      const lastCell = container.querySelector<HTMLElement>('.cell[data-i="9"]')!;
      lastCell.dispatchEvent(new FocusEvent('focus'));
      flush();
      expect(container.querySelector<HTMLElement>('.cell[tabindex="0"]')?.dataset.i).toBe('9');
      total.set(9);
      flush();
      const active = container.querySelector<HTMLElement>('.cell[tabindex="0"]');
      expect(active).not.toBeNull();
      expect(Number(active!.dataset.i)).toBeLessThan(9);
      // Grid below the fold must never emit negative visHi.
      gridTop = 1200;
      window.dispatchEvent(new Event('scroll'));
      await vi.advanceTimersByTimeAsync(20);
      flush();
      const below = ranges.at(-1)!;
      expect(below.every((v) => v >= 0 && v <= 9)).toBe(true);
      expect(below[2]).toBeLessThanOrEqual(below[3]);
    } finally {
      unmount();
    }
  });

  it('biases overscan 3:1 toward scroll direction, caches toolbar queries, and guards invalid aspect ratios', async () => {
    const tb = document.createElement('div');
    tb.className = 'toolbar';
    document.body.appendChild(tb);
    const qsSpy = vi.spyOn(document, 'querySelector');
    const aspect = signal(4 / 3);
    const ranges: [number, number, number, number][] = [];
    const { container, unmount } = render(() => (
      <TileGrid
        total={400}
        density="m"
        aspect={aspect.get()}
        label={(i) => `tile ${i}`}
        onopen={() => {}}
        onrange={(lo, hi, visLo, visHi) => ranges.push([lo, hi, visLo, visHi])}
        toolbar=".toolbar"
        directional
        tile={(i) => <span>{i()}</span>}
      />
    ));
    try {
      flush();
      qsSpy.mockClear();
      // Scroll down to row 20 (4 cols, rowH = 154): 1 row trailing overscan above, 3 rows leading overscan below.
      gridTop = -20 * 154;
      window.dispatchEvent(new Event('scroll'));
      await vi.advanceTimersByTimeAsync(20);
      flush();
      const down = ranges.at(-1)!;
      expect(down[2] - down[0]).toBe(4); // 1 row above
      expect(down[1] - down[3]).toBe(12); // 3 rows below
      // Cached toolbar avoids querying DOM again during scroll frames.
      expect(qsSpy).not.toHaveBeenCalled();

      // Scroll back up into row 10 (40 px past its top edge): 3 rows leading overscan above, 1 row trailing overscan below.
      gridTop = -10 * 154 - 40;
      window.dispatchEvent(new Event('scroll'));
      await vi.advanceTimersByTimeAsync(20);
      flush();
      const up = ranges.at(-1)!;
      expect(up[2] - up[0]).toBe(12); // 3 rows above
      expect(up[1] - up[3]).toBe(4); // 1 row below

      // Sub-row scroll within the same direction and visible rows must not emit a duplicate onrange call.
      const beforeSubRow = ranges.length;
      gridTop = -10 * 154 - 36;
      window.dispatchEvent(new Event('scroll'));
      await vi.advanceTimersByTimeAsync(20);
      flush();
      expect(ranges.length).toBe(beforeSubRow);

      // Replacing a disconnected toolbar invalidates the toolbar cache and picks up the new element.
      tb.remove();
      const tb2 = document.createElement('div');
      tb2.className = 'toolbar';
      document.body.appendChild(tb2);
      qsSpy.mockClear();
      window.dispatchEvent(new Event('resize'));
      await vi.advanceTimersByTimeAsync(20);
      flush();
      expect(qsSpy).toHaveBeenCalledWith('.toolbar');
      tb2.remove();

      // Invalid aspect (0 / NaN) must fall back cleanly without NaN heights or empty grid collapse.
      aspect.set(0);
      flush();
      const h = parseFloat((container.querySelector('.grid') as HTMLElement).style.height);
      expect(Number.isFinite(h) && h > 0).toBe(true);
    } finally {
      unmount();
      tb.remove();
    }
  });

  it('defers ResizeObserver width updates via coalesced rAF, skips height-only notifications, and cleans up on unmount', async () => {
    const roCallbacks: ResizeObserverCallback[] = [];
    const disconnect = vi.fn();
    class FakeResizeObserver {
      constructor(cb: ResizeObserverCallback) {
        roCallbacks.push(cb);
      }
      observe() {}
      unobserve() {}
      disconnect = disconnect;
    }
    const origRO = globalThis.ResizeObserver;
    globalThis.ResizeObserver = FakeResizeObserver as unknown as typeof ResizeObserver;

    let gridWidth = 800;
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function (this: HTMLElement) {
      return this.classList.contains('grid') ? gridWidth : 0;
    });
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame');
    const cafSpy = vi.spyOn(window, 'cancelAnimationFrame');

    const { container, unmount } = render(() => (
      <TileGrid
        total={16}
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
      const grid = container.querySelector('.grid') as HTMLElement;
      const initialHeight = grid.style.height;
      const initialCols = grid.getAttribute('aria-colcount');
      expect(initialCols).toBe('4');
      expect(roCallbacks.length).toBe(1);

      // Height-only ResizeObserver notification (clientWidth unchanged) must not schedule a frame.
      rafSpy.mockClear();
      roCallbacks[0]!([], {} as ResizeObserver);
      expect(rafSpy).not.toHaveBeenCalled();

      // Width shrink (e.g. vertical scrollbar appearing or viewport narrowing): must NOT mutate `.grid` height
      // synchronously inside the ResizeObserver callback (which triggers Chromium's ResizeObserver loop error).
      gridWidth = 360;
      roCallbacks[0]!([], {} as ResizeObserver);
      roCallbacks[0]!([], {} as ResizeObserver);
      flush();
      expect(rafSpy).toHaveBeenCalledTimes(1);
      expect(grid.style.height).toBe(initialHeight);
      expect(grid.getAttribute('aria-colcount')).toBe(initialCols);

      // Next animation frame applies the new column count and height cleanly.
      await vi.advanceTimersByTimeAsync(20);
      flush();
      expect(grid.getAttribute('aria-colcount')).toBe('2');
      expect(grid.style.height).not.toBe(initialHeight);

      // Pending rAF is cancelled and observer disconnected when the grid unmounts.
      gridWidth = 600;
      roCallbacks[0]!([], {} as ResizeObserver);
      unmount();
      expect(cafSpy).toHaveBeenCalled();
      expect(disconnect).toHaveBeenCalledTimes(1);
    } finally {
      globalThis.ResizeObserver = origRO;
    }
  });
});
