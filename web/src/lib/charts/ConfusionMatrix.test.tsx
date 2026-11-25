// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import ConfusionMatrix, { type ConfusionMatrixApi, type ConfusionMatrixProps } from './ConfusionMatrix';

/** Port of `mount(ConfusionMatrix, { target, props })`: the instance's `reveal` comes through the `expose` prop. */
const mount = (_: typeof ConfusionMatrix, o: { target: HTMLElement; props: ConfusionMatrixProps }) => {
  let api!: ConfusionMatrixApi;
  const r = render(() => <ConfusionMatrix {...o.props} expose={(a) => (api = a)} />, o.target);
  return { reveal: (row: number, col: number) => api.reveal(row, col), unmount: () => r.unmount() };
};
const unmount = (c: { unmount(): void }) => c.unmount();

describe('ConfusionMatrix reveal', () => {
  it('makes a cell the active one: focused, marked and stated by the readout', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const matrix = [
      [8, 1, 1],
      [2, 6, 2],
      [1, 0, 0],
    ];
    const cm = mount(ConfusionMatrix, { target, props: { classes: ['cat', 'dog'], matrix, minCell: 32 } });
    flushSync();
    // dog (row 1) missed as background (column 2).
    cm.reveal(1, 2);
    flushSync();
    const cell = target.querySelector<HTMLElement>('[data-r="1"][data-c="2"]')!;
    expect(document.activeElement).toBe(cell);
    expect(cell.classList.contains('pin')).toBe(true);
    expect(cell.tabIndex).toBe(0);
    expect(target.querySelectorAll('.cell.pin').length).toBe(1);
    expect(target.querySelector('.readout')!.textContent).toMatch(/dog\s+predicted as\s+background:\s+2 of\s+10\s+\(20%\)/);
    unmount(cm);
  });
});

describe('ConfusionMatrix labels', () => {
  it('labels every share in whole percentages (8, not 8.3), with a dot as the fallback, and names every cell with its value', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    // bearing: 50 right, 5 taken for nut (8.3%), 2 for bolt (3.3%), 3 missed (5.0%); nothing taken for bearing by bolt.
    const matrix = [
      [40, 0, 0, 0],
      [0, 30, 0, 0],
      [2, 5, 50, 3],
      [1, 0, 0, 0],
    ];
    const cm = mount(ConfusionMatrix, { target, props: { classes: ['bolt', 'nut', 'bearing'], matrix, minCell: 32 } });
    flushSync();
    const cell = (r: number, c: number) => target.querySelector<HTMLElement>(`[data-r="${r}"][data-c="${c}"]`)!;
    expect(cell(2, 1).textContent!.trim()).toBe('8');
    expect(cell(2, 3).textContent!.trim()).toBe('5');
    expect(cell(2, 2).textContent!.trim()).toBe('83');
    expect(cell(2, 0).getAttribute('aria-label')).toBe('bearing predicted as bolt: 2 of 60, 3%');
    expect(cell(0, 1).getAttribute('aria-label')).toBe('bolt predicted as nut: 0 of 40, 0%');
    for (const el of target.querySelectorAll('.cell')) expect(el.getAttribute('aria-label')).toMatch(/: \d+ of \d+, (<1|\d+)%$/);
    unmount(cm);
  });

  it('delegates pointer and focus events on the grid root without child flicker and rAF-throttles scroll', () => {
    vi.useFakeTimers();
    const target = document.createElement('div');
    document.body.appendChild(target);
    const matrix = [
      [8, 2, 0],
      [1, 9, 0],
      [0, 0, 0],
    ];
    const cm = mount(ConfusionMatrix, { target, props: { classes: ['cat', 'dog'], matrix, minCell: 32 } });
    try {
      flushSync();
      const grid = target.querySelector<HTMLElement>('.cm')!;
      const c00 = target.querySelector<HTMLElement>('[data-r="0"][data-c="0"]')!;
      const c00Val = c00.querySelector<HTMLElement>('.v')!;
      const c01 = target.querySelector<HTMLElement>('[data-r="0"][data-c="1"]')!;

      // Hover cell (0, 0) via delegated pointerover -> readout updates.
      c00.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, relatedTarget: grid }));
      flushSync();
      expect(c00.classList.contains('hl')).toBe(true);
      expect(target.querySelector('.readout')!.textContent).toMatch(/cat\s+predicted as\s+cat/);

      // Move onto cell child (.v) -> stays hovered without clearing.
      c00.dispatchEvent(new PointerEvent('pointerout', { bubbles: true, relatedTarget: c00Val }));
      c00Val.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, relatedTarget: c00 }));
      flushSync();
      expect(c00.classList.contains('hl')).toBe(true);

      // Focus cell (0, 1) via delegated focusin -> pins cell (0, 1).
      c01.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
      flushSync();
      expect(c01.classList.contains('pin')).toBe(true);

      // Leaving hovered cell to outside restores pinned cell (0, 1).
      c00.dispatchEvent(new PointerEvent('pointerout', { bubbles: true, relatedTarget: target }));
      flushSync();
      expect(target.querySelector('.readout')!.textContent).toMatch(/cat\s+predicted as\s+dog/);

      // Focus leaving grid clears pin and hover.
      c01.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: target }));
      flushSync();
      expect(target.querySelectorAll('.cell.pin')).toHaveLength(0);

      // Burst of scroll events coalesces into one rAF measurement.
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame');
      grid.dispatchEvent(new Event('scroll'));
      grid.dispatchEvent(new Event('scroll'));
      grid.dispatchEvent(new Event('scroll'));
      expect(rafSpy).toHaveBeenCalledTimes(1);
      vi.advanceTimersByTime(20);
    } finally {
      unmount(cm);
      target.remove();
      vi.restoreAllMocks();
      vi.useRealTimers();
    }
  });
});
