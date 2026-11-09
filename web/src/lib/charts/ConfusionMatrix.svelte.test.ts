// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import ConfusionMatrix from './ConfusionMatrix.svelte';

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
    expect(target.querySelector('.readout')!.textContent).toMatch(/dog\s+predicted as\s+background:\s+2 of\s+10\s+\(20\.0%\)/);
    unmount(cm);
  });
});

describe('ConfusionMatrix labels', () => {
  it('labels shares from 5% (8.3, not only 10% and up), marks true zeros, and names every cell with its value', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    // bearing: 50 right, 5 taken for nut (8.3%), 2 for bolt (3.3%), 3 missed (5.0%); nothing taken for bearing by bolt.
    const matrix = [
      [40, 0, 0, 0],
      [0, 30, 0, 0],
      [2, 5, 50, 3],
      [1, 0, 0, 0],
    ];
    const cm = mount(ConfusionMatrix, { target, props: { classes: ['bolt', 'nut', 'bearing'], matrix, sparseLabels: true, minCell: 32 } });
    flushSync();
    const cell = (r: number, c: number) => target.querySelector<HTMLElement>(`[data-r="${r}"][data-c="${c}"]`)!;
    expect(cell(2, 1).textContent!.trim()).toBe('8.3');
    expect(cell(2, 3).textContent!.trim()).toBe('5.0');
    expect(cell(2, 2).textContent!.trim()).toBe('83');
    // Under 5%: no label, but its value is in its name; a true zero is the empty surface.
    expect(cell(2, 0).textContent!.trim()).toBe('');
    expect(cell(2, 0).getAttribute('aria-label')).toBe('bearing predicted as bolt: 2 of 60, 3.3%');
    expect(cell(2, 0).classList.contains('zero')).toBe(false);
    expect(cell(0, 1).classList.contains('zero')).toBe(true);
    expect(cell(0, 1).getAttribute('aria-label')).toBe('bolt predicted as nut: 0 of 40, 0.0%');
    for (const el of target.querySelectorAll('.cell')) expect(el.getAttribute('aria-label')).toMatch(/: \d+ of \d+, \d+\.\d%$/);
    unmount(cm);
  });
});
