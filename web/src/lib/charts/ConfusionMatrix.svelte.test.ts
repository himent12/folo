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
