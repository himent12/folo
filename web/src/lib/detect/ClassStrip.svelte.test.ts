// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import ClassStrip from './ClassStrip.svelte';

describe('ClassStrip', () => {
  it('is one disclosure that opens a name / count / share breakdown, with no native titles', () => {
    // Round 26: the segments' names and counts were only in native hover titles.
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(ClassStrip, { target, props: { counts: [300, 0, 100, 600], classes: ['bolt', 'nut', 'washer', 'screw'] } });
    flushSync();
    const buttons = target.querySelectorAll('button');
    expect(buttons).toHaveLength(1);
    const b = buttons[0]!;
    expect(b.getAttribute('aria-expanded')).toBe('false');
    expect(target.querySelector('[title]')).toBeNull();
    b.click();
    flushSync();
    expect(b.getAttribute('aria-expanded')).toBe('true');
    const pop = document.getElementById(b.getAttribute('aria-controls')!)!;
    const rows = [...pop.querySelectorAll('[role=listitem]')].map((r) => r.textContent!.replace(/\s+/g, ' ').trim());
    // Largest first; a class with no boxes is not listed.
    expect(rows).toEqual(['screw 600 60%', 'bolt 300 30%', 'washer 100 10%']);
    unmount(c);
    target.remove();
  });
});
