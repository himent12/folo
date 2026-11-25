// @vitest-environment happy-dom
import { flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import ClassFilterStrip, { rankedClassChips } from './ClassFilterStrip';

let target: HTMLElement;
let mounted: { unmount(): void } | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target?.remove();
});

describe('ClassFilterStrip', () => {
  it('ranks classes by descending box count, truncates to limit, and pins an active tail class', () => {
    const classes = ['a', 'b', 'c', 'd', 'e'];
    const counts = [10, 50, 0, 50, 2];

    const collapsed = rankedClassChips(classes, counts, undefined, false, 3);
    expect(collapsed.visible.map((c) => c.cls)).toEqual([1, 3, 0]);
    expect(collapsed.hiddenCount).toBe(2);
    expect(collapsed.totalBoxes).toBe(112);
    expect(collapsed.visible[0]!.pctOfMax).toBe(100);

    // Pinning an active class outside the top N keeps it visible in the collapsed strip
    const pinned = rankedClassChips(classes, counts, 2, false, 3);
    expect(pinned.visible.map((c) => c.cls)).toEqual([1, 3, 0, 2]);
    expect(pinned.hiddenCount).toBe(1);
    expect(pinned.visible[3]!.pctOfMax).toBe(0);

    const expanded = rankedClassChips(classes, counts, 2, true, 3);
    expect(expanded.visible.map((c) => c.cls)).toEqual([1, 3, 0, 4, 2]);
    expect(expanded.hiddenCount).toBe(0);
  });

  it('toggles class selection on click and expands/collapses overflow classes', () => {
    const active = signal<number | undefined>(undefined);
    const onchange = vi.fn((c: number | undefined) => active.set(c));
    target = document.createElement('div');
    document.body.appendChild(target);
    mounted = render(
      () => (
        <ClassFilterStrip
          classes={['circle', 'square', 'triangle', 'star']}
          counts={[40, 120, 0, 15]}
          value={active.get()}
          onchange={onchange}
          limit={2}
        />
      ),
      target,
    );
    flushSync();

    const chips = () => [...target.querySelectorAll<HTMLButtonElement>('button.chip')];
    expect(chips().map((b) => b.querySelector('.cname')?.textContent)).toEqual(['square', 'circle']);

    // Click top chip selects class 1 ('square')
    chips()[0]!.click();
    flushSync();
    expect(onchange).toHaveBeenLastCalledWith(1);
    expect(chips()[0]!.getAttribute('aria-pressed')).toBe('true');
    expect(chips()[0]!.classList.contains('active')).toBe(true);

    // Clicking active chip again clears the filter back to undefined
    chips()[0]!.click();
    flushSync();
    expect(onchange).toHaveBeenLastCalledWith(undefined);
    expect(chips()[0]!.getAttribute('aria-pressed')).toBe('false');

    // Expand shows remaining classes including zero-count absent class
    const more = target.querySelector<HTMLButtonElement>('button.more-btn')!;
    expect(more.textContent?.trim()).toBe('+2 more');
    more.click();
    flushSync();
    expect(chips().map((b) => b.querySelector('.cname')?.textContent)).toEqual(['square', 'circle', 'star', 'triangle']);
    expect(chips()[3]!.classList.contains('absent')).toBe(true);
    expect(more.textContent?.trim()).toBe('Show fewer');
  });
});
