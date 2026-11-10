// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
import SortHeader from './SortHeader.svelte';

function render(props: { label: string; dir?: 'asc' | 'desc' | null; onsort: () => void; align?: 'left' | 'right' }) {
  const target = document.createElement('tr');
  document.body.appendChild(target);
  const c = mount(SortHeader, { target, props });
  return {
    th: target.querySelector('th')!,
    done: () => {
      unmount(c);
      target.remove();
    },
  };
}

describe('SortHeader', () => {
  it('marks the sorted column with aria-sort, leaves it off the others, and sorts from its button', () => {
    const onsort = vi.fn();
    const sorted = render({ label: 'mAP50-95', dir: 'desc', onsort, align: 'right' });
    expect(sorted.th.getAttribute('aria-sort')).toBe('descending');
    sorted.th.querySelector('button')!.click();
    flushSync();
    expect(onsort).toHaveBeenCalledOnce();
    sorted.done();
    const unsorted = render({ label: 'Model', dir: null, onsort: () => {} });
    expect(unsorted.th.hasAttribute('aria-sort')).toBe(false);
    unsorted.done();
  });
});
