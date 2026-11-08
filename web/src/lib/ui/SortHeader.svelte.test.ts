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
  it('marks the sorted column with aria-sort and sorts from its button', () => {
    const onsort = vi.fn();
    const { th, done } = render({ label: 'mAP50-95', dir: 'desc', onsort, align: 'right' });
    expect(th.getAttribute('aria-sort')).toBe('descending');
    expect(th.getAttribute('scope')).toBe('col');
    expect(th.classList.contains('r')).toBe(true);
    const button = th.querySelector('button')!;
    expect(button.textContent?.trim()).toBe('mAP50-95');
    button.click();
    flushSync();
    expect(onsort).toHaveBeenCalledOnce();
    done();
  });

  it('leaves aria-sort off unsorted columns', () => {
    const { th, done } = render({ label: 'Model', dir: null, onsort: () => {} });
    expect(th.hasAttribute('aria-sort')).toBe(false);
    done();
  });
});
