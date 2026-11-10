// @vitest-environment happy-dom
import { createRawSnippet, flushSync, mount, unmount } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
import Page from './Page.svelte';

const text = (s: string) => createRawSnippet(() => ({ render: () => `<span>${s}</span>` }));

describe('Page header actions', () => {
  it('runs a `more` action from its wide button, and names the phone menu that holds it', () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const onclick = vi.fn();
    const c = mount(Page, {
      target,
      props: { title: 'folo-s-best', children: text('body'), actions: text('Try'), more: [{ label: 'Export', onclick }], moreLabel: 'More model actions' } as never,
    });
    flushSync();
    target.querySelector<HTMLButtonElement>('.more-wide button')!.click();
    expect(onclick).toHaveBeenCalledOnce();
    expect(target.querySelector('.more-phone button[aria-haspopup="menu"]')!.getAttribute('aria-label')).toBe('More model actions');
    unmount(c);
    target.remove();
  });
});
