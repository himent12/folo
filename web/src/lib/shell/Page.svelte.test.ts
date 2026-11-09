// @vitest-environment happy-dom
import { createRawSnippet, flushSync, mount, unmount } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
import Page from './Page.svelte';

const text = (s: string) => createRawSnippet(() => ({ render: () => `<span>${s}</span>` }));

function render(props: Record<string, unknown>) {
  const target = document.createElement('div');
  document.body.appendChild(target);
  const c = mount(Page, { target, props: { title: 'folo-s-best', children: text('body'), ...props } as never });
  flushSync();
  return {
    target,
    done: () => {
      unmount(c);
      target.remove();
    },
  };
}

describe('Page header actions', () => {
  it('puts the actions after the title block, in one row the layout places beside the title', () => {
    const r = render({ sub: text('folo-s · 3 classes'), actions: text('<button>Try</button>') });
    const header = r.target.querySelector('header')!;
    const kids = [...header.children].map((e) => e.className.split(' ')[0]);
    expect(kids).toEqual(['titles', 'actions']);
    expect(header.querySelector('.titles h1')?.textContent).toBe('folo-s-best');
    r.done();
  });

  it('renders `more` as buttons for wide screens and as one ⋯ menu for phones', () => {
    const onclick = vi.fn();
    const r = render({ actions: text('Try'), more: [{ label: 'Export', onclick }], moreLabel: 'More model actions' });
    const wide = r.target.querySelector('.more-wide button')!;
    expect(wide.textContent?.trim()).toBe('Export');
    (wide as HTMLButtonElement).click();
    expect(onclick).toHaveBeenCalledOnce();
    const menu = r.target.querySelector('.more-phone button[aria-haspopup="menu"]')!;
    expect(menu.getAttribute('aria-label')).toBe('More model actions');
    // Order: secondary buttons, the page's actions (primary last), the phone menu.
    const order = [...r.target.querySelector('.actions')!.children].map((e) => e.className.split(' ')[0]);
    expect(order[0]).toBe('more-wide');
    expect(order.at(-1)).toBe('more-phone');
    r.done();
  });

  it('a title-only header is bare (it takes no room on phones)', () => {
    const r = render({});
    expect(r.target.querySelector('header')!.classList.contains('bare')).toBe(true);
    expect(r.target.querySelector('.actions')).toBeNull();
    r.done();
  });

  it('`more` alone is enough to give the header an actions slot', () => {
    const r = render({ more: [{ label: 'Delete', onclick: () => {}, danger: true }] });
    expect(r.target.querySelector('header')!.classList.contains('bare')).toBe(false);
    expect(r.target.querySelector('.more-wide button')!.className).toContain('danger');
    r.done();
  });
});

describe('Page header on phones: one row', () => {
  it('`subPhone` replaces the meta line on phones and makes the header one row', () => {
    const r = render({ sub: text('Training runs, newest first. Select up to four to compare.'), subPhone: text('Newest first'), actions: text('<button>New run</button>') });
    const header = r.target.querySelector('header')!;
    expect(header.classList.contains('one-row')).toBe(true);
    expect(header.querySelector('.sub.wide-only')?.textContent).toContain('Select up to four');
    expect(header.querySelector('.sub.phone-only')?.textContent).toBe('Newest first');
    r.done();
  });

  it('without `subPhone` the header keeps its single meta line and no phone copy', () => {
    const r = render({ sub: text('folo-s · 3 classes'), actions: text('<button>Try</button>') });
    const header = r.target.querySelector('header')!;
    expect(header.classList.contains('one-row')).toBe(false);
    expect(header.querySelectorAll('.sub')).toHaveLength(1);
    expect(header.querySelector('.sub')!.classList.contains('wide-only')).toBe(false);
    r.done();
  });
});
