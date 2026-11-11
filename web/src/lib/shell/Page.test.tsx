// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { describe, expect, it, vi } from 'vitest';
import { fromHTML, render } from '../../../framework/testing';
import Page, { type PageProps } from './Page';

/** Port of `mount(Page, { target, props })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof Page, o: { target: HTMLElement; props: PageProps }) => render(() => <Page {...o.props} />, o.target);
const unmount = (c: { unmount(): void }) => c.unmount();
const text = (s: string) => () => fromHTML(`<span>${s}</span>`);

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
