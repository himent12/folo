// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest';
import { flush as flushSync } from 'fw';
import { render } from '../../../framework/testing';
import InfoPanel from './InfoPanel.testing';

/** Port of `mount(C, { target, props })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mount = (C: (p: any) => JSX.Element, o: { target: HTMLElement; props?: object }) => render(() => <C {...(o.props ?? {})} />, o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

// fw's MediaQuery shares one MediaQueryList per query text for the module's life (Svelte makes one per instance),
// so a test that stubs matchMedia for the phone query runs in a file of its own, before any other MediaQuery of it.

describe('InfoButton', () => {
  it('over a chart on a phone, opens in place under the panel heading, not over the plot', () => {
    const real = window.matchMedia;
    window.matchMedia = ((q: string) => ({ matches: q.includes('max-width: 767px'), media: q, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(InfoPanel, { target });
    flushSync();
    const button = target.querySelector<HTMLButtonElement>('button.info-btn')!;
    const region = document.getElementById(button.getAttribute('aria-controls')!)!;
    // The panel's region, after its header and before the readout: no popover is rendered.
    expect(region.classList.contains('info-region')).toBe(true);
    expect(region.previousElementSibling?.tagName).toBe('HEADER');
    expect(target.querySelector('.info-pop')).toBeNull();
    expect(region.hidden).toBe(true);
    button.click();
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    expect(region.hidden).toBe(false);
    expect(region.textContent).toContain('Size = √(box area');
    button.click();
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(region.hidden).toBe(true);
    unmount(c);
    target.remove();
    window.matchMedia = real;
  });

});
