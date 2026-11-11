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
  it('keeps a chart’s open definition on screen across the phone breakpoint, in place or in its popover', () => {
    // Round 26 review: open in place at 700 px, widened to 1100 px, the region went and the popover was never shown,
    // while aria-expanded stayed true. The Popover API stubbed (happy-dom has none), and a phone query that can change.
    const proto = HTMLElement.prototype as unknown as Record<string, unknown>;
    const realMatches = Element.prototype.matches;
    const shown = new WeakSet<Element>();
    proto.showPopover = function (this: HTMLElement) {
      shown.add(this);
    };
    proto.hidePopover = function (this: HTMLElement) {
      shown.delete(this);
    };
    Element.prototype.matches = function (this: Element, q: string) {
      return q === ':popover-open' ? shown.has(this) : realMatches.call(this, q);
    } as typeof realMatches;
    const real = window.matchMedia;
    let phone = true;
    const listeners = new Set<(e: Event) => void>();
    window.matchMedia = ((q: string) => ({
      get matches() {
        return q.includes('max-width: 767px') && phone;
      },
      media: q,
      addEventListener: (_: string, f: (e: Event) => void) => listeners.add(f),
      removeEventListener: (_: string, f: (e: Event) => void) => listeners.delete(f),
    })) as unknown as typeof window.matchMedia;
    const resize = (toPhone: boolean) => {
      phone = toPhone;
      for (const f of listeners) f(new Event('change'));
      flushSync();
    };
    try {
      const target = document.createElement('div');
      document.body.appendChild(target);
      const c = mount(InfoPanel, { target });
      flushSync();
      const button = target.querySelector<HTMLButtonElement>('button.info-btn')!;
      button.click();
      flushSync();
      const region = document.getElementById(button.getAttribute('aria-controls')!)!;
      expect(region.classList.contains('info-region')).toBe(true);
      expect(region.hidden).toBe(false);
      // Wider than a phone: the popover takes over, shown, and still open.
      resize(false);
      const pop = target.querySelector<HTMLElement>('.info-pop')!;
      expect(pop).not.toBeNull();
      expect(button.getAttribute('aria-controls')).toBe(pop.id);
      expect(shown.has(pop)).toBe(true);
      expect(button.getAttribute('aria-expanded')).toBe('true');
      expect(region.hidden).toBe(true);
      // And back: in place again, the popover gone with its element.
      resize(true);
      expect(target.querySelector('.info-pop')).toBeNull();
      expect(button.getAttribute('aria-expanded')).toBe('true');
      expect(region.hidden).toBe(false);
      expect(region.textContent).toContain('Size = √(box area');
      unmount(c);
      target.remove();
    } finally {
      delete proto.showPopover;
      delete proto.hidePopover;
      Element.prototype.matches = realMatches;
      window.matchMedia = real;
    }
  });

});
