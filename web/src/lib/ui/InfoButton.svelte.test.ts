// @vitest-environment happy-dom
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { flushSync, mount, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import InfoButton, { HELP_MAX } from './InfoButton.svelte';
import InfoPanel from './InfoPanel.test.svelte';

// happy-dom has no Popover API, so these run InfoButton's own open state (the same one a browser without popovers
// gets); in a browser the popover's light dismiss does the closing and the state follows its toggle event.
function setup(props: Record<string, unknown>) {
  const target = document.createElement('div');
  document.body.appendChild(target);
  const c = mount(InfoButton, { target, props: { label: 'What the loss terms mean', ...props } });
  const button = target.querySelector<HTMLButtonElement>('button')!;
  return {
    target,
    button,
    done: () => {
      unmount(c);
      target.remove();
    },
  };
}

describe('InfoButton', () => {
  it('opens by tap, click, Enter or Space (a click), with aria-expanded and aria-controls', () => {
    const { target, button, done } = setup({ text: 'Box: how far predicted boxes are from the true ones.' });
    const pop = document.getElementById(button.getAttribute('aria-controls')!)!;
    expect(button.getAttribute('aria-label')).toBe('What the loss terms mean');
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(pop.textContent).toContain('how far predicted boxes');
    expect(target.contains(pop)).toBe(true);
    button.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    button.click();
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    expect(pop.classList.contains('open')).toBe(true);
    // A second press on the button closes it.
    button.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    button.click();
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    done();
  });

  it('closes on Escape (focus stays on the button) and on a tap outside', () => {
    const { button, done } = setup({ text: 'Size = √(box area ÷ image area).' });
    button.focus();
    button.click();
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(button);
    button.click();
    flushSync();
    document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    done();
  });

  it('stays on screen when its fold opens: placed again, scrolling within the room it has', () => {
    const { button, done } = setup({ text: 'Loss: how wrong the model is.', how: 'A long method.' });
    const pop = document.getElementById(button.getAttribute('aria-controls')!)!;
    // A 600 px tall window, the button low on it; the popover 120 px tall, 500 px once its fold is open.
    const inner = window.innerHeight;
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 600 });
    let natural = 120;
    const rect = (x: number, y: number, w: number, h: number) => ({ x, y, left: x, top: y, width: w, height: h, right: x + w, bottom: y + h }) as DOMRect;
    button.getBoundingClientRect = () => rect(300, 400, 24, 24);
    pop.getBoundingClientRect = () => rect(0, 0, 320, pop.style.maxHeight ? parseFloat(pop.style.maxHeight) : natural);
    flushSync();
    button.click();
    flushSync();
    const y = () => Number(pop.style.translate.split(' ')[1]!.replace('px', ''));
    expect(y()).toBeGreaterThanOrEqual(424);
    expect(pop.style.maxHeight).toBe('');
    natural = 500;
    const fold = pop.querySelector('details')!;
    fold.open = true;
    fold.dispatchEvent(new Event('toggle'));
    const height = parseFloat(pop.style.maxHeight);
    expect(height).toBeLessThan(500);
    expect(y()).toBeGreaterThanOrEqual(0);
    expect(y() + height).toBeLessThanOrEqual(600);
    // Above the button (the side with more room), its foot by the button.
    expect(y() + height).toBeLessThanOrEqual(400);
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: inner });
    done();
  });

  it('drives an inline disclosure the page renders (Model’s metric help)', () => {
    const region = document.createElement('dl');
    region.id = 'metric-help';
    document.body.appendChild(region);
    const { target, button, done } = setup({ label: 'What these metrics mean', controls: 'metric-help' });
    expect(button.getAttribute('aria-controls')).toBe('metric-help');
    expect(target.querySelector('[popover], .info-pop')).toBeNull();
    button.click();
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    // A tap inside the region keeps it open; Escape closes it.
    region.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    done();
    region.remove();
  });

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

  it('every icon ⓘ says how it opens: in place over a chart (`chart`), a page disclosure (`controls`), or words', () => {
    const src = join(__dirname, '..', '..');
    const files = (dir: string): string[] =>
      readdirSync(dir).flatMap((n) => {
        const p = join(dir, n);
        return statSync(p).isDirectory() ? files(p) : n.endsWith('.svelte') && !n.endsWith('.test.svelte') ? [p] : [];
      });
    const bad: string[] = [];
    for (const f of files(src)) {
      for (const m of readFileSync(f, 'utf-8').matchAll(/<InfoButton\b[^>]*>/g)) {
        if (!/\bchart\b|\bcontrols=|\btrigger=/.test(m[0])) bad.push(`${relative(src, f)}: ${m[0].replace(/\s+/g, ' ')}`);
      }
    }
    expect(bad).toEqual([]);
  });

  it("keeps every definition to two lines (HELP_MAX), its detail folded under How it's computed", () => {
    const src = join(__dirname, '..', '..');
    const files = (dir: string): string[] =>
      readdirSync(dir).flatMap((n) => {
        const p = join(dir, n);
        return statSync(p).isDirectory() ? files(p) : n.endsWith('.svelte') && !n.endsWith('.test.svelte') ? [p] : [];
      });
    const long: string[] = [];
    let seen = 0;
    for (const f of files(src)) {
      for (const m of readFileSync(f, 'utf-8').matchAll(/<InfoButton\b(?:[^<>]|=>)*?\btext="([^"]*)"/g)) {
        seen++;
        if (m[1]!.length > HELP_MAX) long.push(`${relative(src, f)}: ${m[1]!.length} characters: ${m[1]}`);
      }
    }
    expect(seen).toBeGreaterThan(0);
    expect(long).toEqual([]);
    // The fold: closed on every opening, the detail behind it.
    const { target, button, done } = setup({ text: 'A short line.', how: 'Size = √(box area ÷ image area).' });
    button.click();
    flushSync();
    const fold = target.querySelector<HTMLDetailsElement>('details.info-how')!;
    expect(fold.querySelector('summary')?.textContent).toBe("How it's computed");
    expect(fold.open).toBe(false);
    expect(fold.textContent).toContain('Size = √(box area');
    done();
  });

  it('every ⓘ in the app is an InfoButton, never a hover-only icon button', () => {
    const src = join(__dirname, '..', '..');
    const files = (dir: string): string[] =>
      readdirSync(dir).flatMap((n) => {
        const p = join(dir, n);
        return statSync(p).isDirectory() ? files(p) : n.endsWith('.svelte') ? [p] : [];
      });
    const bad: string[] = [];
    for (const f of files(src)) {
      if (f.endsWith('InfoButton.svelte')) continue;
      const text = readFileSync(f, 'utf-8');
      // The info icon inside a hand-made button (its attributes may hold an arrow function), or as an icon button's.
      if (/<button\b(?:[^<]|=>)*?>\s*<Info\b/.test(text) || /icon=\{Info\}/.test(text)) bad.push(relative(src, f));
    }
    expect(bad).toEqual([]);
  });
});
