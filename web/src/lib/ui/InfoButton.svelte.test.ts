// @vitest-environment happy-dom
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { flushSync, mount, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
import InfoButton from './InfoButton.svelte';
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
