// @vitest-environment happy-dom
import { flushSync, mount, tick, unmount } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import LossOptions, { lossSummary } from './LossOptions.svelte';

describe('LossOptions', () => {
  it('names the current view on its one button', () => {
    expect(lossSummary('0.9', 'log')).toBe('EMA 0.9 · Log');
    expect(lossSummary('0', 'lin')).toBe('Raw · Linear');
  });

  it('opens on the chosen smoothing, applies a choice, and closes on Escape back to its button', async () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const props = { smoothing: '0.9' as const, scale: 'log' as const };
    const c = mount(LossOptions, { target, props });
    flushSync();
    const button = target.querySelector<HTMLButtonElement>('button.trigger')!;
    expect(button.getAttribute('aria-expanded')).toBe('false');
    button.click();
    flushSync();
    await tick();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    const checked = target.querySelector<HTMLInputElement>('input:checked')!;
    expect(document.activeElement).toBe(checked);
    expect(checked.value).toBe('0.9');
    // Two native radio groups: smoothing and y scale.
    expect(target.querySelectorAll('fieldset').length).toBe(2);
    const lin = target.querySelector<HTMLInputElement>('input[value="lin"]')!;
    lin.checked = true;
    lin.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(button.textContent).toContain('EMA 0.9 · Linear');
    target.querySelector('[role="dialog"]')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    expect(target.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(button);
    unmount(c);
  });

  it('closes on Escape from its button too (after Shift+Tab back from the panel)', async () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(LossOptions, { target, props: { smoothing: '0.9' as const, scale: 'log' as const } });
    flushSync();
    const button = target.querySelector<HTMLButtonElement>('button.trigger')!;
    button.click();
    flushSync();
    await tick();
    button.focus();
    button.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    expect(target.querySelector('[role="dialog"]')).toBeNull();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(button);
    unmount(c);
  });

  describe('on phones', () => {
    const width = window.innerWidth;
    const setWidth = (w: number) => (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: w });
    afterEach(() => setWidth(width));

    it('is the shared modal sheet: one dismiss (Close), radio marks, Tab stays inside, choices apply at once, focus returns', async () => {
      HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
        this.open = true;
      };
      HTMLDialogElement.prototype.close ??= function (this: HTMLDialogElement) {
        this.open = false;
        this.dispatchEvent(new Event('close'));
      };
      setWidth(390);
      const target = document.createElement('div');
      document.body.appendChild(target);
      const c = mount(LossOptions, { target, props: { smoothing: '0.9' as const, scale: 'log' as const } });
      flushSync();
      const button = target.querySelector<HTMLButtonElement>('button.trigger')!;
      button.focus();
      button.click();
      flushSync();
      await tick();
      // A native modal <dialog> (the page inert behind it), not a div with role=dialog.
      const dialog = document.querySelector<HTMLDialogElement>('dialog')!;
      expect(dialog.open).toBe(true);
      expect(target.querySelector('div[role="dialog"]')).toBeNull();
      const done = dialog.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!;
      expect(done).not.toBeNull();
      expect([...dialog.querySelectorAll('button')].some((b) => b.textContent?.trim() === 'Done')).toBe(false);
      expect(dialog.querySelectorAll('.opt .mark')).toHaveLength(6);
      const checked = dialog.querySelector<HTMLInputElement>('input:checked')!;
      expect(document.activeElement).toBe(checked);
      // Tab out of a group does not dismiss it.
      checked.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: done }));
      flushSync();
      expect(dialog.open).toBe(true);
      // A choice applies at once.
      const lin = dialog.querySelector<HTMLInputElement>('input[value="lin"]')!;
      lin.checked = true;
      lin.dispatchEvent(new Event('change', { bubbles: true }));
      flushSync();
      expect(button.textContent).toContain('EMA 0.9 · Linear');
      done.click();
      flushSync();
      expect(dialog.open).toBe(false);
      expect(button.getAttribute('aria-expanded')).toBe('false');
      expect(document.activeElement).toBe(button);
      unmount(c);
      target.remove();
    });
  });
});
