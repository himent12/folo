// @vitest-environment happy-dom
import { flushSync, mount, tick, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';
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
});
