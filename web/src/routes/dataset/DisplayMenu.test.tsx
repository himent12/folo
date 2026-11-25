// @vitest-environment happy-dom
// DisplayMenu Escape handling from inside the popover and from its trigger, with focus return and stopPropagation.
import { flush, tick } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import DisplayMenu from './DisplayMenu';

let target: HTMLElement;

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  target.remove();
});

describe('DisplayMenu keyboard accessibility', () => {
  it('closes on Escape from both the popover and the trigger button, returns focus, and stops propagation', async () => {
    const windowEsc = vi.fn();
    const onWinKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') windowEsc();
    };
    window.addEventListener('keydown', onWinKey);
    const view = render(() => <DisplayMenu boxes={true} density="m" onboxes={() => {}} ondensity={() => {}} />, target);
    try {
      const trigger = target.querySelector<HTMLButtonElement>('button.trigger')!;
      trigger.click();
      await tick();
      flush();
      expect(trigger.getAttribute('aria-expanded')).toBe('true');
      const pop = target.querySelector<HTMLElement>('.pop')!;
      expect(pop).not.toBeNull();

      // Escape from inside the popover closes it, returns focus to trigger, and does not bubble to window shortcuts.
      pop.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      flush();
      expect(trigger.getAttribute('aria-expanded')).toBe('false');
      expect(document.activeElement).toBe(trigger);
      expect(windowEsc).not.toHaveBeenCalled();

      // Re-open, Shift+Tab back to trigger, and press Escape on the trigger itself.
      trigger.click();
      await tick();
      flush();
      trigger.focus();
      trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      flush();
      expect(trigger.getAttribute('aria-expanded')).toBe('false');
      expect(document.activeElement).toBe(trigger);
      expect(windowEsc).not.toHaveBeenCalled();
    } finally {
      window.removeEventListener('keydown', onWinKey);
      view.unmount();
    }
  });
});
