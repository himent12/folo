// @vitest-environment happy-dom
import { flush as flushSync, tick } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import ClassPicker from './ClassPicker';

let target: HTMLElement;
let c: { unmount(): void } | undefined;
afterEach(() => {
  c?.unmount();
  target.remove();
});

describe('ClassPicker', () => {
  it('says its numbers are boxes (the filter it sets counts images)', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <ClassPicker classes={['circle', 'square']} counts={[1441, 1]} value={undefined} onchange={() => {}} />, target);
    target.querySelector<HTMLButtonElement>('button[aria-haspopup="listbox"]')!.click();
    flushSync();
    const [all, circle, square] = [...target.querySelectorAll('[role="option"]')].map((o) => o.querySelector('.n')!.textContent!.trim());
    expect(all).toBe('boxes');
    expect(circle).toBe('1,441 boxes');
    expect(square).toBe('1 box');
  });

  it('moves to a class by its first letters, as a select does (studio-data F2: 80 classes)', () => {
    target = document.createElement('div');
    document.body.appendChild(target);
    const classes = ['person', 'tie', 'tennis racket', 'toaster', 'toothbrush', 'teddy bear'];
    c = render(() => <ClassPicker classes={classes} value={undefined} onchange={() => {}} />, target);
    target.querySelector<HTMLButtonElement>('button[aria-haspopup="listbox"]')!.click();
    flushSync();
    const list = target.querySelector<HTMLElement>('[role="listbox"]')!;
    const type = (key: string) => list.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
    const focused = () => document.activeElement?.querySelector('.name')?.textContent;
    vi.useFakeTimers({ toFake: ['Date'] });
    onTestFinished(() => void vi.useRealTimers());
    type('t');
    expect(focused()).toBe('tie');
    type('o');
    type('o');
    expect(focused()).toBe('toothbrush');
    // After a pause, the same letter again steps through the classes that start with it, round to the first.
    for (const want of ['teddy bear', 'tie', 'tennis racket']) {
      vi.advanceTimersByTime(1500);
      type('t');
      expect(focused()).toBe(want);
    }
    // A key an input method is composing is not a letter yet: left to the input method, focus stays.
    vi.advanceTimersByTime(1500);
    const composing = new KeyboardEvent('keydown', { key: 'p', isComposing: true, bubbles: true, cancelable: true });
    list.dispatchEvent(composing);
    expect([composing.defaultPrevented, focused()]).toEqual([false, 'tennis racket']);
    // Closed and opened again at once after a letter, it starts a new word.
    type('t');
    type('Escape');
    target.querySelector<HTMLButtonElement>('button[aria-haspopup="listbox"]')!.click();
    flushSync();
    target.querySelector<HTMLElement>('[role="listbox"]')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'p', bubbles: true }));
    expect(focused()).toBe('person');
  });

  it('opens on ArrowDown, closes on Escape from the trigger without leaking to window, and closes on outside focus', async () => {
    target = document.createElement('div');
    const outside = document.createElement('button');
    outside.textContent = 'Outside';
    document.body.append(target, outside);
    c = render(() => <ClassPicker classes={['circle', 'square']} value={undefined} onchange={() => {}} />, target);
    const trigger = target.querySelector<HTMLButtonElement>('button[aria-haspopup="listbox"]')!;
    const winEsc = vi.fn();
    const onWin = (e: KeyboardEvent) => {
      if (e.key === 'Escape') winEsc(e);
    };
    window.addEventListener('keydown', onWin);
    try {
      trigger.focus();
      trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true }));
      flushSync();
      await tick();
      expect(target.querySelector('[role="listbox"]')).not.toBeNull();

      // ArrowDown from the trigger while already open moves focus into the listbox.
      trigger.focus();
      trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true }));
      flushSync();
      expect(document.activeElement?.getAttribute('role')).toBe('option');

      // Escape while focus is back on trigger closes listbox, refocuses trigger, and stops propagation.
      trigger.focus();
      trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
      flushSync();
      expect(target.querySelector('[role="listbox"]')).toBeNull();
      expect(document.activeElement).toBe(trigger);
      expect(winEsc).not.toHaveBeenCalled();

      // Moving focus to an element outside the picker closes the listbox.
      trigger.click();
      flushSync();
      await tick();
      expect(target.querySelector('[role="listbox"]')).not.toBeNull();
      target.querySelector('.picker')!.dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: outside }));
      flushSync();
      expect(target.querySelector('[role="listbox"]')).toBeNull();

      // Unmounting before tick() resolves after opening does not steal focus.
      outside.focus();
      trigger.click();
      c.unmount();
      c = undefined;
      outside.focus();
      await tick();
      expect(document.activeElement).toBe(outside);
    } finally {
      window.removeEventListener('keydown', onWin);
      outside.remove();
    }
  });
});
