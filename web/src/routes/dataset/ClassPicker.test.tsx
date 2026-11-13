// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
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
});
