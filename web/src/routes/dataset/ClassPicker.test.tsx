// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, describe, expect, it } from 'vitest';
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
});
