// @vitest-environment happy-dom
// Dialog focus return on normal close and on conditional unmount while open.
import { flush, signal } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Dialog from './Dialog';

let target: HTMLElement;

beforeEach(() => {
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
  HTMLDialogElement.prototype.close ??= function (this: HTMLDialogElement) {
    this.open = false;
  };
  window.matchMedia ??= (() => ({ matches: false, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  target.remove();
  document.body.innerHTML = '';
});

describe('Dialog focus return', () => {
  it('returns focus to the opener when unmounted while open without invoking onclose', () => {
    const opener = document.createElement('button');
    opener.textContent = 'Open';
    document.body.appendChild(opener);
    opener.focus();
    expect(document.activeElement).toBe(opener);

    const open = signal(true);
    const onclose = vi.fn();
    const view = render(() => <Dialog open={open.get()} title="Confirm" onclose={onclose} />, target);
    flush();

    const dialog = target.querySelector<HTMLDialogElement>('dialog')!;
    expect(dialog.open).toBe(true);
    expect(document.activeElement).toBe(dialog);

    view.unmount();
    expect(onclose).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(opener);
  });
});
