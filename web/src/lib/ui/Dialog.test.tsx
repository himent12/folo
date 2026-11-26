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
  it('returns focus to the opener on normal close and invokes onclose', () => {
    const opener = document.createElement('button');
    opener.textContent = 'Open';
    document.body.appendChild(opener);
    opener.focus();

    const open = signal(true);
    const onclose = vi.fn();
    const view = render(() => <Dialog open={open.get()} title="Confirm" onclose={onclose} />, target);
    flush();

    const dialog = target.querySelector<HTMLDialogElement>('dialog')!;
    expect(dialog.open).toBe(true);
    expect(document.activeElement).toBe(dialog);

    dialog.open = false;
    dialog.dispatchEvent(new Event('close'));
    flush();
    expect(onclose).toHaveBeenCalledTimes(1);
    expect(document.activeElement).toBe(opener);
    view.unmount();
  });

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

  it('closes and notifies onclose synchronously on cancel and Close click before an async native close event fires', () => {
    const opener = document.createElement('button');
    opener.textContent = 'Open';
    document.body.appendChild(opener);
    opener.focus();

    const open = signal(true);
    const onclose = vi.fn(() => open.set(false));
    const view = render(() => <Dialog open={open.get()} title="Confirm" onclose={onclose} />, target);
    flush();

    const dialog = target.querySelector<HTMLDialogElement>('dialog')!;
    expect(dialog.open).toBe(true);

    // Simulate Chromium Escape: `cancel` fires synchronously, while `close` is queued for a later task.
    const cancel = new Event('cancel', { cancelable: true });
    dialog.dispatchEvent(cancel);
    expect(cancel.defaultPrevented).toBe(true);
    expect(dialog.open).toBe(false);
    expect(onclose).toHaveBeenCalledTimes(1);
    expect(document.activeElement).toBe(opener);

    // Late queued `close` event must not fire `onclose` a second time.
    dialog.dispatchEvent(new Event('close'));
    expect(onclose).toHaveBeenCalledTimes(1);

    // Reopen immediately and dismiss via Close button: synchronous close + no duplicate on late `close` event.
    open.set(true);
    flush();
    expect(dialog.open).toBe(true);
    const closeBtn = dialog.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!;
    closeBtn.click();
    expect(dialog.open).toBe(false);
    expect(onclose).toHaveBeenCalledTimes(2);
    expect(document.activeElement).toBe(opener);

    dialog.dispatchEvent(new Event('close'));
    expect(onclose).toHaveBeenCalledTimes(2);
    view.unmount();
  });
});
