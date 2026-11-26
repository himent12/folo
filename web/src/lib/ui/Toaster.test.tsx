// @vitest-environment happy-dom
// Toasts never take focus, and closing one never leaves focus on <body>.
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import Toaster from './Toaster';

// happy-dom's Web Animations reject when an outro is cancelled; the fly is beside the point here.
vi.mock('fw', async (orig) => ({ ...(await orig<typeof import('fw')>()), fly: () => ({}) }));
import { toast, TOAST_MS } from './toast';

let c: Rendered | undefined;

afterEach(() => {
  vi.useRealTimers();
  c?.unmount();
  c = undefined;
  toast.hold(false);
  for (const t of toast.items) toast.dismiss(t.id);
  document.body.innerHTML = '';
});

function setup() {
  const field = document.createElement('input');
  const target = document.createElement('div');
  document.body.append(field, target);
  c = render(() => <Toaster />, target);
  field.focus();
  return field;
}

describe('Toaster and focus', () => {
  it('showing and auto-closing a toast leaves focus where it was', () => {
    const field = setup();
    const id = toast.info('Saved');
    flushSync();
    expect(document.querySelector('.toast')).not.toBeNull();
    expect(document.activeElement).toBe(field);
    toast.dismiss(id);
    flushSync();
    expect(document.activeElement).toBe(field);
  });

  it('a mouse press on its buttons does not move focus into the toast', () => {
    setup();
    toast.error(new Error('boom'), 'Prediction failed');
    flushSync();
    const close = document.querySelector<HTMLButtonElement>('.toast .close')!;
    const down = new MouseEvent('mousedown', { bubbles: true, cancelable: true });
    close.dispatchEvent(down);
    expect(down.defaultPrevented).toBe(true);
  });

  it('closing the toast that holds keyboard focus hands focus back, not to <body>', () => {
    const field = setup();
    toast.info('Copy failed', 'Your browser blocked clipboard access.');
    flushSync();
    const close = document.querySelector<HTMLButtonElement>('.toast .close')!;
    close.focus();
    // happy-dom does not fill relatedTarget on focus moves: tell the toaster where focus came from, as a browser does.
    close.dispatchEvent(new FocusEvent('focusin', { bubbles: true, relatedTarget: field }));
    close.click();
    flushSync();
    expect(document.activeElement).toBe(field);
  });

  it('nothing auto-dismisses while focus is in the stack', () => {
    vi.useFakeTimers();
    setup();
    const id = toast.show({ tone: 'info', title: 'Hold', timeout: 20 });
    flushSync();
    const close = document.querySelector<HTMLButtonElement>('.toast .close')!;
    close.focus();
    close.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    vi.advanceTimersByTime(60);
    expect(toast.items.some((t) => t.id === id)).toBe(true);
  });

  it('a toast shown while the stack is held waits too', () => {
    vi.useFakeTimers();
    setup();
    toast.show({ tone: 'info', title: 'First', timeout: 0 });
    flushSync();
    const close = document.querySelector<HTMLButtonElement>('.toast .close')!;
    close.focus();
    close.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    const later = toast.show({ tone: 'info', title: 'Later', timeout: 20 });
    vi.advanceTimersByTime(60);
    expect(toast.items.some((t) => t.id === later)).toBe(true);
  });

  it('closing another toast by mouse leaves keyboard focus in the one that holds it', () => {
    const field = setup();
    toast.show({ tone: 'info', title: 'A', timeout: 0 });
    toast.show({ tone: 'info', title: 'B', timeout: 0 });
    flushSync();
    const [closeA, closeB] = [...document.querySelectorAll<HTMLButtonElement>('.toast .close')];
    closeA!.focus();
    closeA!.dispatchEvent(new FocusEvent('focusin', { bubbles: true, relatedTarget: field }));
    closeB!.click();
    flushSync();
    expect(document.activeElement).toBe(closeA);
  });

  it('releases hold and restores focus when a focused toast is dismissed programmatically', () => {
    vi.useFakeTimers();
    const field = setup();
    const id = toast.show({ tone: 'info', title: 'Scoped', key: 'op', timeout: 50 });
    const other = toast.show({ tone: 'info', title: 'Other', timeout: 50 });
    flushSync();
    const close = document.querySelector<HTMLButtonElement>(`[data-toast="${id}"] .close`)!;
    close.focus();
    close.dispatchEvent(new FocusEvent('focusin', { bubbles: true, relatedTarget: field }));
    expect(document.activeElement).toBe(close);

    // Programmatic clear removes the focused toast without firing focusout.
    toast.clear('op');
    flushSync();
    expect(document.activeElement).toBe(field);
    // Hold must be released so the remaining toast resumes auto-dismissing.
    vi.advanceTimersByTime(3000);
    expect(toast.items.some((t) => t.id === other)).toBe(false);
  });

  it('releases toast hold on unmount', () => {
    vi.useFakeTimers();
    setup();
    const id = toast.show({ tone: 'info', title: 'Hold me', timeout: 50 });
    flushSync();
    const close = document.querySelector<HTMLButtonElement>('.toast .close')!;
    close.focus();
    close.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    c!.unmount();
    c = undefined;
    vi.advanceTimersByTime(3000);
    expect(toast.items.some((t) => t.id === id)).toBe(false);
  });
});

describe('Toast timing and announcement', () => {
  it('every toast expires: a note soon, one with an action later, an error later still (models F6)', () => {
    vi.useFakeTimers();
    const plain = toast.info('Saved');
    const act = toast.success('Run finished', undefined, { label: 'Open run', run: () => {} });
    const err = toast.error(new Error('nope'), 'Prediction failed');
    const by = (id: number) => toast.items.find((t) => t.id === id)?.timeout;
    expect(by(plain)! < by(act)! && by(act)! < by(err)!).toBe(true);
    const shown = () => toast.items.map((t) => t.id);
    vi.advanceTimersByTime(TOAST_MS.note);
    expect(shown()).toEqual([act, err]);
    vi.advanceTimersByTime(TOAST_MS.action - TOAST_MS.note);
    expect(shown()).toEqual([err]);
    vi.advanceTimersByTime(TOAST_MS.error - TOAST_MS.action);
    expect(shown()).toEqual([]);
  });

  it("the same action's next report takes an error's place, and the same toast twice is shown once", () => {
    toast.error(new Error('Unsupported image'), 'Prediction failed', { key: 'predict' });
    toast.error(new Error('Unsupported image'), 'Prediction failed', { key: 'predict' });
    expect(toast.items.map((t) => t.title)).toEqual(['Prediction failed']);
    toast.success('Detected 3 objects', undefined, undefined, { key: 'predict' });
    expect(toast.items.map((t) => t.title)).toEqual(['Detected 3 objects']);
    toast.info('Copied');
    toast.info('Copied');
    toast.info('Copied');
    expect(toast.items.map((t) => t.title)).toEqual(['Detected 3 objects', 'Copied']);
  });

  it('each toast is its own live region; the list is not one', () => {
    setup();
    toast.info('Saved');
    toast.error(new Error('nope'));
    flushSync();
    expect(document.querySelector('.toaster ol')?.hasAttribute('aria-live')).toBe(false);
    expect([...document.querySelectorAll('.toast')].map((t) => t.getAttribute('role'))).toEqual(['status', 'alert']);
  });
});
