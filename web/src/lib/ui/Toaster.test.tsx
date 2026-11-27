// @vitest-environment happy-dom
// Toasts never take focus, and closing one never leaves focus on <body>.
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import Toaster from './Toaster';

// happy-dom's Web Animations reject when an outro is cancelled; the fly is beside the point except when testing outros.
const flyState = vi.hoisted(() => ({ duration: 0 }));
vi.mock('fw', async (orig) => ({ ...(await orig<typeof import('fw')>()), fly: () => (flyState.duration ? { duration: flyState.duration } : {}) }));
import { toast, TOAST_MS } from './toast';

let c: Rendered | undefined;

afterEach(() => {
  flyState.duration = 0;
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

  it('releases hold and restores focus immediately on programmatic dismissal even while the outro keeps the row attached', () => {
    vi.useFakeTimers();
    flyState.duration = 160;
    const field = setup();
    const id = toast.show({ tone: 'info', title: 'Outro', key: 'op', timeout: 50 });
    const other = toast.show({ tone: 'info', title: 'Other', timeout: 50 });
    flushSync();
    const row = document.querySelector<HTMLLIElement>(`[data-toast="${id}"]`)!;
    // Keep the row attached in the DOM during its outro animation (as browsers do for 160 ms).
    row.animate = (() => ({ onfinish: null, cancel() {}, finished: Promise.resolve(), currentTime: 0 })) as unknown as typeof row.animate;
    const close = row.querySelector<HTMLButtonElement>('.close')!;
    close.focus();
    close.dispatchEvent(new FocusEvent('focusin', { bubbles: true, relatedTarget: field }));
    expect(document.activeElement).toBe(close);

    toast.clear('op');
    flushSync();
    // Row is still attached while outro runs, yet focus must already return to `field` and hold must release.
    expect(row.isConnected).toBe(true);
    expect(document.activeElement).toBe(field);
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

  it('pauses and resumes remaining time on hover instead of cutting a fresh error toast to 2.5 s', async () => {
    vi.useFakeTimers();
    setup();
    await Promise.resolve();
    const id = toast.error(new Error('boom'), 'Run failed');
    flushSync();
    const section = document.querySelector<HTMLElement>('.toaster')!;
    // 100 ms elapse, user hovers for 400 ms, then leaves.
    vi.advanceTimersByTime(100);
    section.dispatchEvent(new PointerEvent('pointerenter', { bubbles: false }));
    vi.advanceTimersByTime(400);
    section.dispatchEvent(new PointerEvent('pointerleave', { bubbles: false }));

    // After 2.5 s (and even 15 s) the fresh 20 s error toast must still be visible.
    vi.advanceTimersByTime(15_000);
    expect(toast.items.some((t) => t.id === id)).toBe(true);

    // Dismisses once the remaining ~19.9 s elapse.
    vi.advanceTimersByTime(4900);
    expect(toast.items.some((t) => t.id === id)).toBe(false);
  });

  it('keeps at least a 2.5 s reading grace window when unhovering a nearly-expired toast', async () => {
    vi.useFakeTimers();
    setup();
    await Promise.resolve();
    const id = toast.info('Saved');
    flushSync();
    const section = document.querySelector<HTMLElement>('.toaster')!;
    // 5.5 s of 6 s elapse, then hover and leave.
    vi.advanceTimersByTime(5500);
    section.dispatchEvent(new PointerEvent('pointerenter', { bubbles: false }));
    vi.advanceTimersByTime(500);
    section.dispatchEvent(new PointerEvent('pointerleave', { bubbles: false }));

    vi.advanceTimersByTime(2000);
    expect(toast.items.some((t) => t.id === id)).toBe(true);
    vi.advanceTimersByTime(600);
    expect(toast.items.some((t) => t.id === id)).toBe(false);
  });

  it('publishes --toast-h while a toast is visible and clears it on dismissal and unmount', () => {
    setup();
    const section = document.querySelector<HTMLElement>('.toaster')!;
    Object.defineProperty(section, 'offsetHeight', { configurable: true, value: 70 });

    const id = toast.success('JSON exported', 'Saved Model A/B comparison JSON.');
    flushSync();
    expect(document.documentElement.style.getPropertyValue('--toast-h')).toBe('86px');

    toast.dismiss(id);
    flushSync();
    expect(document.documentElement.style.getPropertyValue('--toast-h')).toBe('');

    toast.info('Still open');
    flushSync();
    expect(document.documentElement.style.getPropertyValue('--toast-h')).toBe('86px');
    c!.unmount();
    c = undefined;
    expect(document.documentElement.style.getPropertyValue('--toast-h')).toBe('');
  });
});
