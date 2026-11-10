// @vitest-environment happy-dom
// Toasts never take focus, and closing one never leaves focus on <body>.
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Toaster from './Toaster.svelte';

// happy-dom's Web Animations reject when Svelte cancels an outro; the fly is beside the point here.
vi.mock('svelte/transition', () => ({ fly: () => ({}) }));
import { toast } from './toast.svelte';

let c: ReturnType<typeof mount> | undefined;

afterEach(() => {
  vi.useRealTimers();
  if (c) unmount(c);
  c = undefined;
  toast.hold(false);
  for (const t of toast.items) toast.dismiss(t.id);
  document.body.innerHTML = '';
});

function setup() {
  const field = document.createElement('input');
  const target = document.createElement('div');
  document.body.append(field, target);
  c = mount(Toaster, { target });
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
});

describe('Toast timing and announcement', () => {
  it('errors and toasts with an action stay until closed; plain ones time out', () => {
    const plain = toast.info('Saved');
    const err = toast.error(new Error('nope'));
    const act = toast.success('Run finished', undefined, { label: 'Open run', run: () => {} });
    const by = (id: number) => toast.items.find((t) => t.id === id)?.timeout;
    expect(by(plain)).toBeGreaterThan(0);
    expect(by(err)).toBe(0);
    expect(by(act)).toBe(0);
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
