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
  if (c) unmount(c);
  c = undefined;
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

  it('nothing auto-dismisses while focus is in the stack', async () => {
    setup();
    const id = toast.show({ tone: 'info', title: 'Hold', timeout: 20 });
    flushSync();
    const close = document.querySelector<HTMLButtonElement>('.toast .close')!;
    close.focus();
    close.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    await new Promise((r) => setTimeout(r, 60));
    expect(toast.items.some((t) => t.id === id)).toBe(true);
  });
});
