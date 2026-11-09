// @vitest-environment happy-dom
import { flushSync, mount, tick, unmount } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
import Menu from './Menu.svelte';

describe('Menu', () => {
  it('opens from the keyboard and returns focus to its trigger when an action runs', async () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const run = vi.fn();
    const c = mount(Menu, { target, props: { label: 'More actions', items: [{ label: 'Delete', onclick: run }] } });
    const trigger = target.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')!;
    trigger.click();
    flushSync();
    await tick();
    const item = target.querySelector<HTMLElement>('[role="menuitem"]')!;
    expect(document.activeElement).toBe(item);
    item.click();
    flushSync();
    expect(run).toHaveBeenCalledOnce();
    expect(target.querySelector('[role="menu"]')).toBeNull();
    expect(document.activeElement).toBe(trigger);
    unmount(c);
    target.remove();
  });

  it('closes on Escape and refocuses the trigger', async () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(Menu, { target, props: { label: 'More actions', items: [{ label: 'Open', href: '#/x' }] } });
    const trigger = target.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')!;
    trigger.click();
    flushSync();
    await tick();
    target.querySelector('[role="menu"]')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    flushSync();
    expect(target.querySelector('[role="menu"]')).toBeNull();
    expect(document.activeElement).toBe(trigger);
    unmount(c);
    target.remove();
  });
});

describe('Menu order', () => {
  it('puts a destructive item last, under a separator, whatever order it was given in', async () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const items = [
      { label: 'Delete run', danger: true, onclick: () => {} },
      { label: 'Open model', onclick: () => {} },
      { label: 'Try model', onclick: () => {} },
    ];
    const c = mount(Menu, { target, props: { label: 'More run actions', items } });
    target.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')!.click();
    flushSync();
    const list = target.querySelector('[role="menu"]')!;
    expect([...list.querySelectorAll('[role="menuitem"]')].map((e) => e.textContent?.trim())).toEqual(['Open model', 'Try model', 'Delete run']);
    expect(list.querySelector('[role="separator"] + [role="menuitem"]')?.textContent?.trim()).toBe('Delete run');
    unmount(c);
    target.remove();
  });
});

describe('Menu keyboard', () => {
  const items = [{ label: 'Compare' }, { label: 'Delete run' }, { label: 'Duplicate' }, { label: 'Open model' }].map((i) => ({ ...i, onclick: () => {} }));
  const setup = () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const c = mount(Menu, { target, props: { label: 'More actions', items } });
    const trigger = target.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')!;
    const key = (k: string, el: Element = target.querySelector('[role="menu"]')!) =>
      el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }));
    const focused = () => (document.activeElement as HTMLElement | null)?.textContent?.trim();
    return { target, c, trigger, key, focused };
  };

  it('ArrowDown on the button opens on the first item, ArrowUp on the last', async () => {
    const { target, c, trigger, key, focused } = setup();
    trigger.focus();
    key('ArrowDown', trigger);
    flushSync();
    await tick();
    expect(focused()).toBe('Compare');
    key('Escape');
    flushSync();
    key('ArrowUp', trigger);
    flushSync();
    await tick();
    expect(focused()).toBe('Open model');
    unmount(c);
    target.remove();
  });

  it('Home and End go to the ends; a letter moves to the next item starting with it', async () => {
    const { target, c, trigger, key, focused } = setup();
    trigger.click();
    flushSync();
    await tick();
    key('End');
    expect(focused()).toBe('Open model');
    key('Home');
    expect(focused()).toBe('Compare');
    key('d');
    expect(focused()).toBe('Delete run');
    key('d');
    expect(focused()).toBe('Duplicate');
    unmount(c);
    target.remove();
  });
});

describe('Menu on a phone', () => {
  const width = window.innerWidth;
  const viewport = (w: number) => (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: w });

  // Round 24: the queued run's ⋯ opened a popover over its callout's text. On phones a menu is the shared bottom sheet
  // (ui/Dialog), as the loss view's options are; the action still runs and focus comes back to the trigger.
  it('opens as the shared sheet, never a popover, and runs the action', async () => {
    viewport(390);
    const target = document.createElement('div');
    document.body.appendChild(target);
    const run = vi.fn();
    const c = mount(Menu, { target, props: { label: 'More run actions', items: [{ label: 'Remove from queue', onclick: run }] } });
    const trigger = target.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')!;
    trigger.click();
    flushSync();
    await tick();
    expect(target.querySelector('.list')).toBeNull();
    const menu = target.querySelector('dialog [role="menu"]')!;
    expect(menu).not.toBeNull();
    expect(menu.closest('dialog')?.querySelector('h2')?.textContent).toBe('More run actions');
    menu.querySelector<HTMLElement>('[role="menuitem"]')!.click();
    await tick();
    flushSync();
    expect(run).toHaveBeenCalledOnce();
    unmount(c);
    target.remove();
    viewport(width);
  });

  // Review of round 25: every item sat at tabindex -1, so once Tab had gone from an action to the sheet's Close, neither
  // Tab nor Shift+Tab came back to the actions (the modal sheet keeps focus inside). The item last focused stays in
  // the tab order (roving tabindex), so the sheet's tab sequence is Close and that item.
  it('Tab from the sheet’s Close comes back to the actions', async () => {
    viewport(390);
    const target = document.createElement('div');
    document.body.appendChild(target);
    const items = ['Start next', 'Remove from queue', 'Duplicate'].map((label) => ({ label, onclick: () => {} }));
    const c = mount(Menu, { target, props: { label: 'More run actions', items } });
    target.querySelector<HTMLButtonElement>('button[aria-haspopup="menu"]')!.click();
    flushSync();
    await tick();
    const dialog = target.querySelector('dialog')!;
    const menu = dialog.querySelector('[role="menu"]')!;
    const text = (el: Element | null) => el?.textContent?.trim() || el?.getAttribute('aria-label');
    // The modal sheet's tab sequence, in document order: what Tab and Shift+Tab cycle through.
    const sequence = () => [...dialog.querySelectorAll<HTMLElement>('button, a[href]')].filter((el) => el.tabIndex >= 0).map(text);
    expect(text(document.activeElement)).toBe('Start next');
    expect(sequence()).toEqual(['Close', 'Start next']);
    menu.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true }));
    flushSync();
    expect(text(document.activeElement)).toBe('Remove from queue');
    expect(sequence()).toEqual(['Close', 'Remove from queue']);
    // Tab to Close: the action focused last keeps its place, so Tab (or Shift+Tab) from Close lands on it again.
    dialog.querySelector<HTMLElement>('button[aria-label="Close"]')!.focus();
    flushSync();
    expect(sequence()).toEqual(['Close', 'Remove from queue']);
    unmount(c);
    target.remove();
    viewport(width);
  });
});
