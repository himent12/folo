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
