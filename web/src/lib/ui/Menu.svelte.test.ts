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
