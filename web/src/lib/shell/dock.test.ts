// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest';
import { dock, holdDock } from './dock';

const docked = () => document.documentElement.hasAttribute('data-dock');

describe('dock', () => {
  it('marks the root while a dock is held, and holds nest', () => {
    expect(docked()).toBe(false);
    const a = holdDock();
    const b = holdDock();
    expect(docked()).toBe(true);
    a();
    expect(docked()).toBe(true);
    b();
    expect(docked()).toBe(false);
  });

  it('releases once, however often the release is called', () => {
    const a = holdDock();
    const b = holdDock();
    a();
    a();
    expect(docked()).toBe(true);
    b();
    expect(docked()).toBe(false);
  });

  it('use:dock holds while the element is mounted and clears its published height after', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const action = dock(el);
    expect(docked()).toBe(true);
    expect(document.documentElement.style.getPropertyValue('--dock-h')).toMatch(/px$/);
    action.destroy();
    expect(docked()).toBe(false);
    expect(document.documentElement.style.getPropertyValue('--dock-h')).toBe('');
    el.remove();
  });
});
