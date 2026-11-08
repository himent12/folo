// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { tooltip } from './tooltip';

describe('tooltip', () => {
  afterEach(() => vi.useRealTimers());

  it('closes on Escape and keeps the element’s own description', () => {
    vi.useFakeTimers();
    const b = document.createElement('button');
    b.setAttribute('aria-describedby', 'hint');
    document.body.appendChild(b);
    const t = tooltip(b, 'Stop the run');
    b.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    expect(b.getAttribute('aria-describedby')).toBe('hint folo-tooltip');
    expect(document.getElementById('folo-tooltip')?.hasAttribute('data-show')).toBe(true);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(document.getElementById('folo-tooltip')?.hasAttribute('data-show')).toBe(false);
    expect(b.getAttribute('aria-describedby')).toBe('hint');
    t.destroy();
    b.remove();
  });

  it('shows for keyboard focus inside a wrapper and describes the focused button', () => {
    vi.useFakeTimers();
    const wrap = document.createElement('span');
    const b = document.createElement('button');
    wrap.appendChild(b);
    document.body.appendChild(wrap);
    const t = tooltip(wrap, { text: 'Zoom in', keys: '+' });
    const visible = vi.spyOn(b, 'matches').mockImplementation((sel: string) => sel === ':focus-visible' || Element.prototype.matches.call(b, sel));
    b.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    vi.advanceTimersByTime(10);
    expect(document.getElementById('folo-tooltip')?.hasAttribute('data-show')).toBe(true);
    expect(b.getAttribute('aria-describedby')).toBe('folo-tooltip');
    expect(wrap.hasAttribute('aria-describedby')).toBe(false);
    b.dispatchEvent(new FocusEvent('focusout', { bubbles: true }));
    expect(document.getElementById('folo-tooltip')?.hasAttribute('data-show')).toBe(false);
    expect(b.hasAttribute('aria-describedby')).toBe(false);
    visible.mockRestore();
    t.destroy();
    wrap.remove();
  });

  it('describes one element at a time: hover the first button, focus the second, destroy', () => {
    vi.useFakeTimers();
    const wrap = document.createElement('span');
    const a = document.createElement('button');
    const b = document.createElement('button');
    a.setAttribute('aria-describedby', 'hint-a');
    wrap.append(a, b);
    document.body.appendChild(wrap);
    const t = tooltip(wrap, 'Zoom');
    wrap.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    expect(a.getAttribute('aria-describedby')).toBe('hint-a folo-tooltip');
    const visible = vi.spyOn(b, 'matches').mockImplementation((sel: string) => sel === ':focus-visible' || Element.prototype.matches.call(b, sel));
    b.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    vi.advanceTimersByTime(10);
    expect(a.getAttribute('aria-describedby')).toBe('hint-a');
    expect(b.getAttribute('aria-describedby')).toBe('folo-tooltip');
    t.destroy();
    expect(a.getAttribute('aria-describedby')).toBe('hint-a');
    expect(b.hasAttribute('aria-describedby')).toBe(false);
    visible.mockRestore();
    wrap.remove();
  });

  it('stays while the pointer moves from the trigger onto the tooltip, and hides after it leaves both', () => {
    vi.useFakeTimers();
    const b = document.createElement('button');
    document.body.appendChild(b);
    const t = tooltip(b, 'Stop the run');
    b.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    const tip = document.getElementById('folo-tooltip')!;
    expect(tip.hasAttribute('data-show')).toBe(true);
    b.dispatchEvent(new PointerEvent('pointerleave'));
    vi.advanceTimersByTime(50);
    tip.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    expect(tip.hasAttribute('data-show')).toBe(true);
    tip.dispatchEvent(new PointerEvent('pointerleave'));
    vi.advanceTimersByTime(200);
    expect(tip.hasAttribute('data-show')).toBe(false);
    t.destroy();
    b.remove();
  });

  it('never opens when the pointer leaves before the delay is up', () => {
    vi.useFakeTimers();
    vi.setSystemTime(Date.now() + 60_000); // well past the quick re-show window after the tests above
    const b = document.createElement('button');
    document.body.appendChild(b);
    const t = tooltip(b, 'Compare runs');
    b.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(50);
    b.dispatchEvent(new PointerEvent('pointerleave'));
    vi.advanceTimersByTime(550);
    expect(document.getElementById('folo-tooltip')?.hasAttribute('data-show') ?? false).toBe(false);
    t.destroy();
    b.remove();
  });
});
