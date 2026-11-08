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
});
