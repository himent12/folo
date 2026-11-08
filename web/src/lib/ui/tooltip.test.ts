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
});
