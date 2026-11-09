// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { placeTip, tooltip } from './tooltip';

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

  it('renders inside the open modal dialog that holds its anchor, and back in the body outside one', () => {
    vi.useFakeTimers();
    // The dataset inspector and New run's Summary sheet are modal dialogs in the top layer: a tooltip left in <body>
    // would sit behind them (and be inert), so it moves into the dialog and keeps its description and Escape.
    const dialog = document.createElement('dialog');
    dialog.setAttribute('open', '');
    const zoom = document.createElement('button');
    dialog.appendChild(zoom);
    const page = document.createElement('button');
    document.body.append(dialog, page);
    const a = tooltip(zoom, { text: 'Zoom in', keys: '+' });
    const b = tooltip(page, 'Compare runs');
    zoom.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    const tip = document.getElementById('folo-tooltip')!;
    expect(tip.parentElement).toBe(dialog);
    expect(tip.hasAttribute('data-show')).toBe(true);
    expect(zoom.getAttribute('aria-describedby')).toBe('folo-tooltip');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(tip.hasAttribute('data-show')).toBe(false);
    expect(zoom.hasAttribute('aria-describedby')).toBe(false);
    page.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    expect(tip.parentElement).toBe(document.body);
    expect(page.getAttribute('aria-describedby')).toBe('folo-tooltip');
    a.destroy();
    b.destroy();
    dialog.remove();
    page.remove();
  });

  describe('placeTip', () => {
    const view = { left: 4, top: 4, right: 1020, bottom: 764 };
    const size = { width: 240, height: 40 };

    it('centres above the anchor when there is room, the arrow on the anchor', () => {
      const p = placeTip({ left: 400, top: 300, right: 480, bottom: 320 }, size, 'top', view);
      expect(p).toEqual({ x: 320, y: 252, side: 'top', arrow: 120 });
    });

    it('flips below when the top has no room', () => {
      const p = placeTip({ left: 400, top: 10, right: 480, bottom: 30 }, size, 'top', view);
      expect(p.side).toBe('bottom');
      expect(p.y).toBe(38);
    });

    it('shifts inside a rail’s right edge and keeps the arrow on the anchor', () => {
      // The Summary rail ends at 999: an anchor at its right edge (New run's Schedule steps).
      const rail = { ...view, left: 700, right: 999 };
      const anchor = { left: 890, top: 505, right: 983, bottom: 521 };
      const p = placeTip(anchor, size, 'top', rail);
      expect(p.x + size.width).toBe(999);
      expect(p.x + p.arrow).toBe(Math.round((anchor.left + anchor.right) / 2));
    });

    it('keeps the arrow off the corners when the anchor sits at the bounds’ edge', () => {
      const p = placeTip({ left: 1014, top: 300, right: 1020, bottom: 316 }, size, 'top', view);
      expect(p.arrow).toBe(size.width - 8);
    });

    it('moves a side tooltip that fits neither side above the anchor', () => {
      const narrow = { left: 4, top: 4, right: 386, bottom: 840 };
      const p = placeTip({ left: 100, top: 400, right: 290, bottom: 428 }, size, 'right', narrow);
      expect(p.side).toBe('top');
    });
  });
});
