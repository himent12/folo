// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { boundsFor, placeTip, tooltip } from './tooltip';

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

  it('does not dismiss an open or pending tooltip when an unrelated tooltip anchor unmounts', () => {
    vi.useFakeTimers();
    vi.setSystemTime(Date.now() + 60_000);
    const active = document.createElement('button');
    const other = document.createElement('button');
    document.body.append(active, other);
    const tActive = tooltip(active, 'Keep open');
    const tOther = tooltip(other, 'Unrelated');

    // Pending timer on `active` survives `other.destroy()`.
    active.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(100);
    tOther.destroy();
    other.remove();
    vi.advanceTimersByTime(400);
    const tip = document.getElementById('folo-tooltip')!;
    expect(tip.hasAttribute('data-show')).toBe(true);
    expect(tip.textContent).toBe('Keep open');
    expect(active.getAttribute('aria-describedby')).toBe('folo-tooltip');

    // A second unrelated anchor mounting and unmounting while `active` is visible leaves it open.
    const another = document.createElement('button');
    document.body.appendChild(another);
    const tAnother = tooltip(another, 'Transient');
    tAnother.destroy();
    another.remove();
    expect(tip.hasAttribute('data-show')).toBe(true);
    expect(active.getAttribute('aria-describedby')).toBe('folo-tooltip');

    // Destroying the active anchor itself dismisses the tooltip and releases its description.
    tActive.destroy();
    expect(tip.hasAttribute('data-show')).toBe(false);
    expect(active.hasAttribute('aria-describedby')).toBe(false);
    active.remove();
  });

  it('ignores update(null) on unrelated anchors and dismisses when the active anchor clears', () => {
    vi.useFakeTimers();
    const active = document.createElement('button');
    const other = document.createElement('button');
    document.body.append(active, other);
    const tActive = tooltip(active, 'Active tip');
    const tOther = tooltip(other, 'Other tip');
    active.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    const tip = document.getElementById('folo-tooltip')!;
    expect(tip.hasAttribute('data-show')).toBe(true);

    tOther.update(null);
    expect(tip.hasAttribute('data-show')).toBe(true);
    expect(active.getAttribute('aria-describedby')).toBe('folo-tooltip');

    tActive.update(null);
    expect(tip.hasAttribute('data-show')).toBe(false);
    expect(active.hasAttribute('aria-describedby')).toBe(false);
    tActive.destroy();
    tOther.destroy();
    active.remove();
    other.remove();
  });

  it('repaints an already-open tooltip live when update() receives new options', () => {
    vi.useFakeTimers();
    const b = document.createElement('button');
    document.body.appendChild(b);
    const t = tooltip(b, { text: 'Theme: Midnight (dark)', side: 'bottom' });
    b.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    const tip = document.getElementById('folo-tooltip')!;
    expect(tip.hasAttribute('data-show')).toBe(true);
    expect(tip.textContent).toBe('Theme: Midnight (dark)');

    t.update({ text: 'Theme: Light', side: 'bottom' });
    expect(tip.hasAttribute('data-show')).toBe(true);
    expect(tip.textContent).toBe('Theme: Light');
    expect(b.getAttribute('aria-describedby')).toBe('folo-tooltip');

    t.destroy();
    b.remove();
  });

  it('dismisses on ancestor or window scroll, ignores unrelated sibling scrollers, and removes the exact listener', () => {
    vi.useFakeTimers();
    const scroller = document.createElement('div');
    const sibling = document.createElement('div');
    const b = document.createElement('button');
    scroller.appendChild(b);
    document.body.append(scroller, sibling);
    const addSpy = vi.spyOn(window, 'addEventListener');
    const removeSpy = vi.spyOn(window, 'removeEventListener');
    const t = tooltip(b, 'Scroll away');

    b.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    const tip = document.getElementById('folo-tooltip')!;
    expect(tip.hasAttribute('data-show')).toBe(true);
    const scrollAdd = addSpy.mock.calls.find(([type]) => type === 'scroll');
    expect(scrollAdd?.[2]).toEqual({ capture: true, passive: true });
    const scrollListener = scrollAdd?.[1];
    expect(typeof scrollListener).toBe('function');

    // Unrelated sibling container scroll (e.g. live streaming logs) must not dismiss the tooltip.
    sibling.dispatchEvent(new Event('scroll', { bubbles: false }));
    expect(tip.hasAttribute('data-show')).toBe(true);
    expect(b.getAttribute('aria-describedby')).toBe('folo-tooltip');

    // Ancestor container scroll does not bubble, so capture-phase listening dismisses it and removes the exact callback.
    scroller.dispatchEvent(new Event('scroll', { bubbles: false }));
    expect(tip.hasAttribute('data-show')).toBe(false);
    expect(b.hasAttribute('aria-describedby')).toBe(false);
    expect(removeSpy).toHaveBeenCalledWith('scroll', scrollListener, true);

    // Window scroll also dismisses a visible tooltip.
    b.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    expect(tip.hasAttribute('data-show')).toBe(true);
    window.dispatchEvent(new Event('scroll'));
    expect(tip.hasAttribute('data-show')).toBe(false);

    addSpy.mockRestore();
    removeSpy.mockRestore();
    t.destroy();
    scroller.remove();
    sibling.remove();
  });

  it('cancels a queued handoff on scroll and keeps handoff descriptions and grace timers clean', () => {
    vi.useFakeTimers();
    const a = document.createElement('button');
    const b = document.createElement('button');
    a.setAttribute('aria-describedby', 'hint-a');
    document.body.append(a, b);
    const tA = tooltip(a, 'First');
    const tB = tooltip(b, 'Second');

    // Scroll mid-handoff: A is visible, pointer enters B (0 ms timer queued), window scrolls before timer fires.
    a.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    const tip = document.getElementById('folo-tooltip')!;
    expect(tip.hasAttribute('data-show')).toBe(true);
    b.dispatchEvent(new PointerEvent('pointerenter'));
    window.dispatchEvent(new Event('scroll'));
    vi.advanceTimersByTime(500);
    expect(tip.hasAttribute('data-show')).toBe(false);
    expect(a.getAttribute('aria-describedby')).toBe('hint-a');
    expect(b.hasAttribute('aria-describedby')).toBe(false);

    // Direct handoff: A's own aria-describedby survives, B gets folo-tooltip, and A's grace timer does not dismiss B.
    a.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    expect(a.getAttribute('aria-describedby')).toBe('hint-a folo-tooltip');
    a.dispatchEvent(new PointerEvent('pointerleave'));
    b.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    expect(tip.hasAttribute('data-show')).toBe(true);
    expect(tip.textContent).toBe('Second');
    expect(a.getAttribute('aria-describedby')).toBe('hint-a');
    expect(b.getAttribute('aria-describedby')).toBe('folo-tooltip');

    // Dismissing while hovered on the tooltip element then entering B must not let a stale hideTimer close B.
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    tip.dispatchEvent(new PointerEvent('pointerleave'));
    b.dispatchEvent(new PointerEvent('pointerenter'));
    vi.advanceTimersByTime(500);
    expect(tip.hasAttribute('data-show')).toBe(true);
    expect(b.getAttribute('aria-describedby')).toBe('folo-tooltip');

    tA.destroy();
    tB.destroy();
    a.remove();
    b.remove();
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

    it('keeps 8 px from the viewport and treats fixed chrome as outside: near the top bar it flips below', () => {
      // Round 24: the Playground's "no runtime" tip sat over the top bar's New run, touching the viewport's edge.
      const rect = (b: Partial<DOMRect>) => () => ({ left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0, x: 0, y: 0, toJSON() {}, ...b }) as DOMRect;
      const bar = document.createElement('header');
      bar.setAttribute('data-tip-chrome', '');
      bar.getBoundingClientRect = rect({ left: 0, top: 0, right: 1024, bottom: 48, width: 1024, height: 48 });
      const anchor = document.createElement('button');
      anchor.getBoundingClientRect = rect({ left: 960, top: 72, right: 1000, bottom: 100, width: 40, height: 28 });
      document.body.append(bar, anchor);
      const b = boundsFor(anchor, 240);
      expect(b.top).toBe(56);
      expect(b.right).toBe((document.documentElement.clientWidth || window.innerWidth) - 8);
      const p = placeTip(anchor.getBoundingClientRect(), { width: 240, height: 40 }, 'top', b);
      expect(p.side).toBe('bottom');
      expect(p.x + 240).toBeLessThanOrEqual(b.right);
      bar.remove();
      anchor.remove();
    });

    it('moves a side tooltip that fits neither side above the anchor', () => {
      const narrow = { left: 4, top: 4, right: 386, bottom: 840 };
      const p = placeTip({ left: 100, top: 400, right: 290, bottom: 428 }, size, 'right', narrow);
      expect(p.side).toBe('top');
    });
  });
});
