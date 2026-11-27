// `use:tooltip={'Text'}` or `use:tooltip={{ text, keys: 'G R', side: 'right' }}`: a small tooltip on hover (after
// a short delay) and on keyboard focus. One shared element, positioned with fixed coordinates, so it never clips.
// It renders in the anchor's own context: inside the open modal dialog that holds the anchor (the dataset inspector,
// New run's Summary sheet), as a manual popover in the top layer, so it is never drawn behind the dialog and stays
// hoverable there (everything outside a modal dialog is inert); elsewhere in <body>.
// The node may be a wrapper (a span around a component's button): focus inside it shows the tooltip, and the
// description goes on the focusable element, which is what assistive technology reads. A tooltip shown by hover stays
// while the pointer moves onto it (WCAG 1.4.13, hoverable): leaving the trigger or the tooltip hides it after a short
// grace period, unless the pointer arrives on the other. Placement is collision-aware (`placeTip`): the tooltip flips
// to the other side when its own does not fit, shifts along the anchor to stay inside the anchor's panel (a rail's
// edge) and the viewport, 8 px in from its edges, and its arrow keeps pointing at the anchor's centre. Fixed chrome (the
// top bar, the tab bar, a dock: `data-tip-chrome`) is outside the bounds, so a tooltip near the top of a page flips
// below its anchor rather than land on the top bar's buttons (round 24's "no runtime" tip over New run).

import { formatKeys } from '$lib/keys';

export type Side = 'top' | 'bottom' | 'right' | 'left';
export type TooltipOptions = string | { text: string; keys?: string; side?: Side } | null | undefined;

interface Box {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** Space between the anchor and the tooltip (the arrow sits in it), and the margin kept from the viewport's edges. */
const GAP = 8;
export const EDGE = 8;
/** The arrow never comes closer to a corner than this, so it never sits on the rounded radius. */
const ARROW_INSET = 8;

/**
 * Where a tooltip of `size` goes next to `anchor`: on `side` when it fits inside `bounds`, else on the opposite side,
 * else on whichever of the two has more room; then shifted along the anchor to stay inside `bounds`. `arrow` is the
 * arrow's offset along the tooltip's edge (from its left for top / bottom, from its top for left / right), at the
 * anchor's centre.
 */
export function placeTip(anchor: Box, size: { width: number; height: number }, side: Side, bounds: Box) {
  const room: Record<Side, number> = {
    top: anchor.top - bounds.top,
    bottom: bounds.bottom - anchor.bottom,
    left: anchor.left - bounds.left,
    right: bounds.right - anchor.right,
  };
  const need = (s: Side) => (s === 'top' || s === 'bottom' ? size.height : size.width) + GAP;
  const opposite: Record<Side, Side> = { top: 'bottom', bottom: 'top', left: 'right', right: 'left' };
  let s = side;
  if (room[s] < need(s)) {
    const o = opposite[s];
    if (room[o] >= need(o)) s = o;
    // Neither horizontal side fits (a narrow screen): above or below, where there is more room.
    else if (s === 'left' || s === 'right') s = room.top >= need('top') || room.top > room.bottom ? 'top' : 'bottom';
    else if (room[o] > room[s]) s = o;
  }
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  const cx = (anchor.left + anchor.right) / 2;
  const cy = (anchor.top + anchor.bottom) / 2;
  let x: number;
  let y: number;
  if (s === 'top' || s === 'bottom') {
    x = clamp(cx - size.width / 2, bounds.left, bounds.right - size.width);
    y = s === 'top' ? anchor.top - GAP - size.height : anchor.bottom + GAP;
  } else {
    x = s === 'left' ? anchor.left - GAP - size.width : anchor.right + GAP;
    y = clamp(cy - size.height / 2, bounds.top, bounds.bottom - size.height);
  }
  const arrow = s === 'top' || s === 'bottom' ? clamp(cx - x, ARROW_INSET, size.width - ARROW_INSET) : clamp(cy - y, ARROW_INSET, size.height - ARROW_INSET);
  return { x: Math.round(x), y: Math.round(y), side: s, arrow: Math.round(arrow) };
}

/**
 * The box a tooltip (or ui/InfoButton's popover) stays inside: the viewport, narrowed to the anchor's panel (a rail, a
 * card) when the tooltip fits across it, so it never runs past the panel's edge onto the canvas beside it.
 */
export function boundsFor(node: HTMLElement, width: number): Box {
  const vw = document.documentElement.clientWidth || window.innerWidth;
  const vh = window.innerHeight;
  const view = { left: EDGE, top: EDGE, right: vw - EDGE, bottom: vh - EDGE };
  // Fixed chrome above or below the anchor is not room: the tooltip keeps the same margin from it.
  const a = node.getBoundingClientRect();
  for (const c of document.querySelectorAll<HTMLElement>('[data-tip-chrome]')) {
    if (c.contains(node)) continue;
    const r = c.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (r.bottom <= a.top) view.top = Math.max(view.top, r.bottom + EDGE);
    else if (r.top >= a.bottom) view.bottom = Math.min(view.bottom, r.top - EDGE);
  }
  const panel = node.closest<HTMLElement>('[data-tip-bounds], section.panel, [role="dialog"], dialog');
  if (!panel) return view;
  const p = panel.getBoundingClientRect();
  const left = Math.max(view.left, p.left);
  const right = Math.min(view.right, p.right);
  return right - left >= width ? { ...view, left, right } : view;
}

let tip: HTMLDivElement | null = null;
let showTimer: ReturnType<typeof setTimeout> | undefined;
let hideTimer: ReturnType<typeof setTimeout> | undefined;
let lastHide = 0;
/** Anchor whose tooltip is currently visible, so unrelated unmounts never dismiss it. */
let activeAnchor: HTMLElement | null = null;
/** Anchor with a pending show timer, so its teardown never cancels another trigger's timer. */
let pendingAnchor: HTMLElement | null = null;
/** Hides the tooltip currently shown (set by the trigger that showed it). */
let hideCurrent: (() => void) | null = null;
/** Time the pointer has to cross from a trigger to its tooltip (8 px apart, the arrow between) or back. */
const GRACE_MS = 150;

const leave = () => {
  if (!activeAnchor) return;
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => hideCurrent?.(), GRACE_MS);
};

/** Whether this browser has the Popover API (the top layer); without it the tooltip relies on its z-index. */
export const popovers = () => typeof HTMLElement !== 'undefined' && 'showPopover' in HTMLElement.prototype;

/**
 * The shared element, moved into `host` (the open dialog that holds the anchor, else <body>) and shown there. A modal
 * dialog sits in the top layer and makes the rest of the page inert, so a tooltip outside it would be drawn behind it
 * and could not be hovered: inside it, as a popover, it is drawn above the dialog and above its clipping edge.
 */
function mount(host: HTMLElement): HTMLDivElement {
  if (!tip) {
    tip = document.createElement('div');
    tip.className = 'folo-tooltip';
    tip.setAttribute('role', 'tooltip');
    tip.id = 'folo-tooltip';
    tip.hidden = true;
    if (popovers()) tip.popover = 'manual';
    tip.addEventListener('pointerenter', () => clearTimeout(hideTimer));
    tip.addEventListener('pointerleave', leave);
  }
  tip.hidden = false;
  // Moving an open popover hides it; it is shown again below.
  if (tip.parentElement !== host) host.appendChild(tip);
  if (popovers() && !tip.matches(':popover-open')) tip.showPopover();
  return tip;
}

function norm(o: TooltipOptions) {
  if (!o) return null;
  return typeof o === 'string' ? { text: o, side: 'top' as const } : { side: 'top' as const, ...o };
}

const FOCUSABLE = 'button, a[href], input, select, textarea, [tabindex]';

export function tooltip(node: HTMLElement, opts: TooltipOptions) {
  let o = norm(opts);
  /** The element the tooltip describes: the node, or the focusable element it wraps. */
  let described: HTMLElement = node;
  const target = () => (node.matches(FOCUSABLE) ? node : (node.querySelector<HTMLElement>(FOCUSABLE) ?? node));
  // The element's own aria-describedby (if any) is kept: the tooltip id is added to it, then removed again.
  const own = (el: HTMLElement) => (el.getAttribute('aria-describedby') ?? '').split(/\s+/).filter((id) => id && id !== 'folo-tooltip');
  /** Takes the tooltip id off an element's description, keeping its own ids. */
  const release = (el: HTMLElement) => {
    const rest = own(el);
    if (rest.length) el.setAttribute('aria-describedby', rest.join(' '));
    else el.removeAttribute('aria-describedby');
  };
  // Escape dismisses a visible tooltip without moving focus (WCAG 1.4.13) and consumes the event so global back-navigation does not also fire.
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      hide();
    }
  };
  // Dismiss only when the scrolling target can actually move this anchor (ignore unrelated sibling scrollers like live logs).
  const onScroll = (e: Event) => {
    const t = e.target;
    if (t !== window && t !== e.currentTarget && t !== document && !(t instanceof Node && t.contains(node))) return;
    // Cancel any queued handoff timer too so scrolling mid-handoff cannot reopen the tooltip.
    clearTimeout(showTimer);
    pendingAnchor = null;
    hide();
  };

  const paint = (el: HTMLDivElement) => {
    if (!o) return;
    el.textContent = o.text;
    if (o.keys) {
      const k = document.createElement('span');
      k.className = 'folo-tooltip-keys';
      for (const part of formatKeys(o.keys)) {
        const kb = document.createElement('kbd');
        kb.textContent = part;
        k.appendChild(kb);
      }
      el.appendChild(k);
    }
    const r = node.getBoundingClientRect();
    const t = el.getBoundingClientRect();
    const p = placeTip(r, t, o.side, boundsFor(node, t.width));
    el.dataset.side = p.side;
    el.style.setProperty('--arrow', `${p.arrow}px`);
    el.style.transform = `translate(${p.x}px, ${p.y}px)`;
  };

  const show = (immediate: boolean, focused?: HTMLElement) => {
    if (!o) return;
    clearTimeout(showTimer);
    pendingAnchor = node;
    // Moving between neighbours while a tooltip is up shows the next one at once.
    const delay = immediate || activeAnchor !== null || Date.now() - lastHide < 300 ? 0 : 450;
    showTimer = setTimeout(() => {
      if (pendingAnchor === node) pendingAnchor = null;
      if (!o || !node.isConnected) return;
      clearTimeout(hideTimer);
      if (activeAnchor && activeAnchor !== node) hideCurrent?.();
      const el = mount(node.closest<HTMLElement>('dialog[open]') ?? document.body);
      el.dataset.show = '';
      activeAnchor = node;
      hideCurrent = hide;
      paint(el);
      // A new target (hover described one button, focus lands on another): the old one stops being described.
      const next = focused ?? target();
      if (next !== described) release(described);
      described = next;
      described.setAttribute('aria-describedby', [...own(described), el.id].join(' '));
      document.addEventListener('keydown', onKey);
      window.addEventListener('scroll', onScroll, { capture: true, passive: true });
    }, delay);
  };

  const hide = () => {
    if (pendingAnchor === node) {
      clearTimeout(showTimer);
      pendingAnchor = null;
    }
    if (activeAnchor === node) {
      clearTimeout(hideTimer);
      activeAnchor = null;
      if (hideCurrent === hide) hideCurrent = null;
      if (tip && 'show' in tip.dataset) {
        delete tip.dataset.show;
        lastHide = Date.now();
        if (popovers() && tip.matches(':popover-open')) tip.hidePopover();
        tip.hidden = true;
        tip.textContent = '';
      }
      document.removeEventListener('keydown', onKey);
      window.removeEventListener('scroll', onScroll, true);
      release(described);
    }
  };

  const onEnter = () => {
    if (activeAnchor === node) clearTimeout(hideTimer);
    show(false);
  };
  // Leaving before the delay is up cancels the pending show; a shown tooltip gets the grace period.
  const onLeave = () => {
    if (pendingAnchor === node) {
      clearTimeout(showTimer);
      pendingAnchor = null;
    }
    if (activeAnchor === node) leave();
  };
  // focusin / focusout bubble: focus on the node or anything inside it (a wrapped button) counts.
  const onFocus = (e: FocusEvent) => {
    if (e.target instanceof HTMLElement && e.target.matches(':focus-visible')) show(true, e.target);
  };
  node.addEventListener('pointerenter', onEnter);
  node.addEventListener('pointerleave', onLeave);
  node.addEventListener('focusin', onFocus);
  node.addEventListener('focusout', hide);
  node.addEventListener('pointerdown', hide);

  return {
    update(next: TooltipOptions) {
      o = norm(next);
      if (!o) {
        if (activeAnchor === node || pendingAnchor === node) hide();
      } else if (activeAnchor === node && tip && 'show' in tip.dataset) {
        paint(tip);
      }
    },
    destroy() {
      o = null;
      if (pendingAnchor === node) {
        clearTimeout(showTimer);
        pendingAnchor = null;
      }
      if (activeAnchor === node) hide();
      node.removeEventListener('pointerenter', onEnter);
      node.removeEventListener('pointerleave', onLeave);
      node.removeEventListener('focusin', onFocus);
      node.removeEventListener('focusout', hide);
      node.removeEventListener('pointerdown', hide);
    },
  };
}
