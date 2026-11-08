// `use:tooltip={'Text'}` or `use:tooltip={{ text, keys: 'G R', side: 'right' }}`: a small tooltip on hover (after
// a short delay) and on keyboard focus. One shared element, positioned with fixed coordinates, so it never clips.
// The node may be a wrapper (a span around a component's button): focus inside it shows the tooltip, and the
// description goes on the focusable element, which is what assistive technology reads.

import { formatKeys } from '$lib/keys';

export type TooltipOptions = string | { text: string; keys?: string; side?: 'top' | 'bottom' | 'right' | 'left' } | null | undefined;

let tip: HTMLDivElement | null = null;
let showTimer: ReturnType<typeof setTimeout> | undefined;
let lastHide = 0;

function ensure(): HTMLDivElement {
  if (tip) return tip;
  tip = document.createElement('div');
  tip.className = 'folo-tooltip';
  tip.setAttribute('role', 'tooltip');
  tip.id = 'folo-tooltip';
  document.body.appendChild(tip);
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
  // Escape dismisses a visible tooltip without moving focus (WCAG 1.4.13).
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') hide();
  };

  const show = (immediate: boolean, focused?: HTMLElement) => {
    if (!o) return;
    clearTimeout(showTimer);
    // Moving between neighbours while a tooltip is up shows the next one at once.
    const delay = immediate || Date.now() - lastHide < 300 ? 0 : 450;
    showTimer = setTimeout(() => {
      if (!o || !node.isConnected) return;
      const el = ensure();
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
      el.dataset.show = '';
      const r = node.getBoundingClientRect();
      const t = el.getBoundingClientRect();
      const gap = 6;
      let x = r.left + r.width / 2 - t.width / 2;
      let y = r.top - t.height - gap;
      if (o.side === 'bottom' || (o.side === 'top' && y < 4)) y = r.bottom + gap;
      if (o.side === 'right') {
        x = r.right + gap;
        y = r.top + r.height / 2 - t.height / 2;
      }
      if (o.side === 'left') {
        x = r.left - t.width - gap;
        y = r.top + r.height / 2 - t.height / 2;
      }
      x = Math.max(4, Math.min(window.innerWidth - t.width - 4, x));
      el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      described = focused ?? target();
      described.setAttribute('aria-describedby', [...own(described), el.id].join(' '));
      document.addEventListener('keydown', onKey);
    }, delay);
  };

  const hide = () => {
    clearTimeout(showTimer);
    if (tip && 'show' in tip.dataset) {
      delete tip.dataset.show;
      lastHide = Date.now();
    }
    document.removeEventListener('keydown', onKey);
    const rest = own(described);
    if (rest.length) described.setAttribute('aria-describedby', rest.join(' '));
    else described.removeAttribute('aria-describedby');
  };

  const onEnter = () => show(false);
  // focusin / focusout bubble: focus on the node or anything inside it (a wrapped button) counts.
  const onFocus = (e: FocusEvent) => {
    if (e.target instanceof HTMLElement && e.target.matches(':focus-visible')) show(true, e.target);
  };
  node.addEventListener('pointerenter', onEnter);
  node.addEventListener('pointerleave', hide);
  node.addEventListener('focusin', onFocus);
  node.addEventListener('focusout', hide);
  node.addEventListener('pointerdown', hide);

  return {
    update(next: TooltipOptions) {
      o = norm(next);
      if (!o) hide();
    },
    destroy() {
      hide();
      node.removeEventListener('pointerenter', onEnter);
      node.removeEventListener('pointerleave', hide);
      node.removeEventListener('focusin', onFocus);
      node.removeEventListener('focusout', hide);
      node.removeEventListener('pointerdown', hide);
    },
  };
}
