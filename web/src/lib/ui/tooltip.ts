// `use:tooltip={'Text'}` or `use:tooltip={{ text, keys: 'G R', side: 'right' }}`: a small tooltip on hover (after
// a short delay) and on keyboard focus. One shared element, positioned with fixed coordinates, so it never clips.

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

export function tooltip(node: HTMLElement, opts: TooltipOptions) {
  let o = norm(opts);
  // The element's own aria-describedby (if any) is kept: the tooltip id is added to it, then removed again.
  const own = () => (node.getAttribute('aria-describedby') ?? '').split(/\s+/).filter((id) => id && id !== 'folo-tooltip');
  // Escape dismisses a visible tooltip without moving focus (WCAG 1.4.13).
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') hide();
  };

  const show = (immediate: boolean) => {
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
      node.setAttribute('aria-describedby', [...own(), el.id].join(' '));
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
    const rest = own();
    if (rest.length) node.setAttribute('aria-describedby', rest.join(' '));
    else node.removeAttribute('aria-describedby');
  };

  const onEnter = () => show(false);
  const onFocus = () => {
    if (node.matches(':focus-visible')) show(true);
  };
  node.addEventListener('pointerenter', onEnter);
  node.addEventListener('pointerleave', hide);
  node.addEventListener('focus', onFocus);
  node.addEventListener('blur', hide);
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
      node.removeEventListener('focus', onFocus);
      node.removeEventListener('blur', hide);
      node.removeEventListener('pointerdown', hide);
    },
  };
}
