/**
 * Caches an element's `DOMRect` across high-frequency pointer/wheel events (charts, image inspector) so `pointermove`
 * and `wheel` handlers do not force a layout read on every frame.
 *
 * A single shared `window` capture `scroll` + `resize` listener invalidates all active caches while any component using
 * one is mounted, and detaches when the last component unmounts. Callers also call `clear()` on `pointerenter`,
 * `pointerleave`, fresh `pointerdown`, and element resize (`width`/`height` signal changes).
 */
import { onCleanup } from 'fw';

export interface RectCache {
  /** Returns the cached `DOMRect` for `el`, measuring lazily if cleared or for a different element. */
  get(el: Element): DOMRect;
  /** Invalidates the cached rect so the next `get()` re-measures. */
  clear(): void;
}

const active = new Set<() => void>();

function invalidateAll() {
  for (const clear of active) clear();
}

function attachGlobal() {
  if (typeof window === 'undefined') return;
  // Capture catches non-bubbling scrolls on inner containers; bubble catches synthetic window scroll events.
  window.addEventListener('scroll', invalidateAll, { capture: true, passive: true });
  window.addEventListener('scroll', invalidateAll, { passive: true });
  window.addEventListener('resize', invalidateAll, { passive: true });
}

function detachGlobal() {
  if (typeof window === 'undefined') return;
  window.removeEventListener('scroll', invalidateAll, { capture: true });
  window.removeEventListener('scroll', invalidateAll);
  window.removeEventListener('resize', invalidateAll);
}

/** Creates a component-scoped `DOMRect` cache, automatically unregistered when its owner cleans up. */
export function createRectCache(): RectCache {
  let cachedEl: Element | null = null;
  let cachedRect: DOMRect | null = null;

  const clear = () => {
    cachedEl = null;
    cachedRect = null;
  };

  if (active.size === 0) attachGlobal();
  active.add(clear);
  onCleanup(() => {
    active.delete(clear);
    if (active.size === 0) detachGlobal();
    clear();
  });

  return {
    get(el: Element): DOMRect {
      if (cachedRect && cachedEl === el) return cachedRect;
      cachedEl = el;
      cachedRect = el.getBoundingClientRect();
      return cachedRect;
    },
    clear,
  };
}
