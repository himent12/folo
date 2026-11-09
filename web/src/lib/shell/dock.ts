// A page with a bottom dock (New run's Start bar) hides the phone tab bar while the dock is mounted, so the fixed
// chrome at the bottom is one bar, not two. Put `use:dock` on the dock element (or call holdDock() and release it).
// While any dock is held, <html data-dock> is set: base.css drops --tabbar-h to 0 (so a dock positioned at
// `bottom: calc(var(--tabbar-h) + env(safe-area-inset-bottom))` moves to the screen's edge) and TabBar hides. The dock's
// height is published as --dock-h, so toasts sit above it. The tab bar only exists below 768 px; wider, this is a no-op.

let holds = 0;

function sync() {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  root.toggleAttribute('data-dock', holds > 0);
  if (holds === 0) root.style.removeProperty('--dock-h');
}

/** Hides the tab bar until the returned function is called (once; further calls do nothing). Holds nest. */
export function holdDock(): () => void {
  holds++;
  sync();
  let released = false;
  return () => {
    if (released) return;
    released = true;
    holds--;
    sync();
  };
}

/** `use:dock` on a page's bottom dock: hides the tab bar while the element is mounted and publishes its height. */
export function dock(node: HTMLElement) {
  const release = holdDock();
  const publish = () => document.documentElement.style.setProperty('--dock-h', `${node.offsetHeight}px`);
  publish();
  const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(publish);
  ro?.observe(node);
  return {
    destroy() {
      ro?.disconnect();
      release();
    },
  };
}
