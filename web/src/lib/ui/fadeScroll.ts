// `use:fadeScroll` on a box that scrolls sideways (a command line): it fades the edge that has more behind it
// (`data-fade="start" | "end" | "both"`, styled by the box) and takes a tab stop only while it overflows, so a keyboard
// can scroll it then and skips it otherwise. Pair it with `data-scrolls` (a deliberate scroller, which the shoot's
// overflow check skips).

export function fadeScroll(node: HTMLElement) {
  let raf = 0;
  const update = () => {
    const max = node.scrollWidth - node.clientWidth;
    const start = node.scrollLeft > 1;
    const end = max > 1 && node.scrollLeft < max - 1;
    const next = start || end ? (start && end ? 'both' : start ? 'start' : 'end') : undefined;
    if (next !== node.dataset.fade) {
      if (next) node.dataset.fade = next;
      else delete node.dataset.fade;
    }
    if (max > 1) {
      if (node.getAttribute('tabindex') !== '0') node.tabIndex = 0;
    } else if (node.hasAttribute('tabindex')) {
      node.removeAttribute('tabindex');
    }
  };
  // Defer ResizeObserver DOM writes to the next animation frame so layout-driven attribute/mask changes
  // never trigger Chromium's "ResizeObserver loop completed with undelivered notifications." error.
  const schedule = () => {
    if (raf) return;
    if (typeof requestAnimationFrame === 'undefined') {
      update();
      return;
    }
    raf = requestAnimationFrame(() => {
      raf = 0;
      if (node.isConnected) update();
    });
  };
  update();
  node.addEventListener('scroll', update, { passive: true });
  const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(schedule);
  ro?.observe(node);
  return {
    update,
    destroy() {
      if (raf && typeof cancelAnimationFrame !== 'undefined') cancelAnimationFrame(raf);
      raf = 0;
      node.removeEventListener('scroll', update);
      ro?.disconnect();
    },
  };
}
