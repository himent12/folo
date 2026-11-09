// `use:fadeScroll` on a box that scrolls sideways (a command line): it fades the edge that has more behind it
// (`data-fade="start" | "end" | "both"`, styled by the box) and takes a tab stop only while it overflows, so a keyboard
// can scroll it then and skips it otherwise. Pair it with `data-scrolls` (a deliberate scroller, which the shoot's
// overflow check skips).

export function fadeScroll(node: HTMLElement) {
  const update = () => {
    const max = node.scrollWidth - node.clientWidth;
    const start = node.scrollLeft > 1;
    const end = max > 1 && node.scrollLeft < max - 1;
    if (start || end) node.dataset.fade = start && end ? 'both' : start ? 'start' : 'end';
    else delete node.dataset.fade;
    if (max > 1) node.tabIndex = 0;
    else node.removeAttribute('tabindex');
  };
  update();
  node.addEventListener('scroll', update, { passive: true });
  const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(update);
  ro?.observe(node);
  return {
    update,
    destroy() {
      node.removeEventListener('scroll', update);
      ro?.disconnect();
    },
  };
}
