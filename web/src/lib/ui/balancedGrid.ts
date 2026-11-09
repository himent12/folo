// A grid of cards that never leaves one orphan (DESIGN_DIRECTION §2.2): as many columns of at least `min` px as fit,
// fewer when that would put a single card alone on the last row, so four datasets are 2 × 2, never 3 + 1. A plain
// `auto-fill` grid cannot know how many items it holds; this action measures the width and sets `--cols`, which the
// grid reads (`grid-template-columns: repeat(var(--cols, 1), minmax(0, 1fr))`).

/** Columns for `n` items when at most `max` fit: the widest count whose last row holds more than one item. */
export function balancedColumns(n: number, max: number): number {
  const top = Math.max(1, Math.min(n, Math.floor(max)));
  for (let c = top; c > 1; c--) if (n % c !== 1) return c;
  return 1;
}

/** How many columns of at least `min` px fit in `width` with `gap` between them. */
export function fitColumns(width: number, min: number, gap: number): number {
  return Math.max(1, Math.floor((width + gap) / (min + gap)));
}

export interface BalancedOptions {
  /** Items in the grid. */
  n: number;
  /** A column's minimum width, px. */
  min: number;
  /** The grid's column gap, px. */
  gap?: number;
  /** A cap on columns whatever the width. */
  max?: number;
}

export function balanced(node: HTMLElement, opts: BalancedOptions) {
  let o = opts;
  const apply = () => {
    const fit = fitColumns(node.clientWidth, o.min, o.gap ?? 16);
    node.style.setProperty('--cols', String(balancedColumns(o.n, Math.min(fit, o.max ?? Infinity))));
  };
  const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(apply);
  ro?.observe(node);
  apply();
  return {
    update(next: BalancedOptions) {
      o = next;
      apply();
    },
    destroy() {
      ro?.disconnect();
    },
  };
}
