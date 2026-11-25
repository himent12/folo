/**
 * A virtualised grid of image tiles over the window scroll (a dataset's images, a labeling job's): only the rows in
 * view (plus a little overscan) are in the DOM, and `onrange` asks for the items they show. Roving focus: arrows /
 * PageUp / PageDown / Home / End move, Enter or click opens. The tile's content is the caller's (`tile`).
 */
import { computed, css, effect, For, Index, signal, tick, untrack } from 'fw';

export type Density = 's' | 'm' | 'l';

/**
 * The cells' shape for a set of images: the commonest width / height among them, within 3:4 to 16:9 (4:3 for none),
 * so a set of square photos fills square cells instead of sitting in letterboxed 4:3 ones, whose bands read as uneven
 * gaps (wide beside each image, narrow between rows). An image of another shape is still shown whole on the cell's
 * bg.inset well.
 */
export function dominantAspect(images: Iterable<{ width: number; height: number } | undefined>): number {
  const seen = new Map<number, number>();
  for (const im of images) {
    if (!im?.width || !im.height) continue;
    const a = Math.round((im.width / im.height) * 100) / 100;
    seen.set(a, (seen.get(a) ?? 0) + 1);
  }
  let best = 4 / 3;
  let n = 0;
  for (const [a, c] of seen) if (c > n) [best, n] = [a, c];
  return Math.min(16 / 9, Math.max(3 / 4, best));
}

export interface TileGridProps {
  total: number;
  density: Density;
  /** The cells' width over height. */
  aspect: number;
  /** A tile's accessible name. */
  label: (i: number) => string;
  onopen: (i: number) => void;
  /** The items [lo, hi) of the rendered rows (overscan included) and [visLo, visHi) of the rows in the viewport. */
  onrange?: (lo: number, hi: number, visLo: number, visHi: number) => void;
  /** The sticky toolbar above the grid: keyboard moves keep the focused row clear of it. */
  toolbar: string;
  /** A tile's content; `i()` is its item's index, `inView()` is whether its row is in the viewport, and `scrollSeq()` bumps on scroll movement. */
  tile: (i: () => number, inView: () => boolean, scrollSeq: () => number) => JSX.Element;
  /** Bias overscan 3:1 toward scroll direction (same 4-row DOM budget, more preload ahead of scrolling). */
  directional?: boolean;
  /** An item to scroll into view and focus, each time a new object is given (where a viewer over the grid closed). */
  reveal?: { i: number } | null;
}

const MIN = { s: 112, m: 172, l: 248 } as const;
const GAP = 8;
/** Symmetric overscan at rest; biased 3:1 toward the scroll direction while scrolling (same 4-row DOM budget). */
const OVERSCAN = 2;
const OVERSCAN_LEAD = 3;
const OVERSCAN_TRAIL = 1;

export default function TileGrid(props: TileGridProps) {
  let el: HTMLDivElement | undefined;
  let toolbarEl: Element | null = null;
  let toolbarSel = '';
  const width = signal(0);
  /** Viewport-relative top of the grid, sticky toolbar bottom, and window height, updated on scroll / resize. */
  const top = signal(0);
  const head = signal(0);
  const vh = signal(800);
  const focusIdx = signal(0);
  /** Last scroll direction (-1 up, 0 rest, 1 down) and scroll movement counter for preload windowing. */
  const scrollDir = signal<-1 | 0 | 1>(0);
  const scrollSeq = signal(0);

  const safeAspect = computed(() => (Number.isFinite(props.aspect) && props.aspect > 0 ? Math.max(0.1, props.aspect) : 4 / 3));
  const cols = computed(() => Math.max(1, Math.floor((width.get() + GAP) / (MIN[props.density] + GAP))));
  const cellW = computed(() => (width.get() ? (width.get() - GAP * (cols.get() - 1)) / cols.get() : MIN[props.density]));
  const cellH = computed(() => Math.max(1, Math.round(cellW.get() / safeAspect.get())));
  const rowH = computed(() => cellH.get() + GAP);
  const rows = computed(() => Math.ceil(props.total / cols.get()));
  /** Rows in the unobscured viewport (without overscan): integer row indices, so sub-row scrolling never moves them. */
  const viewFirst = computed(() => Math.max(0, Math.floor((head.get() - top.get()) / rowH.get())));
  const viewLast = computed(() => Math.min(rows.get(), Math.ceil((vh.get() - top.get()) / rowH.get())));
  const overscanAbove = computed(() => (!props.directional ? OVERSCAN : scrollDir.get() < 0 ? OVERSCAN_LEAD : scrollDir.get() > 0 ? OVERSCAN_TRAIL : OVERSCAN));
  const overscanBelow = computed(() => (!props.directional ? OVERSCAN : scrollDir.get() > 0 ? OVERSCAN_LEAD : scrollDir.get() < 0 ? OVERSCAN_TRAIL : OVERSCAN));
  const first = computed(() => Math.max(0, viewFirst.get() - overscanAbove.get()));
  const last = computed(() => Math.min(rows.get(), viewLast.get() + overscanBelow.get()));
  const visible = computed(() => Array.from({ length: Math.max(0, last.get() - first.get()) }, (_, k) => first.get() + k));
  /** The cell that takes Tab focus: the focused one when rendered and within `total`, else the first in view. */
  const tabIdx = computed(() => {
    const f = focusIdx.get();
    const c = cols.get();
    const total = props.total;
    return f < total && f >= first.get() * c && f < last.get() * c
      ? f
      : Math.min(Math.max(0, total - 1), Math.min(rows.get(), viewFirst.get()) * c);
  });

  function resolveToolbar(): Element | null {
    const sel = props.toolbar;
    if (!toolbarEl || !toolbarEl.isConnected || toolbarSel !== sel) {
      toolbarSel = sel;
      toolbarEl = sel ? document.querySelector(sel) : null;
    }
    return toolbarEl;
  }

  function measure(fromScroll = false) {
    if (!el) return;
    const nextTop = el.getBoundingClientRect().top;
    if (fromScroll) {
      const prevTop = top.peek();
      if (nextTop < prevTop) {
        scrollDir.set(1);
        scrollSeq.set((n) => n + 1);
      } else if (nextTop > prevTop) {
        scrollDir.set(-1);
        scrollSeq.set((n) => n + 1);
      }
    }
    top.set(nextTop);
    const tb = resolveToolbar();
    head.set(tb ? Math.max(0, tb.getBoundingClientRect().bottom) : 0);
    vh.set(window.innerHeight);
  }

  effect(() => {
    let raf = 0;
    let sawScroll = false;
    const scheduleMeasure = (isScroll: boolean) => {
      if (isScroll) sawScroll = true;
      if (raf) return;
      raf = requestAnimationFrame(() => {
        const scrolled = sawScroll;
        raf = 0;
        sawScroll = false;
        measure(scrolled);
      });
    };
    const onScroll = () => scheduleMeasure(true);
    const onResize = () => scheduleMeasure(false);
    measure(false);
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onResize);
    return () => {
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onResize);
    };
  });

  // Re-measure when the layout above the grid may have moved (density, data arriving).
  effect(() => {
    void cols.get();
    void props.total;
    void props.density;
    untrack(() => {
      scrollDir.set(0);
      measure(false);
    });
  });

  let lastRange: [number, number, number, number, TileGridProps['onrange']] | null = null;
  effect(() => {
    const c = cols.get();
    const total = props.total;
    const lo = Math.min(total, first.get() * c);
    const hi = Math.min(total, Math.max(lo, last.get() * c));
    const visLo = Math.min(total, Math.max(lo, viewFirst.get() * c));
    const visHi = Math.min(hi, Math.max(visLo, Math.max(0, viewLast.get()) * c));
    const cb = props.onrange;
    if (lastRange && lastRange[0] === lo && lastRange[1] === hi && lastRange[2] === visLo && lastRange[3] === visHi && lastRange[4] === cb) return;
    lastRange = [lo, hi, visLo, visHi, cb];
    cb?.(lo, hi, visLo, visHi);
  });

  /** Height of sticky chrome over the top of the viewport (top bar + toolbar). */
  function stickyTop(): number {
    const tb = resolveToolbar();
    return tb ? tb.getBoundingClientRect().bottom : 48;
  }

  async function moveTo(i: number) {
    if (!el || props.total === 0) return;
    const j = Math.max(0, Math.min(props.total - 1, i));
    focusIdx.set(j);
    const r = Math.floor(j / cols.peek());
    const docTop = el.getBoundingClientRect().top + window.scrollY;
    const rowTop = docTop + r * rowH.peek();
    const head = stickyTop() + 8;
    const foot = window.innerHeight - (window.innerWidth < 768 ? 72 : 8);
    if (rowTop - window.scrollY < head) window.scrollTo({ top: rowTop - head });
    else if (rowTop + cellH.peek() - window.scrollY > foot) window.scrollTo({ top: rowTop + cellH.peek() - foot });
    measure();
    await tick();
    el.querySelector<HTMLElement>(`[data-i="${j}"]`)?.focus({ preventScroll: true });
  }

  effect(() => {
    const r = props.reveal;
    if (r) untrack(() => void moveTo(r.i));
  });

  function onkeydown(e: KeyboardEvent) {
    const c = cols.peek();
    const page = Math.max(1, Math.floor((window.innerHeight - 160) / rowH.peek())) * c;
    const map: Record<string, number> = {
      ArrowRight: 1,
      ArrowLeft: -1,
      ArrowDown: c,
      ArrowUp: -c,
      PageDown: page,
      PageUp: -page,
    };
    if (e.key in map) {
      e.preventDefault();
      void moveTo(focusIdx.peek() + (map[e.key] ?? 0));
    } else if (e.key === 'Home') {
      e.preventDefault();
      void moveTo(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      void moveTo(props.total - 1);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      props.onopen(focusIdx.peek());
    }
  }

  const cellsOf = (r: () => number) => Array.from({ length: Math.max(0, Math.min(cols.get(), props.total - r() * cols.get())) });
  const readScrollSeq = () => scrollSeq.get();

  return (
    <div
      class={`grid d-${props.density}`}
      ref={(e) => (el = e)}
      bind:clientWidth={width}
      style:height={`${rows.get() * rowH.get() - (rows.get() ? GAP : 0)}px`}
      role="grid"
      aria-label="Images"
      aria-rowcount={rows.get()}
      aria-colcount={cols.get()}
      tabindex={-1}
      onkeydown={onkeydown}
    >
      <For each={visible.get()}>
        {(r) => {
          const inView = () => r() >= viewFirst.get() && r() < viewLast.get();
          return (
            <div class="row" role="row" aria-rowindex={r() + 1} style:top={`${r() * rowH.get()}px`} style:height={`${cellH.get()}px`} style:gap={`${GAP}px`}>
              <Index each={cellsOf(r)}>
                {(_, c) => {
                  const i = () => r() * cols.get() + c;
                  // The grid's keys (onkeydown above) open the focused cell.
                  return (
                    <div
                      class="cell"
                      role="gridcell"
                      data-i={i()}
                      tabindex={i() === tabIdx.get() ? 0 : -1}
                      aria-colindex={c + 1}
                      aria-label={props.label(i())}
                      style:width={`${cellW.get()}px`}
                      onfocus={() => focusIdx.set(i())}
                      onclick={() => props.onopen(i())}
                    >
                      {props.tile(i, inView, readScrollSeq)}
                    </div>
                  );
                }}
              </Index>
            </div>
          );
        }}
      </For>
    </div>
  );
}

css`
  .grid {
    position: relative;
    width: 100%;
    contain: layout;
  }
  .grid:focus {
    outline: none;
  }
  .row {
    position: absolute;
    left: 0;
    right: 0;
    display: flex;
    contain: layout style;
  }
  .cell {
    position: relative;
    flex-shrink: 0;
    height: 100%;
    border-radius: var(--radius-md);
    overflow: hidden;
    contain: layout style;
    cursor: zoom-in;
    background: var(--bg-inset);
    box-shadow: inset 0 0 0 1px var(--line-subtle);
  }
  .d-s .cell {
    border-radius: var(--radius-sm);
  }
  .cell::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    box-shadow: inset 0 0 0 1px var(--line-subtle);
    pointer-events: none;
    transition: box-shadow var(--dur-fast) var(--ease-standard);
  }
  .cell:hover::after {
    box-shadow: inset 0 0 0 1px var(--line-strong);
  }
  .cell:focus-visible {
    outline-offset: 1px;
    border-radius: var(--radius-md);
  }
  @media (hover: none) {
    .cell {
      cursor: pointer;
    }
  }
`;
