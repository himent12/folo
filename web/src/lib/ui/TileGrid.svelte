<script lang="ts" module>
  export type Density = 's' | 'm' | 'l';
</script>

<script lang="ts">
  /**
   * A virtualised grid of image tiles over the window scroll (a dataset's images, a labeling job's): only the rows in
   * view (plus a little overscan) are in the DOM, and `onrange` asks for the items they show. Roving focus: arrows /
   * PageUp / PageDown / Home / End move, Enter or click opens. The tile's content is the caller's (`tile`).
   */
  import { tick, untrack, type Snippet } from 'svelte';

  let {
    total,
    density,
    aspect,
    label,
    onopen,
    onrange,
    toolbar,
    tile,
  }: {
    total: number;
    density: Density;
    /** The cells' width over height. */
    aspect: number;
    /** A tile's accessible name. */
    label: (i: number) => string;
    onopen: (i: number) => void;
    /** The items [lo, hi) of the rows in view, as they change. */
    onrange?: (lo: number, hi: number) => void;
    /** The sticky toolbar above the grid: keyboard moves keep the focused row clear of it. */
    toolbar: string;
    tile: Snippet<[number]>;
  } = $props();

  const MIN = { s: 112, m: 172, l: 248 } as const;
  const GAP = 8;
  const OVERSCAN = 2;

  let el = $state<HTMLDivElement>();
  let width = $state(0);
  /** Viewport-relative top of the grid and the window height, updated on scroll / resize. */
  let top = $state(0);
  let vh = $state(800);
  let focusIdx = $state(0);

  const cols = $derived(Math.max(1, Math.floor((width + GAP) / (MIN[density] + GAP))));
  const cellW = $derived(width ? (width - GAP * (cols - 1)) / cols : MIN[density]);
  const cellH = $derived(Math.round(cellW / aspect));
  const rowH = $derived(cellH + GAP);
  const rows = $derived(Math.ceil(total / cols));
  const first = $derived(Math.max(0, Math.floor(-top / rowH) - OVERSCAN));
  const last = $derived(Math.min(rows, Math.ceil((vh - top) / rowH) + OVERSCAN));
  const visible = $derived(Array.from({ length: Math.max(0, last - first) }, (_, k) => first + k));
  /** The cell that takes Tab focus: the focused one when rendered, else the first in view. */
  const tabIdx = $derived(
    focusIdx >= first * cols && focusIdx < last * cols ? focusIdx : Math.min(total - 1, Math.max(0, Math.floor(-top / rowH)) * cols),
  );

  function measure() {
    if (!el) return;
    top = el.getBoundingClientRect().top;
    vh = window.innerHeight;
  }

  $effect(() => {
    let raf = 0;
    const on = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(measure);
    };
    measure();
    window.addEventListener('scroll', on, { passive: true });
    window.addEventListener('resize', on);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('scroll', on);
      window.removeEventListener('resize', on);
    };
  });

  // Re-measure when the layout above the grid may have moved (density, data arriving).
  $effect(() => {
    void cols;
    void total;
    void density;
    untrack(measure);
  });

  $effect(() => {
    onrange?.(first * cols, last * cols);
  });

  /** Height of sticky chrome over the top of the viewport (top bar + toolbar). */
  function stickyTop(): number {
    const tb = document.querySelector(toolbar);
    return tb ? tb.getBoundingClientRect().bottom : 48;
  }

  async function moveTo(i: number) {
    if (!el || total === 0) return;
    const j = Math.max(0, Math.min(total - 1, i));
    focusIdx = j;
    const r = Math.floor(j / cols);
    const docTop = el.getBoundingClientRect().top + window.scrollY;
    const rowTop = docTop + r * rowH;
    const head = stickyTop() + 8;
    const foot = window.innerHeight - (window.innerWidth < 768 ? 72 : 8);
    if (rowTop - window.scrollY < head) window.scrollTo({ top: rowTop - head });
    else if (rowTop + cellH - window.scrollY > foot) window.scrollTo({ top: rowTop + cellH - foot });
    measure();
    await tick();
    el.querySelector<HTMLElement>(`[data-i="${j}"]`)?.focus({ preventScroll: true });
  }

  function onkeydown(e: KeyboardEvent) {
    const page = Math.max(1, Math.floor((window.innerHeight - 160) / rowH)) * cols;
    const map: Record<string, number> = {
      ArrowRight: 1,
      ArrowLeft: -1,
      ArrowDown: cols,
      ArrowUp: -cols,
      PageDown: page,
      PageUp: -page,
    };
    if (e.key in map) {
      e.preventDefault();
      void moveTo(focusIdx + (map[e.key] ?? 0));
    } else if (e.key === 'Home') {
      e.preventDefault();
      void moveTo(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      void moveTo(total - 1);
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onopen(focusIdx);
    }
  }
</script>

<div
  class="grid d-{density}"
  bind:this={el}
  bind:clientWidth={width}
  style:height="{rows * rowH - (rows ? GAP : 0)}px"
  role="grid"
  aria-label="Images"
  aria-rowcount={rows}
  aria-colcount={cols}
  tabindex="-1"
  {onkeydown}
>
  {#each visible as r (r)}
    <div class="row" role="row" aria-rowindex={r + 1} style:top="{r * rowH}px" style:height="{cellH}px" style:gap="{GAP}px">
      {#each { length: Math.min(cols, total - r * cols) }, c (c)}
        {@const i = r * cols + c}
        <!-- svelte-ignore a11y_click_events_have_key_events: the grid's keys (onkeydown above) open the focused cell. -->
        <div
          class="cell"
          role="gridcell"
          data-i={i}
          tabindex={i === tabIdx ? 0 : -1}
          aria-colindex={c + 1}
          aria-label={label(i)}
          style:width="{cellW}px"
          onfocus={() => (focusIdx = i)}
          onclick={() => onopen(i)}
        >
          {@render tile(i)}
        </div>
      {/each}
    </div>
  {/each}
</div>

<style>
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
  }
  .cell {
    position: relative;
    flex-shrink: 0;
    height: 100%;
    border-radius: var(--radius-md);
    overflow: hidden;
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
</style>
