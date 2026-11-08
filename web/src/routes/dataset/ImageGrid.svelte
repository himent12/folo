<script lang="ts">
  /**
   * A virtualised thumbnail grid over the window scroll: only the rows in view (plus a little overscan) are in the
   * DOM, and image pages load as they come into view. Roving focus: arrows / PageUp / PageDown / Home / End move,
   * Enter or click opens.
   */
  import { tick, untrack } from 'svelte';
  import DatasetImage from '$lib/detect/DatasetImage.svelte';
  import type { ImagePager } from './pager.svelte';

  export type Density = 's' | 'm' | 'l';

  let {
    pager,
    classes,
    density,
    boxes,
    only,
    onopen,
  }: {
    pager: ImagePager;
    classes: readonly string[];
    density: Density;
    boxes: boolean;
    only: ReadonlySet<number> | null;
    onopen: (index: number, id: string) => void;
  } = $props();

  const MIN = { s: 112, m: 172, l: 248 } as const;
  const GAP = { s: 4, m: 8, l: 8 } as const;
  const OVERSCAN = 2;

  let el = $state<HTMLDivElement>();
  let width = $state(0);
  /** Viewport-relative top of the grid and the window height, updated on scroll / resize. */
  let top = $state(0);
  let vh = $state(800);
  let focusIdx = $state(0);

  const gap = $derived(GAP[density]);
  const cols = $derived(Math.max(1, Math.floor((width + gap) / (MIN[density] + gap))));
  const cellW = $derived(width ? (width - gap * (cols - 1)) / cols : MIN[density]);
  const cellH = $derived(Math.round(cellW * 0.75));
  const rowH = $derived(cellH + gap);
  const total = $derived(pager.total ?? 0);
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

  // Load the pages for the rows in view.
  $effect(() => {
    pager.ensure(first * cols, last * cols);
  });

  // A new filter starts at the first image.
  $effect(() => {
    void pager;
    untrack(() => (focusIdx = 0));
  });

  /** Height of sticky chrome over the top of the viewport (top bar + toolbar). */
  function stickyTop(): number {
    const tb = document.querySelector('.ds-toolbar');
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
      const img = pager.get(focusIdx);
      if (img) {
        e.preventDefault();
        onopen(focusIdx, img.id);
      }
    }
  }

  const stem = (file: string) => file.split('/').pop() ?? file;
</script>

<div
  class="grid d-{density}"
  bind:this={el}
  bind:clientWidth={width}
  style:height="{rows * rowH - (rows ? gap : 0)}px"
  role="grid"
  aria-label="Images"
  aria-rowcount={rows}
  aria-colcount={cols}
  tabindex="-1"
  {onkeydown}
>
  {#each visible as r (r)}
    <div class="row" role="row" aria-rowindex={r + 1} style:top="{r * rowH}px" style:height="{cellH}px" style:gap="{gap}px">
      {#each { length: Math.min(cols, total - r * cols) }, c (c)}
        {@const i = r * cols + c}
        {@const img = pager.get(i)}
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <div
          class="cell"
          role="gridcell"
          data-i={i}
          tabindex={i === tabIdx ? 0 : -1}
          aria-colindex={c + 1}
          aria-label={img ? `${stem(img.file)}, ${img.split}, ${img.boxes.length} ${img.boxes.length === 1 ? 'box' : 'boxes'}` : 'Loading image'}
          style:width="{cellW}px"
          onfocus={() => (focusIdx = i)}
          onclick={() => img && onopen(i, img.id)}
        >
          {#if img}
            <DatasetImage
              datasetId={pager.datasetId}
              imageId={img.id}
              width={img.width}
              height={img.height}
              thumb={256}
              boxes={boxes ? img.boxes : []}
              {classes}
              {only}
              compact
              fit="cover"
              alt=""
            />
            <span class="meta">
              <span class="file mono truncate">{stem(img.file)}</span>
              <span class="n num">{img.boxes.length}</span>
            </span>
          {:else}
            <span class="ph"></span>
          {/if}
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
  .meta {
    position: absolute;
    left: 4px;
    right: 4px;
    bottom: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-1);
    pointer-events: none;
    opacity: 0;
    transform: translateY(2px);
    transition:
      opacity var(--dur-fast) var(--ease-standard),
      transform var(--dur-fast) var(--ease-standard);
  }
  .meta > span {
    height: 18px;
    display: inline-flex;
    align-items: center;
    padding: 0 5px;
    border-radius: var(--radius-xs);
    background: var(--bg-scrim);
    color: var(--text-primary);
    font-size: var(--type-micro-size);
    backdrop-filter: blur(4px);
  }
  :global([data-theme='light']) .meta > span {
    background: color-mix(in srgb, var(--bg-overlay) 88%, transparent);
    color: var(--text-primary);
  }
  .cell:hover .meta,
  .cell:focus-visible .meta {
    opacity: 1;
    transform: none;
  }
  .d-s .meta {
    display: none;
  }
  .ph {
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, var(--bg-hover), transparent);
    background-size: 200% 100%;
    animation: shimmer 1.4s linear infinite;
  }
  @keyframes shimmer {
    from {
      background-position: 100% 0;
    }
    to {
      background-position: -100% 0;
    }
  }
  @media (hover: none) {
    .cell {
      cursor: pointer;
    }
  }
</style>
