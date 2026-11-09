<script lang="ts">
  /**
   * Confusion matrix: rows are true classes, columns predicted, plus a "background" row/column for misses and false
   * positives. Cells are row-normalised and shaded on one sequential luminance ramp in ink (the diagonal is told apart
   * by position, not hue), counts on hover. In light, the ramp starts at the first non-zero share: an empty cell is the
   * panel with a hairline grid, so most of a large matrix is not a grey chequerboard. `reveal(r, c)` lets a summary elsewhere point at one cell: it scrolls the
   * cell into view and makes it the active cell, whose counts the readout states. A matrix that scrolls sideways
   * (phones) also gets a whole-matrix overview: every cell as a small square and a band over the columns in view. A tap
   * on it brings those columns into view; cells are picked in the full matrix, whose cells are large enough to hit.
   */
  import { classColor } from '$lib/design/tokens';
  import { prefs } from '$lib/prefs.svelte';
  import { overviewGeometry, overviewHit, overviewWindow } from './cmOverview';
  import { CONFUSION_FLIP } from './confusionInk';
  // Cell labels are pure white or black, switched near the ramp's luminance crossover (see ./confusionInk).
  const flip = $derived(CONFUSION_FLIP[prefs.resolvedTheme]);

  let {
    classes,
    matrix,
    label = 'Confusion matrix',
    sparseLabels = false,
    minCell,
    scale = false,
    classIds,
  }: {
    classes: string[];
    /** (n+1)×(n+1) counts; the last row/column is background. */
    matrix: number[][];
    label?: string;
    /**
     * Label only cells of at least 10% that are at least 32 px wide, so dense matrices don't run numerals together;
     * the readout still gives every value on hover or focus.
     */
    sparseLabels?: boolean;
    /**
     * Minimum cell size in px. Wide matrices then scroll sideways with the row labels pinned, instead of squeezing
     * cells; on touch screens cells are at least 44 px.
     */
    minCell?: number;
    /** Show a labelled 0–100% colour scale under the matrix. */
    scale?: boolean;
    /**
     * The palette index of each entry of `classes` (defaults to its position), for reduced matrices whose rows are a
     * subset of the model's classes; a negative index draws a neutral dot (e.g. "other classes").
     */
    classIds?: number[];
  } = $props();

  const uid = $props.id();
  let hovered = $state<{ r: number; c: number } | null>(null);
  /** A cell chosen by keyboard or tap keeps the readout when the pointer leaves. */
  let pinned = $state<{ r: number; c: number } | null>(null);
  /** Roving tabindex: one cell is in the tab order; arrow keys move it. */
  let focus = $state({ r: 0, c: 0 });
  let root: HTMLDivElement;
  /** A scrolling matrix fades its right edge while there are more columns to scroll to. */
  let moreRight = $state(false);
  /** Fully visible prediction columns (1-based) while the matrix scrolls sideways; null when every column fits. */
  let range = $state<{ first: number; last: number } | null>(null);
  const fixed = $derived(minCell !== undefined);
  function onscroll() {
    moreRight = root.scrollLeft + root.clientWidth < root.scrollWidth - 1;
    const head = root.querySelector<HTMLElement>('.rowhead');
    const cell = root.querySelector<HTMLElement>('.cell');
    if (!head || !cell || root.scrollWidth <= root.clientWidth + 1) {
      range = null;
      return;
    }
    // Columns start after the pinned row labels; each is one cell plus the 2 px grid gap. The faded right edge
    // (24 px) does not count as visible.
    const step = cell.offsetWidth + 2;
    const left = root.scrollLeft;
    const end = left + root.clientWidth - head.offsetWidth - (moreRight ? 12 : 0);
    const first = Math.min(n, Math.ceil(left / step - 0.05) + 1);
    const last = Math.max(first, Math.min(n, Math.floor(end / step + 0.05)));
    range = { first, last };
  }
  const behavior = (): ScrollBehavior => (matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth');
  const scrollEnd = () => root.scrollTo({ left: root.scrollWidth, behavior: behavior() });

  /**
   * Shows the cell at row `r`, column `c` (indices into `classes`; `classes.length` is background) and makes it the
   * active cell: scrolled into view past the pinned row labels, focused (so the keyboard continues from it and a
   * screen reader reads it), marked and stated by the readout.
   */
  export function reveal(r: number, c: number, block: ScrollLogicalPosition = 'center') {
    const cell = root?.querySelector<HTMLElement>(`[data-r="${r}"][data-c="${c}"]`);
    if (!cell) return;
    focus = { r, c };
    cell.focus({ preventScroll: true });
    hovered = pinned = { r, c };
    // scroll-padding keeps the cell clear of the sticky row labels and the faded right edge.
    cell.scrollIntoView({ block, inline: 'nearest', behavior: behavior() });
  }
  $effect(() => {
    if (!fixed) return;
    onscroll();
    const ro = new ResizeObserver(onscroll);
    ro.observe(root);
    return () => ro.disconnect();
  });

  function onkeydown(e: KeyboardEvent) {
    const max = n - 1;
    let { r, c } = focus;
    if (e.key === 'ArrowRight') c = Math.min(max, c + 1);
    else if (e.key === 'ArrowLeft') c = Math.max(0, c - 1);
    else if (e.key === 'ArrowDown') r = Math.min(max, r + 1);
    else if (e.key === 'ArrowUp') r = Math.max(0, r - 1);
    else if (e.key === 'Home') c = 0;
    else if (e.key === 'End') c = max;
    else return;
    e.preventDefault();
    focus = { r, c };
    root.querySelector<HTMLElement>(`[data-r="${r}"][data-c="${c}"]`)?.focus();
  }
  const names = $derived([...classes, 'background']);
  const rowSums = $derived(matrix.map((row) => row.reduce((a, b) => a + b, 0)));
  const norm = (r: number, c: number) => {
    const s = rowSums[r] ?? 0;
    return s ? (matrix[r]?.[c] ?? 0) / s : 0;
  };
  const n = $derived(names.length);
  /** Swatch colour of row/column `i`: the class palette hue, neutral for background and folded rows. */
  const dotColor = (i: number) => {
    const id = i < classes.length ? (classIds?.[i] ?? i) : -1;
    return id >= 0 ? classColor(id) : 'var(--text-tertiary)';
  };

  const geo = $derived(overviewGeometry(n));
  const band = $derived(range ? overviewWindow(range.first, range.last, geo) : null);
  /**
   * A tap on the overview centres the tapped column in the full matrix (sideways only, so the page does not jump).
   * Coarse on purpose: the map's squares are far below a touch target, so a near miss only moves the view a column.
   */
  function onmap(e: MouseEvent) {
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
    const hit = overviewHit(e.clientX - rect.left, e.clientY - rect.top, geo, n);
    const cell = hit && root?.querySelector<HTMLElement>(`[data-r="0"][data-c="${hit.c}"]`);
    if (!cell) return;
    const box = root.getBoundingClientRect();
    const at = cell.getBoundingClientRect();
    root.scrollTo({ left: root.scrollLeft + at.left - box.left - (box.width - at.width) / 2, behavior: behavior() });
  }
</script>

{#snippet readout()}
  <p class="readout" class:faint={!hovered} id="{uid}-readout">
    {#if hovered}
      {@const h = hovered}
      <b>{names[h.r]}</b> predicted as <b>{names[h.c]}</b>:
      <span class="num">{(matrix[h.r]?.[h.c] ?? 0).toLocaleString('en-US')}</span> of
      <span class="num">{(rowSums[h.r] ?? 0).toLocaleString('en-US')}</span>
      (<span class="num">{(norm(h.r, h.c) * 100).toFixed(1)}%</span>)
    {:else if range}
      Tap the map to move to those columns, or a cell for its counts.
    {:else}
      Hover, tap or arrow through the cells for counts.
    {/if}
  </p>
{/snippet}

{#if range && band}
  <!-- The whole matrix at a glance, with the selected cell's figures beside it: the full matrix below shows only a
       few columns at a time. A pointer shortcut; the grid itself stays the keyboard and screen-reader path. -->
  <div class="ov">
    <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
    <svg class="ov-map" width={geo.size} height={geo.size} viewBox="0 0 {geo.size} {geo.size}" aria-hidden="true" onclick={onmap}>
      {#each names as _, r (r)}
        {#each names as _, c (c)}
          <rect class="ov-cell" class:zero={!matrix[r]?.[c]} x={c * (geo.cell + geo.gap)} y={r * (geo.cell + geo.gap)} width={geo.cell} height={geo.cell} rx="1" style:--v={norm(r, c)} />
        {/each}
      {/each}
      <rect class="ov-band" x={band.x - 1.5} y="-1.5" width={band.w + 3} height={geo.size + 3} rx="2" />
      {#if pinned}
        <rect class="ov-pin" x={pinned.c * (geo.cell + geo.gap) - 1} y={pinned.r * (geo.cell + geo.gap) - 1} width={geo.cell + 2} height={geo.cell + 2} rx="2" />
      {/if}
    </svg>
    {@render readout()}
  </div>
{/if}
{#if range}
  <!-- What the sideways scroll hides: the columns in view, and a way to the last ones (background = missed objects). -->
  <p class="range num">
    Columns {range.first}–{range.last} of {n}{#if range.last < n}<span class="sep" aria-hidden="true">·</span><button type="button" class="more" onclick={scrollEnd}
        >Scroll for {n - range.last} more, incl. background →</button
      >{/if}
  </p>
{/if}
<!-- svelte-ignore a11y_interactive_supports_focus -->
<div class="cm" class:sparse={sparseLabels} class:fixed class:more-r={moreRight} style:--n={n} style:--min-cell={minCell !== undefined ? `${minCell}px` : undefined} role="grid" aria-label={label} aria-describedby="{uid}-readout" bind:this={root} {onkeydown} {onscroll}>
  <div class="row" role="row">
    <div class="corner" role="columnheader"><span>Predicted →</span><span>True ↓</span></div>
    {#each names as name, c (c)}
      <div class="colhead" role="columnheader" title={name}>
        <span class="dot" style:background={dotColor(c)}></span>
        <span class="lbl">{name}</span>
      </div>
    {/each}
  </div>
  {#each names as rname, r (r)}
    <div class="row" role="row">
      <div class="rowhead" role="rowheader" title={rname}>
        <span class="dot" style:background={dotColor(r)}></span>
        <span class="lbl">{rname}</span>
      </div>
      {#each names as cname, c (c)}
        {@const v = norm(r, c)}
        <div
          class="cell"
          class:zero={!matrix[r]?.[c]}
          class:hl={hovered && (hovered.r === r || hovered.c === c)}
          class:pin={pinned !== null && pinned.r === r && pinned.c === c}
          role="gridcell"
          tabindex={focus.r === r && focus.c === c ? 0 : -1}
          data-r={r}
          data-c={c}
          aria-label="{rname} predicted as {cname}: {(matrix[r]?.[c] ?? 0).toLocaleString('en-US')} of {(rowSums[r] ?? 0).toLocaleString('en-US')}, {(v * 100).toFixed(1)}%"
          style:--v={v}
          onpointerenter={() => (hovered = { r, c })}
          onpointerleave={() => (hovered = pinned)}
          onfocus={() => ((focus = { r, c }), (hovered = pinned = { r, c }))}
          onblur={() => (hovered = pinned = null)}
          onclick={() => (hovered = pinned = { r, c })}
          onkeydown={() => {}}
        >
          {#if v >= (sparseLabels ? 0.1 : 0.005)}<span class:hi={v > flip}>{(v * 100).toFixed(v >= 0.1 ? 0 : 1)}</span>{/if}
        </div>
      {/each}
    </div>
  {/each}
</div>
{#if scale}
  <div class="scale" aria-hidden="true">
    <span class="scale-name">Share of the true class</span>
    <span class="keys">
      <span class="none-key"><i></i>None</span>
      <span class="ramp-key">
        <span class="ramp"></span>
        <span class="ticks"><span>&gt;0%</span><span>50%</span><span>100%</span></span>
      </span>
    </span>
  </div>
{/if}
{#if !range}{@render readout()}{/if}

<style>
  .row {
    display: contents;
  }
  .cm {
    display: grid;
    grid-template-columns: minmax(64px, 96px) repeat(var(--n), minmax(22px, 1fr));
    gap: 2px;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    overflow-x: auto;
  }
  /* Fixed minimum cells: the matrix scrolls sideways under pinned row labels. */
  .cm.fixed {
    /* Cells grow with their column up to 72 px, so a wide panel does not turn the matrix into a wall. */
    grid-template-columns: 96px repeat(var(--n), minmax(var(--min-cell), 72px));
    overscroll-behavior-x: contain;
    /* Scrolling a cell into view stops short of the pinned labels (96 + 2 px gap) and the 24 px fade. */
    scroll-padding-inline: 98px 24px;
  }
  .cm.more-r {
    mask-image: linear-gradient(to left, transparent, #000 24px);
  }
  .fixed .corner,
  .fixed .rowhead {
    position: sticky;
    left: 0;
    z-index: 1;
    background: var(--bg-panel);
    /* Covers the 2 px gap so scrolled cells don't show between the label and the first column. */
    box-shadow: 2px 0 0 var(--bg-panel);
  }
  .fixed .cell {
    min-width: var(--min-cell);
  }
  @media (pointer: coarse) {
    .cm.fixed {
      --min-cell: 44px !important;
    }
  }
  /* Orientation, at the helper size: what the columns and the rows are. */
  .corner {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    gap: 2px;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    padding: 0 0 4px;
    white-space: nowrap;
  }
  .colhead {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    gap: 4px;
    min-height: 64px;
    padding-bottom: 4px;
    color: var(--text-secondary);
  }
  .colhead .lbl {
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    max-height: 72px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .rowhead {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    color: var(--text-secondary);
    padding-right: 4px;
  }
  .rowhead .lbl {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  /* The raw palette hue, edged so pale classes hold up on white. */
  :global([data-theme='light']) .dot {
    box-shadow: inset 0 0 0 1px #0a0e1433;
  }
  .cell {
    aspect-ratio: 1;
    display: grid;
    place-items: center;
    border-radius: 3px;
    /* One ramp from the plot well to full ink, perceptually even (oklab). */
    background: color-mix(in oklab, var(--text-primary) calc(var(--v) * 90% + 4%), var(--bg-inset));
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
    color: var(--text-secondary);
    transition: box-shadow var(--dur-fast) var(--ease-standard);
    min-width: 22px;
  }
  /* Each cell is a size container so a label can hide itself when its cell is too narrow to hold it. */
  .sparse .cell {
    container-type: inline-size;
  }
  @container (max-width: 31.98px) {
    .sparse .cell span {
      display: none;
    }
  }
  /* Light: no objects at all is the panel itself, edged by the hairline grid; the ramp begins at the first object. */
  :global([data-theme='light']) .cell.zero {
    background: var(--bg-panel);
    box-shadow: inset 0 0 0 1px var(--line-subtle);
  }
  .cell.hl,
  :global([data-theme='light']) .cell.zero.hl {
    box-shadow: inset 0 0 0 1px var(--line-strong);
  }
  /* The active cell (tapped, or pointed at from a summary): a selection stroke; keyboard focus has its own ring. */
  .cell.pin:not(:focus-visible) {
    box-shadow:
      inset 0 0 0 2px var(--accent-line),
      inset 0 0 0 3px var(--bg-panel);
  }
  /* Low cells sit near the well, high cells near the ink; the label takes the far extreme (see `flip`). */
  :global([data-theme='dark']) .cell span {
    color: #ffffff;
  }
  :global([data-theme='dark']) .cell span.hi,
  :global([data-theme='light']) .cell span {
    color: #000000;
  }
  :global([data-theme='light']) .cell span.hi {
    color: #ffffff;
  }
  .cell span.hi {
    font-weight: var(--weight-medium);
  }
  .range {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    min-height: 24px;
    margin-bottom: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .range .sep {
    margin-inline: var(--space-2);
  }
  .more {
    color: var(--text-secondary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .more:hover {
    color: var(--text-primary);
  }
  .readout {
    margin-top: var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .ov {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: var(--space-3);
    align-items: start;
    margin-bottom: var(--space-3);
  }
  .ov .readout {
    margin-top: 0;
  }
  .ov-map {
    display: block;
    overflow: visible;
    cursor: pointer;
    touch-action: manipulation;
  }
  .ov-cell {
    fill: color-mix(in oklab, var(--text-primary) calc(var(--v) * 90% + 4%), var(--bg-inset));
  }
  :global([data-theme='light']) .ov-cell.zero {
    fill: var(--bg-panel);
    stroke: var(--line-subtle);
    stroke-width: 1;
  }
  .ov-band {
    fill: none;
    stroke: var(--text-secondary);
    stroke-width: 1;
  }
  .ov-pin {
    fill: none;
    stroke: var(--accent-line);
    stroke-width: 2;
  }
  .readout b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .num {
    font-variant-numeric: tabular-nums;
  }
  .scale {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: var(--space-1) var(--space-3);
    max-width: 400px;
    margin-top: var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .keys {
    flex: 1 1 200px;
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
  }
  /* The empty-cell key exists only where empty cells differ from the ramp's start (light). */
  .none-key {
    display: none;
    align-items: center;
    gap: var(--space-1);
    white-space: nowrap;
  }
  :global([data-theme='light']) .none-key {
    display: inline-flex;
  }
  .none-key i {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    background: var(--bg-panel);
    box-shadow: inset 0 0 0 1px var(--line-default);
  }
  .ramp-key {
    flex: 1;
    display: grid;
    gap: var(--space-1);
    padding-top: 4px;
  }
  .ramp {
    height: 8px;
    border-radius: 2px;
    background: linear-gradient(
      to right in oklab,
      color-mix(in oklab, var(--text-primary) 4%, var(--bg-inset)),
      color-mix(in oklab, var(--text-primary) 94%, var(--bg-inset))
    );
  }
  .ticks {
    display: flex;
    justify-content: space-between;
    font-variant-numeric: tabular-nums;
  }
</style>
