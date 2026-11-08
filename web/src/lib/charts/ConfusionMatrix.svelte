<script lang="ts">
  /**
   * Confusion matrix: rows are true classes, columns predicted, plus a "background" row/column for misses and false
   * positives. Cells are row-normalised and shaded on one sequential luminance ramp in ink (the diagonal is told apart
   * by position, not hue), counts on hover.
   */
  import { classColor } from '$lib/design/tokens';

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
     * Label only cells of at least 10 % that are at least 32 px wide, so dense matrices don't run numerals together;
     * the readout still gives every value on hover or focus.
     */
    sparseLabels?: boolean;
    /**
     * Minimum cell size in px. Wide matrices then scroll sideways with the row labels pinned, instead of squeezing
     * cells; on touch screens cells are at least 44 px.
     */
    minCell?: number;
    /** Show a labelled 0–100 % colour scale under the matrix. */
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
  const fixed = $derived(minCell !== undefined);
  const onscroll = () => (moreRight = root.scrollLeft + root.clientWidth < root.scrollWidth - 1);
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
</script>

<!-- svelte-ignore a11y_interactive_supports_focus -->
<div class="cm" class:sparse={sparseLabels} class:fixed class:more-r={moreRight} style:--n={n} style:--min-cell={minCell !== undefined ? `${minCell}px` : undefined} role="grid" aria-label={label} aria-describedby="{uid}-readout" bind:this={root} {onkeydown} {onscroll}>
  <div class="row" role="row">
    <div class="corner" role="columnheader"><span>true ↓ / pred →</span></div>
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
          class:hl={hovered && (hovered.r === r || hovered.c === c)}
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
          {#if v >= (sparseLabels ? 0.1 : 0.005)}<span class:light={v > 0.4}>{(v * 100).toFixed(v >= 0.1 ? 0 : 1)}</span>{/if}
        </div>
      {/each}
    </div>
  {/each}
</div>
{#if scale}
  <div class="scale" aria-hidden="true">
    <span>Share of row</span>
    <span class="ramp"></span>
    <span class="ticks"><span>0 %</span><span>50 %</span><span>100 %</span></span>
  </div>
{/if}
<p class="readout" class:faint={!hovered} id="{uid}-readout">
  {#if hovered}
    {@const h = hovered}
    <b>{names[h.r]}</b> predicted as <b>{names[h.c]}</b>:
    <span class="num">{(matrix[h.r]?.[h.c] ?? 0).toLocaleString('en-US')}</span> of
    <span class="num">{(rowSums[h.r] ?? 0).toLocaleString('en-US')}</span>
    (<span class="num">{(norm(h.r, h.c) * 100).toFixed(1)} %</span>)
  {:else}
    Row-normalised, in %{sparseLabels ? '; cells under 10 % unlabelled' : ''}. Hover, tap or arrow through the cells for counts.
  {/if}
</p>

<style>
  .row {
    display: contents;
  }
  .cm {
    display: grid;
    grid-template-columns: minmax(64px, 96px) repeat(var(--n), minmax(22px, 1fr));
    gap: 2px;
    font-size: var(--type-xs-size);
    overflow-x: auto;
  }
  /* Fixed minimum cells: the matrix scrolls sideways under pinned row labels. */
  .cm.fixed {
    /* Cells grow with the panel up to 56 px, so a wide panel does not turn the matrix into a wall. */
    grid-template-columns: 96px repeat(var(--n), minmax(var(--min-cell), 56px));
    overscroll-behavior-x: contain;
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
  .corner {
    display: flex;
    align-items: flex-end;
    color: var(--text-tertiary);
    font-size: var(--type-micro-size);
    padding: 0 0 4px;
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
    font-size: var(--type-xs-size);
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
  .cell.hl {
    box-shadow: inset 0 0 0 1px var(--line-strong);
  }
  .cell span {
    color: var(--text-primary);
  }
  /* Past 40 % the cell is closer to ink than to the well: labels flip to the inverse text colour. */
  .cell span.light {
    color: var(--text-inverse);
    font-weight: var(--weight-medium);
  }
  .readout {
    margin-top: var(--space-3);
    font-size: var(--type-xs-size);
    color: var(--text-secondary);
  }
  .readout b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .num {
    font-variant-numeric: tabular-nums;
  }
  .scale {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    align-items: center;
    gap: 4px var(--space-2);
    max-width: 320px;
    margin-top: var(--space-3);
    font-size: var(--type-xs-size);
    line-height: var(--type-xs-line);
    color: var(--text-tertiary);
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
    grid-column: 2;
    display: flex;
    justify-content: space-between;
    font-variant-numeric: tabular-nums;
  }
</style>
