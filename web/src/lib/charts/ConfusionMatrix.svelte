<script lang="ts">
  /**
   * Confusion matrix: rows are true classes, columns predicted, plus a "background" row/column for misses and false
   * positives. Cells are row-normalised and shaded in one hue (sequential), counts on hover.
   */
  import { classColor } from '$lib/design/tokens';

  let {
    classes,
    matrix,
    label = 'Confusion matrix',
  }: {
    classes: string[];
    /** (n+1)×(n+1) counts; the last row/column is background. */
    matrix: number[][];
    label?: string;
  } = $props();

  const uid = $props.id();
  let hovered = $state<{ r: number; c: number } | null>(null);
  /** A cell chosen by keyboard or tap keeps the readout when the pointer leaves. */
  let pinned = $state<{ r: number; c: number } | null>(null);
  /** Roving tabindex: one cell is in the tab order; arrow keys move it. */
  let focus = $state({ r: 0, c: 0 });
  let root: HTMLDivElement;

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
</script>

<!-- svelte-ignore a11y_interactive_supports_focus -->
<div class="cm" style:--n={n} role="grid" aria-label={label} aria-describedby="{uid}-readout" bind:this={root} {onkeydown}>
  <div class="row" role="row">
    <div class="corner" role="columnheader"><span>true ↓ / pred →</span></div>
    {#each names as name, c (c)}
      <div class="colhead" role="columnheader" title={name}>
        <span class="dot" style:background={c < classes.length ? classColor(c) : 'var(--text-tertiary)'}></span>
        <span class="lbl">{name}</span>
      </div>
    {/each}
  </div>
  {#each names as rname, r (r)}
    <div class="row" role="row">
      <div class="rowhead" role="rowheader" title={rname}>
        <span class="dot" style:background={r < classes.length ? classColor(r) : 'var(--text-tertiary)'}></span>
        <span class="lbl">{rname}</span>
      </div>
      {#each names as cname, c (c)}
        {@const v = norm(r, c)}
        <div
          class="cell"
          class:diag={r === c && r < classes.length}
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
          {#if v >= 0.005}<span class:light={v > 0.55}>{(v * 100).toFixed(v >= 0.1 ? 0 : 1)}</span>{/if}
        </div>
      {/each}
    </div>
  {/each}
</div>
<p class="readout" class:faint={!hovered} id="{uid}-readout">
  {#if hovered}
    {@const h = hovered}
    <b>{names[h.r]}</b> predicted as <b>{names[h.c]}</b>:
    <span class="mono">{(matrix[h.r]?.[h.c] ?? 0).toLocaleString('en-US')}</span> of
    <span class="mono">{(rowSums[h.r] ?? 0).toLocaleString('en-US')}</span>
    (<span class="mono">{(norm(h.r, h.c) * 100).toFixed(1)}%</span>)
  {:else}
    Row-normalised, in %. Hover, tap or arrow through the cells for counts.
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
    width: 6px;
    height: 6px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .cell {
    aspect-ratio: 1;
    display: grid;
    place-items: center;
    border-radius: 3px;
    background: color-mix(in srgb, var(--chart-2) calc(var(--v) * 92% + 4%), var(--bg-inset));
    font-family: var(--font-mono);
    font-size: var(--type-xs-size);
    color: var(--text-secondary);
    transition: box-shadow var(--dur-fast) var(--ease-standard);
    min-width: 22px;
  }
  .cell.hl {
    box-shadow: inset 0 0 0 1px var(--line-strong);
  }
  .cell.diag {
    background: color-mix(in srgb, var(--chart-1) calc(var(--v) * 92% + 4%), var(--bg-inset));
  }
  .cell span.light {
    color: var(--text-inverse);
    font-weight: var(--weight-medium);
  }
  /* Light theme: cells darken towards the series colour, which keeps dark text above 4.5:1 throughout. */
  :global([data-theme='light']) .cell span,
  :global([data-theme='light']) .cell span.light {
    color: var(--text-primary);
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
  .mono {
    font-family: var(--font-mono);
  }
</style>
