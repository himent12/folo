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

  let hovered = $state<{ r: number; c: number } | null>(null);
  const names = $derived([...classes, 'background']);
  const rowSums = $derived(matrix.map((row) => row.reduce((a, b) => a + b, 0)));
  const norm = (r: number, c: number) => {
    const s = rowSums[r] ?? 0;
    return s ? (matrix[r]?.[c] ?? 0) / s : 0;
  };
  const n = $derived(names.length);
</script>

<div class="cm" style:--n={n} role="table" aria-label={label}>
  <div class="corner" role="presentation"><span>true ↓ / pred →</span></div>
  {#each names as name, c (c)}
    <div class="colhead" role="columnheader" title={name}>
      <span class="dot" style:background={c < classes.length ? classColor(c) : 'var(--text-tertiary)'}></span>
      <span class="lbl">{name}</span>
    </div>
  {/each}
  {#each names as rname, r (r)}
    <div class="rowhead" role="rowheader" title={rname}>
      <span class="dot" style:background={r < classes.length ? classColor(r) : 'var(--text-tertiary)'}></span>
      <span class="lbl">{rname}</span>
    </div>
    {#each names as cname, c (c)}
      {@const v = norm(r, c)}
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div
        class="cell"
        class:diag={r === c && r < classes.length}
        class:hl={hovered && (hovered.r === r || hovered.c === c)}
        role="cell"
        tabindex="-1"
        aria-label="{rname} predicted as {cname}: {(v * 100).toFixed(1)}%"
        style:--v={v}
        onpointerenter={() => (hovered = { r, c })}
        onpointerleave={() => (hovered = null)}
      >
        {#if v >= 0.005}<span class:light={v > 0.55}>{(v * 100).toFixed(v >= 0.1 ? 0 : 1)}</span>{/if}
      </div>
    {/each}
  {/each}
</div>
{#if hovered}
  {@const h = hovered}
  <p class="readout">
    <b>{names[h.r]}</b> predicted as <b>{names[h.c]}</b>:
    <span class="mono">{(matrix[h.r]?.[h.c] ?? 0).toLocaleString('en-US')}</span> of
    <span class="mono">{(rowSums[h.r] ?? 0).toLocaleString('en-US')}</span>
    (<span class="mono">{(norm(h.r, h.c) * 100).toFixed(1)}%</span>)
  </p>
{:else}
  <p class="readout faint">Row-normalised, in %. Hover a cell for counts.</p>
{/if}

<style>
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
    font-size: 10px;
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
  :global([data-theme='light']) .cell span.light {
    color: #fff;
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
