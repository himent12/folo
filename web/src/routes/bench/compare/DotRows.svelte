<script lang="ts" module>
  /** A compared entry as the dot plots draw it: its hue and its shape (the shape says whose dot it is without the hue). */
  export interface DotKey {
    id: string;
    name: string;
    /** `var(--chart-N)`: the mark's ink. */
    color: string;
    shape: 'circle' | 'square' | 'diamond' | 'triangle';
  }
</script>

<script lang="ts">
  /**
   * An aligned dot plot: one row per thing (a condition, a class), one dot per entry on one shared 0–1 scale, so who
   * wins where reads down the rows. Each entry keeps its hue and a shape of its own; a row pointed at (or focused)
   * reads out every entry's value in words under the plot; an entry kept in front (the legend) draws its dots over the
   * others' and the rest step back. The scale's ticks sit over the first row and under the last.
   */
  import { capTicks } from '$lib/charts/domain';
  import { along } from '../leaderboard';

  let {
    rows,
    keys,
    domain,
    format,
    tickFormat,
    label,
    highlight = null,
    sub,
  }: {
    rows: { key: string; name: string; values: (number | null)[]; note?: string }[];
    keys: DotKey[];
    domain: [number, number];
    format: (v: number) => string;
    /** The scale's tick labels (default `format`). */
    tickFormat?: (v: number) => string;
    label: string;
    highlight?: string | null;
    /** A second line under a row's name (a group, a count), small. */
    sub?: (key: string) => string;
  } = $props();

  let pointed = $state<string | null>(null);
  // 3–5 nice ticks inside the scale (./lib/charts/domain), never a crowded last one at its edge.
  const ticks = $derived(capTicks(domain[0], domain[1], 5));
  const pos = (v: number) => `${(along(domain, v) * 100).toFixed(2)}%`;
  const read = $derived(rows.find((r) => r.key === pointed));
  const say = (r: (typeof rows)[number]) => `${r.name}: ${keys.map((k, i) => `${k.name} ${r.values[i] == null ? 'not run' : format(r.values[i]!)}`).join(', ')}`;
  let spoken = $state('');
  function onkeydown(e: KeyboardEvent) {
    if (!rows.length) return;
    const i = rows.findIndex((r) => r.key === pointed);
    let j = i;
    if (e.key === 'ArrowDown') j = Math.min(rows.length - 1, i + 1);
    else if (e.key === 'ArrowUp') j = i < 0 ? 0 : Math.max(0, i - 1);
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = rows.length - 1;
    else if (e.key === 'Escape') {
      pointed = null;
      return;
    } else return;
    e.preventDefault();
    pointed = rows[j]!.key;
    spoken = say(rows[j]!);
  }
  /** Painted last: the entry kept in front. */
  const order = $derived(keys.map((k, i) => ({ k, i })).sort((a, b) => Number(a.k.id === highlight) - Number(b.k.id === highlight)));
</script>

{#snippet mark(k: DotKey, v: number, dim: boolean)}
  <svg class="m" class:dim style:left={pos(v)} width="12" height="12" viewBox="-6 -6 12 12" aria-hidden="true">
    {#if k.shape === 'circle'}<circle r="4.5" style:fill={k.color} />
    {:else if k.shape === 'square'}<rect x="-4" y="-4" width="8" height="8" style:fill={k.color} />
    {:else if k.shape === 'diamond'}<path d="M0 -5.5 L5.5 0 L0 5.5 L-5.5 0 Z" style:fill={k.color} />
    {:else}<path d="M0 -5 L5 4.5 L-5 4.5 Z" style:fill={k.color} />{/if}
  </svg>
{/snippet}

{#snippet glyph(k: DotKey)}
  <svg class="g" width="12" height="12" viewBox="-6 -6 12 12" aria-hidden="true">
    {#if k.shape === 'circle'}<circle r="4.5" style:fill={k.color} />
    {:else if k.shape === 'square'}<rect x="-4" y="-4" width="8" height="8" style:fill={k.color} />
    {:else if k.shape === 'diamond'}<path d="M0 -5.5 L5.5 0 L0 5.5 L-5.5 0 Z" style:fill={k.color} />
    {:else}<path d="M0 -5 L5 4.5 L-5 4.5 Z" style:fill={k.color} />{/if}
  </svg>
{/snippet}

<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (an interactive chart: one stop, its rows read with the arrow keys) -->
<div class="dots" role="group" aria-roledescription="chart" aria-label="{label}. Focus and use the up and down arrow keys to read each row." tabindex="0" {onkeydown} onblur={() => (pointed = null)}>
  <div class="row axis" aria-hidden="true">
    <span></span>
    <span class="scale">{#each ticks as t (t)}<span class="tk num" style:left={pos(t)}>{(tickFormat ?? format)(t)}</span>{/each}</span>
  </div>
  <ul>
    {#each rows as r (r.key)}
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions (pointing reads the row out; the group's arrow keys do the same) -->
      <li class="row" class:on={pointed === r.key} onpointerenter={() => (pointed = r.key)} onpointerleave={() => (pointed = null)}>
        <span class="name"><span class="nm">{r.name}</span>{#if sub}<span class="sub">{sub(r.key)}</span>{/if}</span>
        <span class="track">
          {#each ticks as t (t)}<span class="grid" style:left={pos(t)}></span>{/each}
          {#each order as { k, i } (k.id)}
            {@const v = r.values[i] ?? null}
            {#if v !== null}{@render mark(k, v, highlight !== null && highlight !== k.id)}{/if}
          {/each}
        </span>
      </li>
    {/each}
  </ul>
  <p class="sr-only" aria-live="polite">{spoken}</p>
  <p class="readout num" aria-hidden="true">
    {#if read}
      <span class="t">{read.name}</span>
      {#each keys as k, i (k.id)}<span class="rv">{@render glyph(k)}{read.values[i] === null ? 'not run' : format(read.values[i]!)}</span>{/each}
    {:else}
      <span class="t">Point at a row to read it.</span>
    {/if}
  </p>
</div>

<style>
  .dots {
    display: grid;
    min-width: 0;
  }
  .row {
    display: grid;
    grid-template-columns: minmax(120px, 200px) minmax(0, 1fr);
    align-items: center;
    gap: var(--space-4);
    min-height: 32px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
  }
  li.row {
    border-top: 1px solid var(--line-subtle);
    border-radius: 0;
  }
  li.row.on {
    background: var(--bg-hover);
  }

  .row.axis {
    min-height: 20px;
  }
  .name {
    display: grid;
    min-width: 0;
  }
  .nm {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .sub {
    font-size: var(--type-sm-size);
    line-height: 14px;
    color: var(--text-tertiary);
  }
  .scale,
  .track {
    position: relative;
    display: block;
    margin: 0 8px;
  }
  .scale {
    height: 16px;
  }
  .track {
    height: 28px;
  }
  .tk {
    position: absolute;
    top: 0;
    transform: translateX(-50%);
    color: var(--chart-label);
    /* small: chart tick */
    font-size: var(--type-xs-size);
  }
  .grid {
    position: absolute;
    top: 2px;
    bottom: 2px;
    width: 1px;
    background: var(--chart-grid);
  }
  .m {
    position: absolute;
    top: 50%;
    margin: -6px 0 0 -6px;
    overflow: visible;
  }
  .m :global(:is(circle, rect, path)) {
    stroke: var(--bg-panel);
    stroke-width: 1.5;
  }
  .m.dim {
    opacity: 0.35;
  }
  .readout {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-4);
    min-height: 32px;
    padding: var(--space-2) var(--space-2) 0;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    border-top: 1px solid var(--line-subtle);
  }
  .t {
    color: var(--text-secondary);
  }
  .rv {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--text-primary);
  }
  .g {
    flex-shrink: 0;
  }
  @media (max-width: 767px) {
    .row {
      grid-template-columns: minmax(84px, 112px) minmax(0, 1fr);
      gap: var(--space-2);
    }
  }
</style>
