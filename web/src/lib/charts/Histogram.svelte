<script lang="ts">
  /** Vertical bars for a histogram (`edges` + `counts`) with per-bar hover. */
  import { scaleLinear } from 'd3-scale';
  import { ticks as niceTicks } from './downsample';

  let {
    edges,
    counts,
    height = 160,
    color = 'var(--chart-2)',
    binLabel,
    label,
    unit = 'boxes',
  }: {
    edges: number[];
    counts: number[];
    height?: number;
    color?: string;
    /** Label for bin i (defaults to "a–b"). */
    binLabel?: (lo: number, hi: number, i: number) => string;
    label: string;
    unit?: string;
  } = $props();

  let width = $state(0);
  let hovered = $state<number | null>(null);

  const m = { top: 8, right: 4, bottom: 20, left: 36 };
  const max = $derived(Math.max(1, ...counts));
  const y = $derived(scaleLinear().domain([0, max]).nice(4).range([height - m.bottom, m.top]));
  const yt = $derived(niceTicks(0, y.domain()[1] ?? max, 3));
  const bw = $derived(counts.length ? (width - m.left - m.right) / counts.length : 0);
  const name = (i: number) =>
    binLabel ? binLabel(edges[i] ?? 0, edges[i + 1] ?? 0, i) : `${edges[i]}–${edges[i + 1]}`;
  const total = $derived(counts.reduce((a, b) => a + b, 0));
  // One notation per axis: all ticks in thousands once the axis reaches 1,000.
  const tickFmt = (v: number) => ((y.domain()[1] ?? 0) >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(v));
</script>

<div class="hist" bind:clientWidth={width} style:height="{height}px">
  {#if width > 0}
    <svg {width} {height} role="img" aria-label={label}>
      {#each yt as t (t)}
        <line class="grid" x1={m.left} x2={width - m.right} y1={y(t)} y2={y(t)} />
        <text class="tick" x={m.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{tickFmt(t)}</text>
      {/each}
      {#each counts as c, i (i)}
        {@const h = Math.max(c > 0 ? 2 : 0, height - m.bottom - y(c))}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <g onpointerenter={() => (hovered = i)} onpointerleave={() => (hovered = null)}>
          <rect class="hit" x={m.left + i * bw} y={m.top} width={bw} height={height - m.top - m.bottom} />
          <path
            class="bar"
            class:dim={hovered !== null && hovered !== i}
            style:fill={color}
            d="M{m.left + i * bw + 1},{height - m.bottom} v{-Math.max(0, h - 2)} q0,-2 2,-2 h{Math.max(0, bw - 6)} q2,0 2,2 v{Math.max(0, h - 2)} z"
          />
        </g>
        {#if bw > 34 || i % Math.ceil(34 / Math.max(bw, 1)) === 0}
          <text class="tick" x={m.left + i * bw + bw / 2} y={height - 6} text-anchor="middle">{name(i)}</text>
        {/if}
      {/each}
      <line class="axis" x1={m.left} x2={width - m.right} y1={height - m.bottom} y2={height - m.bottom} />
    </svg>
    {#if hovered !== null}
      {@const c = counts[hovered] ?? 0}
      <div class="tip" style:left="{m.left + hovered * bw + bw / 2}px" style:top="{y(c) - 6}px">
        <span class="t">{name(hovered)}</span>
        <span class="v">{c.toLocaleString('en-US')} {unit}</span>
        <span class="p">{total ? ((c / total) * 100).toFixed(1) : '0'}%</span>
      </div>
    {/if}
  {/if}
</div>

<style>
  .hist {
    position: relative;
    width: 100%;
  }
  svg {
    display: block;
    overflow: visible;
  }
  .grid {
    stroke: var(--chart-grid);
    shape-rendering: crispEdges;
  }
  .axis {
    stroke: var(--chart-axis);
    shape-rendering: crispEdges;
  }
  .tick {
    fill: var(--chart-label);
    font-family: var(--font-mono);
    font-size: 10px;
  }
  .hit {
    fill: transparent;
  }
  .bar {
    transition: opacity var(--dur-fast) var(--ease-standard);
  }
  .bar.dim {
    opacity: 0.45;
  }
  .tip {
    position: absolute;
    transform: translate(-50%, -100%);
    display: grid;
    gap: 1px;
    padding: 6px 8px;
    border-radius: var(--radius-md);
    background: var(--bg-overlay);
    box-shadow: var(--shadow-md);
    font-size: var(--type-xs-size);
    line-height: var(--type-xs-line);
    white-space: nowrap;
    pointer-events: none;
    z-index: 2;
  }
  .t {
    color: var(--text-tertiary);
    font-family: var(--font-mono);
  }
  .v {
    font-variant-numeric: tabular-nums;
  }
  .p {
    color: var(--text-secondary);
    font-family: var(--font-mono);
  }
</style>
