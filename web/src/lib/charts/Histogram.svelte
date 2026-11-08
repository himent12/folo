<script lang="ts">
  /**
   * Vertical bars for a histogram (`edges` + `counts`) with per-bar hover. By default every bin gets the same width;
   * `proportional` places bins on a linear value axis instead, so unequal bins keep their true widths (bar heights are
   * then densities, so each bar's area is its count).
   */
  import { scaleLinear } from 'd3-scale';
  import { ticks as niceTicks } from './downsample';

  let {
    edges,
    counts,
    height = 160,
    color = 'var(--chart-2)',
    hoverColor,
    binLabel,
    label,
    unit = 'boxes',
    proportional = false,
    xMax,
    edgeLabel,
    marker,
    densityPer,
  }: {
    edges: number[];
    counts: number[];
    height?: number;
    color?: string;
    /** Fill of the hovered bar. Without it, hovering dims the other bars instead. */
    hoverColor?: string;
    /** Label for bin i (defaults to "a–b"). */
    binLabel?: (lo: number, hi: number, i: number) => string;
    label: string;
    unit?: string;
    /** Bin widths proportional to their value span (a linear x axis labelled at the bin edges). */
    proportional?: boolean;
    /** Proportional mode: where the x axis ends; a last bin reaching past it is drawn open-ended up to here. */
    xMax?: number;
    /** Proportional mode: x-axis label of an edge value. */
    edgeLabel?: (v: number) => string;
    /** A dashed vertical reference line at a value (e.g. a threshold), with a short label. */
    marker?: { x: number; label: string };
    /** Proportional mode: bar heights are counts per this value span (defaults to the narrowest bin). */
    densityPer?: number;
  } = $props();

  let width = $state(0);
  let hovered = $state<number | null>(null);

  const m = { top: 8, right: 4, bottom: 20, left: 36 };
  const lo0 = $derived(edges[0] ?? 0);
  const hiN = $derived(Math.min(edges[edges.length - 1] ?? 1, xMax ?? Infinity));
  /** Bin i's value span (the true one, also for an open-ended last bin). */
  const span = (i: number) => Math.max(1e-12, (edges[i + 1] ?? 0) - (edges[i] ?? 0));
  // Proportional bars show density (count per `densityPer`), so wide bins don't look bigger than they are.
  const unitSpan = $derived(densityPer ?? Math.min(...counts.map((_, i) => span(i))));
  const heightOf = (i: number) => (proportional ? (counts[i] ?? 0) / (span(i) / unitSpan) : (counts[i] ?? 0));
  const max = $derived(Math.max(1e-9, ...counts.map((_, i) => heightOf(i))));
  const y = $derived(scaleLinear().domain([0, max]).nice(4).range([height - m.bottom, m.top]));
  const yt = $derived(niceTicks(0, y.domain()[1] ?? max, 3));
  const bw = $derived(counts.length ? (width - m.left - m.right) / counts.length : 0);
  const xv = $derived(scaleLinear().domain([lo0, hiN]).range([m.left, width - m.right]).clamp(true));
  /** Left edge and width of bar i in px. */
  const bar = (i: number) => {
    if (!proportional) return { x: m.left + i * bw, w: bw };
    const a = xv(edges[i] ?? 0);
    return { x: a, w: Math.max(1, xv(edges[i + 1] ?? 0) - a) };
  };
  /** Pixel x of a value: linear in proportional mode, interpolated inside its bin otherwise. */
  const xAt = (v: number) => {
    if (proportional) return xv(v);
    const i = Math.max(0, edges.findIndex((e, k) => v >= e && v < (edges[k + 1] ?? Infinity)));
    const lo = edges[i] ?? 0;
    return m.left + (i + (v - lo) / span(i)) * bw;
  };
  const name = (i: number) =>
    binLabel ? binLabel(edges[i] ?? 0, edges[i + 1] ?? 0, i) : `${edges[i]}–${edges[i + 1]}`;
  const total = $derived(counts.reduce((a, b) => a + b, 0));
  // One notation per axis: all ticks in thousands once the axis reaches 1,000; zero is always plain "0".
  const tickFmt = (v: number) => (v !== 0 && (y.domain()[1] ?? 0) >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(+v.toFixed(2)));
  // Proportional mode labels the edges, skipping any closer than 28 px to the previous label.
  const edgeTicks = $derived.by(() => {
    if (!proportional) return [];
    const out: number[] = [];
    let last = -Infinity;
    for (const e of edges) {
      if (e > hiN) break;
      const px = xv(e);
      if (px - last >= 28) {
        out.push(e);
        last = px;
      }
    }
    return out;
  });
</script>

<div class="hist" bind:clientWidth={width} style:height="{height}px">
  {#if width > 0}
    <svg {width} {height} role="img" aria-label={label}>
      {#each yt as t (t)}
        <line class="grid" x1={m.left} x2={width - m.right} y1={y(t)} y2={y(t)} />
        <text class="tick" x={m.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{tickFmt(t)}</text>
      {/each}
      {#each counts as c, i (i)}
        {@const h = Math.max(c > 0 ? 2 : 0, height - m.bottom - y(heightOf(i)))}
        {@const b = bar(i)}
        {@const gap = proportional ? Math.min(1, b.w / 4) : 1}
        {@const r = Math.min(2, Math.max(0, (b.w - 2 * gap) / 2))}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <g onpointerenter={() => (hovered = i)} onpointerleave={() => (hovered = null)}>
          <rect class="hit" x={b.x} y={m.top} width={b.w} height={height - m.top - m.bottom} />
          <path
            class="bar"
            class:dim={!hoverColor && hovered !== null && hovered !== i}
            style:fill={hoverColor && hovered === i ? hoverColor : color}
            d="M{b.x + gap},{height - m.bottom} v{-Math.max(0, h - r)} q0,-{r} {r},-{r} h{Math.max(0, b.w - 2 * gap - 2 * r)} q{r},0 {r},{r} v{Math.max(0, h - r)} z"
          />
        </g>
        {#if !proportional && (bw > 34 || i % Math.ceil(34 / Math.max(bw, 1)) === 0)}
          <text class="tick" x={m.left + i * bw + bw / 2} y={height - 6} text-anchor="middle">{name(i)}</text>
        {/if}
      {/each}
      {#each edgeTicks as e (e)}
        <text class="tick" x={xv(e)} y={height - 6} text-anchor={e === lo0 ? 'start' : 'middle'}>{edgeLabel ? edgeLabel(e) : e}</text>
      {/each}
      {#if marker}
        {@const mx = Math.round(xAt(marker.x)) + 0.5}
        <line class="marker" x1={mx} x2={mx} y1={m.top} y2={height - m.bottom} />
        <!-- The label sits on the threshold's lower side: "tiny" names what lies left of the line. -->
        <text class="marker-label" x={mx - 4} y={m.top + 8} text-anchor="end">{marker.label}</text>
      {/if}
      <line class="axis" x1={m.left} x2={width - m.right} y1={height - m.bottom} y2={height - m.bottom} />
    </svg>
    {#if hovered !== null}
      {@const c = counts[hovered] ?? 0}
      {@const b = bar(hovered)}
      <div class="tip" style:left="{b.x + b.w / 2}px" style:top="{y(heightOf(hovered)) - 6}px">
        <span class="t">{name(hovered)}</span>
        <span class="v">{c.toLocaleString('en-US')} {unit}</span>
        <span class="p">{total ? ((c / total) * 100).toFixed(1) : '0'} %</span>
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
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }
  .marker {
    stroke: var(--text-secondary);
    stroke-dasharray: 3 3;
    shape-rendering: crispEdges;
  }
  .marker-label {
    fill: var(--text-secondary);
    font-size: var(--type-micro-size);
    font-variant-numeric: tabular-nums;
  }
  .hit {
    fill: transparent;
  }
  .bar {
    transition:
      opacity var(--dur-fast) var(--ease-standard),
      fill var(--dur-fast) var(--ease-standard);
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
  }
  .v,
  .p {
    font-variant-numeric: tabular-nums;
  }
  .p {
    color: var(--text-secondary);
  }
</style>
