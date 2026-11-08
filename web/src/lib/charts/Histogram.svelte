<script lang="ts">
  /**
   * Vertical bars for a histogram (`edges` + `counts`). By default every bin gets the same width; `proportional` places
   * bins on a linear value axis instead, so unequal bins keep their true widths (bar heights are then densities, so
   * each bar's area is its count, and `yLabel` should say what the height is). The readout row above the plot is the
   * legend, as on the line charts: the total at rest, the bin under the cursor on hover. Touch: a tap or a sideways
   * scrub pins a bin until it is tapped again, a tap lands outside or Escape (./pin). Keyboard: arrow keys.
   */
  import { scaleLinear } from 'd3-scale';
  import { ticks as niceTicks } from './downsample';
  import { binAt, binDensities } from './histogram';
  import { TouchGesture, type PinAction } from './pin';

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
    yLabel,
    note,
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
    /** What the bar height measures, written above the y axis (needed when it is a density, not a count). */
    yLabel?: string;
    /** A short qualifier after the total in the readout at rest (e.g. "bar area is the count"). */
    note?: string;
  } = $props();

  let width = $state(0);
  let hovered = $state<number | null>(null);

  const m = $derived({ top: yLabel ? 24 : 8, right: 4, bottom: 20, left: 36 });
  const lo0 = $derived(edges[0] ?? 0);
  const hiN = $derived(Math.min(edges[edges.length - 1] ?? 1, xMax ?? Infinity));
  /** Bin i's value span (the true one, also for an open-ended last bin). */
  const span = (i: number) => Math.max(1e-12, (edges[i + 1] ?? 0) - (edges[i] ?? 0));
  // Proportional bars show density over the displayed width (count per `densityPer`), so every bar's area is its
  // count, including a tail bin clipped at `xMax`.
  const densities = $derived(proportional ? binDensities(edges, counts, xMax, densityPer) : counts);
  const heightOf = (i: number) => densities[i] ?? 0;
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
  const firstBin = $derived(Math.max(0, counts.findIndex((c) => c > 0)));
  const shareOf = (c: number) => (total ? ((c / total) * 100).toFixed(1) : '0');
  const readout = $derived(hovered === null ? '' : `${name(hovered)}: ${(counts[hovered] ?? 0).toLocaleString('en-US')} ${unit}, ${shareOf(counts[hovered] ?? 0)}%`);

  // Pointer readout: a mouse reads the bin under it; touch pins a bin (./pin) until dismissed.
  let pinned = $state(false);
  let gesture: TouchGesture | null = null;
  let svgEl = $state<SVGSVGElement>();
  const bars = $derived(counts.map((_, i) => bar(i)));
  const local = (e: PointerEvent) => {
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
    return { px: e.clientX - rect.left, py: e.clientY - rect.top };
  };
  const touchy = (e: PointerEvent) => e.pointerType === 'touch' || e.pointerType === 'pen';
  function apply(a: PinAction) {
    if (a.kind === 'clear') {
      hovered = null;
      pinned = false;
    } else if (a.kind === 'set') {
      const i = binAt(bars, a.px);
      if (i !== null) {
        hovered = i;
        pinned = true;
      }
    }
  }
  $effect(() => {
    if (!pinned) return;
    const outside = (e: PointerEvent) => {
      if (svgEl && !svgEl.contains(e.target as Node)) apply({ kind: 'clear' });
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') apply({ kind: 'clear' });
    };
    document.addEventListener('pointerdown', outside, true);
    document.addEventListener('keydown', esc);
    return () => {
      document.removeEventListener('pointerdown', outside, true);
      document.removeEventListener('keydown', esc);
    };
  });
  function onpointerdown(e: PointerEvent) {
    if (!touchy(e)) return;
    const { px, py } = local(e);
    const at = pinned && hovered !== null ? bars[hovered] : undefined;
    gesture = new TouchGesture(px, py, at ? at.x + at.w / 2 : null);
  }
  function onpointerup(e: PointerEvent) {
    if (!touchy(e) || !gesture) return;
    apply(gesture.up(local(e).px));
    gesture = null;
  }
  function onpointercancel() {
    if (gesture) apply(gesture.cancel());
    gesture = null;
  }
  function onpointermove(e: PointerEvent) {
    const { px, py } = local(e);
    if (touchy(e)) {
      if (gesture) apply(gesture.move(px, py));
      return;
    }
    pinned = false;
    hovered = py >= m.top && py <= height - m.bottom ? binAt(bars, px) : null;
  }
  function onpointerleave(e: PointerEvent) {
    if (!touchy(e) && !pinned) hovered = null;
  }
  function onkeydown(e: KeyboardEvent) {
    const n = counts.length;
    if (!n) return;
    const i = hovered ?? firstBin;
    let j = i;
    if (e.key === 'ArrowRight') j = Math.min(n - 1, i + 1);
    else if (e.key === 'ArrowLeft') j = Math.max(0, i - 1);
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = n - 1;
    else if (e.key === 'Escape') {
      hovered = null;
      pinned = false;
      return;
    } else return;
    e.preventDefault();
    hovered = j;
  }
  // One notation per axis: all ticks in thousands once the axis reaches 1,000; zero is always plain "0".
  const tickFmt = (v: number) => (v !== 0 && (y.domain()[1] ?? 0) >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(+v.toFixed(2)));
  // Proportional mode labels the edges, skipping any closer than 28 px to the previous label.
  const lastLo = $derived(edges[edges.length - 2] ?? 0);
  const tailClipped = $derived(proportional && (edges[edges.length - 1] ?? 0) > hiN && lastLo < hiN);
  const edgeTicks = $derived.by(() => {
    if (!proportional) return [];
    const out: number[] = [];
    let last = -Infinity;
    for (const e of edges) {
      if (e > hiN) break;
      if (tailClipped && e === lastLo) break;
      const px = xv(e);
      if (px - last >= 28) {
        out.push(e);
        last = px;
      }
    }
    return out;
  });
</script>

<!-- The readout: the legend row of the chart frame, in one stable place (no floating tooltip). -->
<p class="readout num" class:at={hovered !== null} aria-hidden="true">
  {#if hovered !== null}
    {@const c = counts[hovered] ?? 0}
    <span class="t">{name(hovered)}</span>
    <span class="v">{c.toLocaleString('en-US')} {unit}</span>
    <span class="p">{shareOf(c)}%</span>
  {:else}
    <span class="v">{total.toLocaleString('en-US')} {unit}</span>
    {#if note}<span class="t">{note}</span>{/if}
  {/if}
</p>
<div class="hist" bind:clientWidth={width} style:height="{height}px">
  {#if width > 0}
    <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
    <svg
      bind:this={svgEl}
      {width}
      {height}
      role="img"
      aria-label="{label}. Focus and use the arrow keys to read each bin."
      tabindex="0"
      {onkeydown}
      {onpointermove}
      {onpointerleave}
      {onpointerdown}
      {onpointerup}
      {onpointercancel}
      onfocus={() => (hovered ??= firstBin)}
      onblur={() => {
        if (!pinned) hovered = null;
      }}
    >
      {#if yLabel}<text class="y-label" x="0" y="8" dy="0.32em" text-anchor="start">{yLabel}</text>{/if}
      {#each yt as t (t)}
        <line class="grid" x1={m.left} x2={width - m.right} y1={y(t)} y2={y(t)} />
        <text class="tick" x={m.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{tickFmt(t)}</text>
      {/each}
      {#each counts as c, i (i)}
        {@const h = Math.max(c > 0 ? 2 : 0, height - m.bottom - y(heightOf(i)))}
        {@const b = bar(i)}
        {@const gap = proportional ? Math.min(1, b.w / 4) : 1}
        {@const r = Math.min(2, Math.max(0, (b.w - 2 * gap) / 2))}
        <g>
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
      {#if tailClipped}
        <text class="tick" x={(xv(lastLo) + xv(hiN)) / 2} y={height - 6} text-anchor="middle">≥{edgeLabel ? edgeLabel(lastLo) : lastLo}</text>
      {/if}
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
    <p class="sr-only" aria-live="polite">{readout}</p>
    <div class="sr-only">
      <table>
        <caption>{label}</caption>
        <thead><tr><th scope="col">Bin</th><th scope="col">{unit}</th><th scope="col">Share</th></tr></thead>
        <tbody>
          {#each counts as c, i (i)}<tr><th scope="row">{name(i)}</th><td>{c.toLocaleString('en-US')}</td><td>{shareOf(c)}%</td></tr>{/each}
        </tbody>
      </table>
    </div>
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
    border-radius: var(--radius-sm);
    touch-action: pan-y;
  }
  /* The legend row: a fixed 20 px line, so the plot never moves as the readout changes. */
  .readout {
    display: flex;
    align-items: baseline;
    gap: var(--space-2);
    height: 20px;
    margin-bottom: var(--space-2);
    overflow: hidden;
    white-space: nowrap;
    font-size: var(--type-sm-size);
    line-height: 20px;
  }
  .readout .t {
    color: var(--text-tertiary);
  }
  .readout.at .t {
    color: var(--text-secondary);
  }
  .readout .v {
    color: var(--text-primary);
  }
  .readout .p {
    color: var(--text-secondary);
  }
  .y-label {
    fill: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  svg:focus {
    outline: none;
  }
  svg:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: 4px;
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
    font-size: var(--type-xs-size);
    font-variant-numeric: tabular-nums;
  }
  .marker {
    stroke: var(--text-secondary);
    stroke-dasharray: 3 3;
    shape-rendering: crispEdges;
  }
  .marker-label {
    fill: var(--text-secondary);
    font-size: var(--type-xs-size);
    font-variant-numeric: tabular-nums;
  }
  .bar {
    transition:
      opacity var(--dur-fast) var(--ease-standard),
      fill var(--dur-fast) var(--ease-standard);
  }
  .bar.dim {
    opacity: 0.45;
  }
</style>
