<script lang="ts">
  /**
   * The plot the benchmark screens draw their results on: lines and dots on two linear scales, with what reading them
   * needs as first-class marks: intervals (whiskers with caps), a band across the plot (a 95% interval every point is
   * read against), reference levels and the diagonal, labelled points, and a strip of counts under the plot. It keeps
   * Studio's chart conventions (LineChart, Histogram): recessive grid and axis, 11 px ticks, both axes named at the axes
   * where both run 0–1, the readout row above the plot as the legend (the caller's snippet, given the cursor), no
   * floating tooltip; a mouse reads where it points, touch pins (./pin), the keyboard steps through `stops` with the
   * arrows, Home and End. Assistive tech gets the chart's own table (`table`).
   */
  import type { Snippet } from 'svelte';
  import { scaleLinear } from 'd3-scale';
  import { seriesText } from './ink';
  import { TouchGesture, type PinAction } from './pin';
  import { chartHint, claimOnce, TAP_HINT_KEY } from './hint';
  import type { XYMark, XYPoint, XYRef, XYSeries, XYStrip } from './xy';

  let {
    label,
    series,
    xDomain,
    yDomain,
    xTicks,
    yTicks,
    xFormat = (v: number) => String(v),
    yFormat = (v: number) => String(v),
    xLabel,
    yLabel,
    height = 240,
    refs = [],
    band = null,
    marks = [],
    strip = null,
    stops: stopsProp,
    cursor = $bindable(null),
    highlight = null,
    readout,
    table,
  }: {
    label: string;
    series: XYSeries[];
    xDomain: [number, number];
    yDomain: [number, number];
    xTicks: number[];
    yTicks: number[];
    xFormat?: (v: number) => string;
    yFormat?: (v: number) => string;
    /** Named at the axis (under the x ticks): for a chart of one 0–1 quantity against another. */
    xLabel?: string;
    /** Named over the y ticks. */
    yLabel?: string;
    /** The svg's height, axes included (the strip adds its own). */
    height?: number;
    refs?: XYRef[];
    /** A band across the plot on y, e.g. the first teacher's 95% interval; its label sits inside it, top left. */
    band?: { lo: number; hi: number; label: string } | null;
    marks?: XYMark[];
    strip?: XYStrip | null;
    /** The x values the cursor snaps to (default: every point's x). */
    stops?: number[];
    /** The x stop being read (hover, a pin, the keyboard), or null. */
    cursor?: number | null;
    /** A series to bring forward: the others step back to context lines (1 px, chart.context, never faded). */
    highlight?: string | null;
    /** The readout row above the plot: the legend at rest (`null`), the values at the cursor's stop. */
    readout: Snippet<[number | null, string | null]>;
    table: { caption: string; head: string[]; rows: string[][] };
  } = $props();

  let width = $state(0);
  const STRIP_H = 28;
  const stops = $derived(
    (stopsProp ?? [...new Set(series.flatMap((s) => s.points.filter((p) => p.y !== null).map((p) => p.x)))]).slice().sort((a, b) => a - b),
  );
  const m = $derived.by(() => {
    const longest = Math.max(1, ...yTicks.map((t) => yFormat(t).length), strip ? String(Math.max(0, ...strip.items.map((i) => i.v))).length : 0);
    return {
      top: yLabel ? 24 : 10,
      right: 14,
      bottom: 22 + (xLabel ? 16 : 0) + (strip ? STRIP_H + 12 : 0),
      left: Math.ceil(longest * 7) + 14,
    };
  });
  const plotBottom = $derived(height - m.bottom);
  const x = $derived(scaleLinear().domain(xDomain).range([m.left, Math.max(m.left + 1, width - m.right)]));
  const y = $derived(scaleLinear().domain(yDomain).range([plotBottom, m.top]));
  const clampY = (v: number) => Math.max(yDomain[0], Math.min(yDomain[1], v));
  const inkOf = (s: XYSeries) => (s.pattern ? seriesText(s.color) : s.color);
  const DASH = { dash: '6 3', dot: '0.01 4.5' } as const;
  const receded = (s: XYSeries) => highlight !== null && s.id !== highlight && series.some((t) => t.id === highlight);

  /** A series' line as an SVG path, broken where a point has no value. */
  function pathOf(s: XYSeries): string {
    let d = '';
    let pen = false;
    for (const p of s.points) {
      if (p.y === null || !Number.isFinite(p.y)) {
        pen = false;
        continue;
      }
      d += `${pen ? 'L' : 'M'}${x(p.x).toFixed(1)} ${y(clampY(p.y)).toFixed(1)}`;
      pen = true;
    }
    return d;
  }
  /** Painted last: the highlighted series, then patterned ones over solid ones. */
  const ordered = $derived([...series].sort((a, b) => Number(a.id === highlight) - Number(b.id === highlight) || Number(!!a.pattern) - Number(!!b.pattern)));

  const stripMax = $derived(strip ? Math.max(1, ...strip.items.map((i) => i.v)) : 1);

  // ---- reading: the mouse, a touch pin, the keyboard ----
  const nearestStop = (px: number) => {
    if (!stops.length) return null;
    let best = stops[0]!;
    for (const s of stops) if (Math.abs(x(s) - px) < Math.abs(x(best) - px)) best = s;
    return best;
  };
  let pinned = $state(false);
  let gesture: TouchGesture | null = null;
  let svgEl = $state<SVGSVGElement>();
  const local = (e: PointerEvent) => {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    return { px: e.clientX - r.left, py: e.clientY - r.top };
  };
  const touchy = (e: PointerEvent) => e.pointerType === 'touch' || e.pointerType === 'pen';
  const inPlot = (px: number) => px >= m.left - 8 && px <= width - m.right + 8;
  function apply(a: PinAction) {
    if (a.kind === 'clear') {
      cursor = null;
      pinned = false;
    } else if (a.kind === 'set' && inPlot(a.px)) {
      cursor = nearestStop(a.px);
      pinned = true;
      tapHint = false;
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
    gesture = new TouchGesture(px, py, pinned && cursor !== null ? x(cursor) : null);
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
    cursor = inPlot(px) && py >= 0 && py <= height ? nearestStop(px) : null;
  }
  function onpointerleave(e: PointerEvent) {
    if (!touchy(e) && !pinned) cursor = null;
  }
  let spoken = $state('');
  function onkeydown(e: KeyboardEvent) {
    if (!stops.length) return;
    const i = cursor === null ? -1 : stops.indexOf(cursor);
    let j = i;
    if (e.key === 'ArrowRight') j = i < 0 ? 0 : Math.min(stops.length - 1, i + (e.shiftKey ? 10 : 1));
    else if (e.key === 'ArrowLeft') j = i < 0 ? stops.length - 1 : Math.max(0, i - (e.shiftKey ? 10 : 1));
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = stops.length - 1;
    else if (e.key === 'Escape') {
      cursor = null;
      return;
    } else return;
    e.preventDefault();
    cursor = stops[j]!;
    const row = table.rows.find((r) => r[0] === xFormat(cursor!));
    spoken = row ? row.map((c, k) => (k ? `${table.head[k]} ${c}` : c)).join(', ') : '';
  }
  // The reading hints of every chart (./hint): "← → to inspect" on keyboard focus, "Tap to pin a value" once on touch.
  let keyFocus = $state(false);
  let tapHint = $state(typeof matchMedia !== 'undefined' && matchMedia('(pointer: coarse)').matches && claimOnce(TAP_HINT_KEY));
  const hint = $derived(chartHint({ reading: cursor !== null, keyFocus, tap: tapHint }));
  function onfocus() {
    try {
      keyFocus = !!svgEl?.matches(':focus-visible');
    } catch {
      keyFocus = false;
    }
  }

  /** The point of `s` at stop `v`, if it has one there. */
  const at = (s: XYSeries, v: number) => s.points.find((p) => p.x === v && p.y !== null);
  const markPos = (k: XYMark) => {
    const px = x(k.x);
    const py = y(clampY(k.y));
    const side = k.side ?? 'right';
    if (side === 'left') return { px, py, tx: px - 9, ty: py, anchor: 'end' as const };
    if (side === 'above') return { px, py, tx: px, ty: py - 12, anchor: 'middle' as const };
    if (side === 'below') return { px, py, tx: px, ty: py + 15, anchor: 'middle' as const };
    return { px, py, tx: px + 9, ty: py, anchor: 'start' as const };
  };
</script>

<div class="xy">
  <div class="readout num" aria-hidden="true">{@render readout(cursor, hint)}</div>
  <p class="sr-only" aria-live="polite">{spoken}</p>
  <div class="sr-only">
    <table>
      <caption>{table.caption}</caption>
      <thead><tr>{#each table.head as h, i (i)}<th scope="col">{h}</th>{/each}</tr></thead>
      <tbody>{#each table.rows as r, i (i)}<tr>{#each r as c, k (k)}{#if k === 0}<th scope="row">{c}</th>{:else}<td>{c}</td>{/if}{/each}</tr>{/each}</tbody>
    </table>
  </div>
  <div class="plot" bind:clientWidth={width} style:height="{height}px">
    {#if width > 0}
      <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
      <svg
        bind:this={svgEl}
        {width}
        {height}
        role="group"
        aria-roledescription="chart"
        aria-label="{label}. Focus and use the arrow keys to read it point by point."
        tabindex="0"
        {onpointermove}
        {onpointerleave}
        {onpointerdown}
        {onpointerup}
        {onpointercancel}
        {onkeydown}
        {onfocus}
        onblur={() => {
          keyFocus = false;
          if (!pinned) cursor = null;
        }}
      >
        {#if band}
          <rect class="band" x={m.left} width={Math.max(0, width - m.right - m.left)} y={y(clampY(band.hi))} height={Math.max(1, y(clampY(band.lo)) - y(clampY(band.hi)))} />
        {/if}
        <g class="grid">
          {#each yTicks as t (t)}<line x1={m.left} x2={width - m.right} y1={y(t)} y2={y(t)} />{/each}
        </g>
        <line class="axis" x1={m.left} x2={width - m.right} y1={plotBottom} y2={plotBottom} />
        <g class="ticks">
          {#each yTicks as t (t)}<text x={m.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{yFormat(t)}</text>{/each}
          {#each xTicks as t (t)}<text x={x(t)} y={height - (xLabel ? 20 : 4)} text-anchor="middle">{xFormat(t)}</text>{/each}
          {#if xLabel}<text class="axis-name" x={(m.left + width - m.right) / 2} y={height - 2} text-anchor="middle">{xLabel}</text>{/if}
          {#if yLabel}<text class="axis-name" x="0" y="8" dy="0.32em" text-anchor="start">{yLabel}</text>{/if}
        </g>
        {#if band}
          <!-- Inside the band at its lower right: clear of the first point's whiskers, which start at the left. -->
          <text class="band-label" x={width - m.right - 6} y={y(clampY(band.lo)) - 6} text-anchor="end">{band.label}</text>
        {/if}
        {#each refs as r, i (i)}
          {#if r.axis === 'diagonal'}
            {@const lo = Math.max(xDomain[0], yDomain[0])}
            {@const hi = Math.min(xDomain[1], yDomain[1])}
            <line class="ref" x1={x(lo)} y1={y(lo)} x2={x(hi)} y2={y(hi)} />
            {#if r.label}<text class="ref-label" x={x(hi) - 4} y={y(hi) + 14} text-anchor="end">{r.label}</text>{/if}
          {:else if r.axis === 'y'}
            <line class="ref" x1={m.left} x2={width - m.right} y1={y(r.value)} y2={y(r.value)} />
            {#if r.label}<text class="ref-label" x={width - m.right - 2} y={y(r.value) + (r.below ? 15 : -5)} text-anchor="end">{r.label}</text>{/if}
          {:else}
            <line class="ref" x1={x(r.value)} x2={x(r.value)} y1={m.top} y2={plotBottom} />
            {#if r.label}<text class="ref-label" x={x(r.value) + 5} y={plotBottom - 6}>{r.label}</text>{/if}
          {/if}
        {/each}
        {#if strip}
          <!-- What each span stands on, on the plot's x scale: bars from the strip's base, the most at its top. -->
          {@const base = plotBottom + 12 + STRIP_H}
          <line class="strip-base" x1={m.left} x2={width - m.right} y1={base} y2={base} />
          <text class="tick-t" x={m.left - 8} y={plotBottom + 12} dy="0.7em" text-anchor="end">{stripMax}</text>
          {#each strip.items as it, i (i)}
            {@const h = it.v > 0 ? Math.max(1, (it.v / stripMax) * STRIP_H) : 0}
            {@const gap = 1}
            <rect class="strip-bar" class:on={cursor !== null && cursor >= it.x0 && cursor < it.x1 + 1e-9} x={x(it.x0) + gap} width={Math.max(0, x(it.x1) - x(it.x0) - 2 * gap)} y={base - h} height={h} rx="1" />
          {/each}
        {/if}
        <g class="data">
          {#each ordered as s (s.id)}
            {#if s.line !== false}
              <path
                class="line"
                class:thin={s.thin}
                class:context={receded(s)}
                class:hl={highlight === s.id}
                class:dotted={s.pattern === 'dot'}
                d={pathOf(s)}
                style:stroke={receded(s) ? undefined : inkOf(s)}
                style:stroke-dasharray={s.pattern ? DASH[s.pattern] : undefined}
              />
            {/if}
            {#each s.points as p, i (i)}
              {#if p.y !== null && (p.lo !== undefined || p.hi !== undefined)}
                {@const px = x(p.x)}
                {@const lo = y(clampY(p.lo ?? p.y))}
                {@const hi = y(clampY(p.hi ?? p.y))}
                <g class="whisker" class:context={receded(s)} style:stroke={receded(s) ? undefined : inkOf(s)}>
                  <line x1={px} x2={px} y1={lo} y2={hi} />
                  <line x1={px - 4} x2={px + 4} y1={lo} y2={lo} />
                  <line x1={px - 4} x2={px + 4} y1={hi} y2={hi} />
                </g>
              {/if}
              {#if p.y !== null && (s.dots || p.lo !== undefined || p.r !== undefined || p.hollow)}
                <circle
                  class="pt"
                  class:hollow={p.hollow}
                  cx={x(p.x)}
                  cy={y(clampY(p.y))}
                  r={p.r ?? 3.5}
                  style:fill={p.hollow ? undefined : receded(s) ? 'var(--chart-context)' : inkOf(s)}
                  style:stroke={p.hollow ? (receded(s) ? 'var(--chart-context)' : inkOf(s)) : undefined}
                />
              {/if}
            {/each}
          {/each}
        </g>
        {#each marks as k, i (i)}
          {@const q = markPos(k)}
          <g class="mark">
            <circle class="ring" cx={q.px} cy={q.py} r="5" />
            <text x={q.tx} y={q.ty} dy="0.32em" text-anchor={q.anchor}>{k.label}</text>
          </g>
        {/each}
        {#if cursor !== null}
          <line class="cursor" class:pinned x1={x(cursor)} x2={x(cursor)} y1={m.top} y2={plotBottom} />
          {#each series as s (s.id)}
            {@const p = at(s, cursor)}
            {#if p && p.y !== null}<circle class="dot" cx={x(p.x)} cy={y(clampY(p.y))} r="4" style:fill={inkOf(s)} />{/if}
          {/each}
        {/if}
      </svg>
    {/if}
  </div>
</div>

<style>
  .xy {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  /* The legend row: what is read where, at rest the legend; it may wrap on a phone, never moving the plot sideways. */
  .readout {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    column-gap: var(--space-1);
    min-height: 20px;
    font-size: var(--type-sm-size);
    line-height: 20px;
    color: var(--text-secondary);
  }
  .plot {
    position: relative;
    width: 100%;
  }
  svg {
    display: block;
    overflow: visible;
    touch-action: pan-y;
  }
  /* Focus: the global :focus-visible ring (base.css), never a halo of its own (DESIGN_DIRECTION §2.9). */
  .grid line {
    stroke: var(--chart-grid);
    stroke-width: 1;
    shape-rendering: crispEdges;
  }
  .axis,
  .strip-base {
    stroke: var(--chart-axis);
    shape-rendering: crispEdges;
  }
  .ticks text,
  .tick-t {
    fill: var(--chart-label);
    font-size: var(--type-xs-size); /* small: chart tick */
    font-variant-numeric: tabular-nums;
  }
  /* Axis names are labels, not ticks: 12 px. */
  .ticks .axis-name {
    fill: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .band {
    fill: var(--bg-active);
  }
  .band-label,
  .ref-label {
    fill: var(--text-tertiary);
    font-size: var(--type-sm-size);
    paint-order: stroke;
    stroke: var(--bg-panel);
    stroke-width: 3px;
  }
  .band-label {
    stroke: none;
  }
  /* A reference is a dashed hairline in the axis ink: it reads as "the line to measure against", never as data. */
  .ref {
    stroke: var(--text-tertiary);
    stroke-width: 1;
    stroke-dasharray: 4 3;
  }
  .strip-bar {
    fill: var(--text-tertiary);
    opacity: 0.55;
  }
  .strip-bar.on {
    opacity: 1;
  }
  .line {
    fill: none;
    stroke-width: 1.75;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .line.thin {
    stroke-width: 1;
  }
  .line.dotted {
    stroke-width: 2.25;
  }
  .line.hl {
    stroke-width: 2.75;
  }
  .line.context {
    stroke: var(--chart-context);
    stroke-width: 1;
  }
  .whisker line {
    stroke-width: 1.5;
    stroke-linecap: round;
  }
  .whisker.context {
    stroke: var(--chart-context);
  }
  .pt {
    stroke: var(--bg-panel);
    stroke-width: 1.5;
  }
  /* Too few behind it to trust: an outline in the series ink, the panel inside. */
  .pt.hollow {
    fill: var(--bg-panel);
    stroke-width: 1.5;
  }
  .mark .ring {
    fill: none;
    stroke: var(--text-primary);
    stroke-width: 1.5;
  }
  .mark text {
    fill: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
    paint-order: stroke;
    stroke: var(--bg-panel);
    stroke-width: 3px;
  }
  .cursor {
    stroke: var(--chart-cursor);
    stroke-width: 1;
    stroke-dasharray: 2 2;
    shape-rendering: crispEdges;
  }
  .cursor.pinned {
    stroke-dasharray: none;
  }
  .dot {
    stroke: var(--bg-panel);
    stroke-width: 2;
  }
</style>
