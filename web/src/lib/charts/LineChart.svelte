<script lang="ts" module>
  import type { Pt } from './downsample';

  export interface Series {
    id: string;
    label: string;
    /** Any CSS colour, usually a token: `var(--chart-1)`. */
    color: string;
    points: Pt[];
    dashed?: boolean;
    /** Fill the area under the line. */
    area?: boolean;
    /** Thin, recessive line (e.g. raw values under a smoothed line); hidden from legend and tooltip. */
    ghost?: boolean;
  }

  export interface Marker {
    x: number;
    y: number;
    label: string;
    color?: string;
  }
</script>

<script lang="ts">
  import { scaleLinear, scaleLog } from 'd3-scale';
  import { area as d3area, curveLinear, curveMonotoneX, line as d3line } from 'd3-shape';
  import { downsample, nearest, ticks as niceTicks } from './downsample';

  interface Props {
    series: Series[];
    height?: number;
    xDomain?: [number, number];
    yDomain?: [number, number];
    yLog?: boolean;
    curve?: 'linear' | 'monotone';
    xFormat?: (x: number) => string;
    yFormat?: (y: number) => string;
    /** Formats values in the tooltip and legend (defaults to yFormat). */
    valueFormat?: (y: number) => string;
    /** Title row for the tooltip, given the hovered x. */
    xTitle?: (x: number) => string;
    markers?: Marker[];
    legend?: boolean;
    yTicks?: number;
    xTicks?: number;
    /** Shared crosshair position in data x (bind it across charts to sync them). */
    cursor?: number | null;
    label: string;
    empty?: string;
  }

  let {
    series,
    height = 200,
    xDomain,
    yDomain,
    yLog = false,
    curve = 'linear',
    xFormat = (x) => String(Math.round(x)),
    yFormat = (y) => String(+y.toFixed(3)),
    valueFormat,
    xTitle,
    markers = [],
    legend = true,
    yTicks = 4,
    xTicks = 6,
    cursor = $bindable(null),
    label,
    empty = 'No data yet',
  }: Props = $props();

  const uid = $props.id();
  let width = $state(0);
  const fmtValue = $derived(valueFormat ?? yFormat);

  const visible = $derived(series.filter((s) => s.points.length > 0));
  const hasData = $derived(visible.length > 0);

  const xExtent = $derived.by((): [number, number] => {
    if (xDomain) return xDomain;
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of visible) {
      const a = s.points[0];
      const b = s.points[s.points.length - 1];
      if (a && a.x < lo) lo = a.x;
      if (b && b.x > hi) hi = b.x;
    }
    if (!Number.isFinite(lo)) return [0, 1];
    return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
  });

  const yExtent = $derived.by((): [number, number] => {
    if (yDomain) return yDomain;
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of visible)
      for (const p of s.points) {
        if (!Number.isFinite(p.y) || (yLog && p.y <= 0)) continue;
        if (p.y < lo) lo = p.y;
        if (p.y > hi) hi = p.y;
      }
    if (!Number.isFinite(lo)) return yLog ? [0.1, 1] : [0, 1];
    if (yLog) return [lo * 0.9, hi * 1.1];
    const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
    return [lo >= 0 && lo - pad < 0 ? 0 : lo - pad, hi + pad];
  });

  const yTickValues = $derived.by(() => {
    if (yLog) {
      const [lo, hi] = yExtent;
      const out: number[] = [];
      for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
        for (const m of [1, 2, 5]) {
          const v = m * 10 ** e;
          if (v >= lo && v <= hi) out.push(v);
        }
      return out.length > yTicks + 2 ? out.filter((_, i) => i % 2 === 0) : out;
    }
    return niceTicks(yExtent[0], yExtent[1], yTicks);
  });

  const margin = $derived.by(() => {
    const longest = Math.max(...yTickValues.map((v) => yFormat(v).length), 1);
    return { top: 8, right: 12, bottom: 22, left: Math.ceil(longest * 6.4) + 14 };
  });

  const x = $derived(scaleLinear().domain(xExtent).range([margin.left, Math.max(margin.left + 1, width - margin.right)]));
  const y = $derived(
    (yLog ? scaleLog().domain(yExtent) : scaleLinear().domain(yExtent)).range([height - margin.bottom, margin.top]),
  );
  const xTickValues = $derived(niceTicks(xExtent[0], xExtent[1], Math.max(2, Math.min(xTicks, Math.floor(width / 70)))));

  const paths = $derived.by(() => {
    const c = curve === 'monotone' ? curveMonotoneX : curveLinear;
    const buckets = Math.max(32, Math.floor((width - margin.left - margin.right) / 1.5));
    const ok = (p: Pt) => Number.isFinite(p.y) && (!yLog || p.y > 0);
    const ln = d3line<Pt>()
      .defined(ok)
      .x((p) => x(p.x))
      .y((p) => y(p.y))
      .curve(c);
    const ar = d3area<Pt>()
      .defined(ok)
      .x((p) => x(p.x))
      .y0(height - margin.bottom)
      .y1((p) => y(p.y))
      .curve(c);
    return visible.map((s) => {
      const pts = downsample(s.points, buckets);
      return { s, d: ln(pts) ?? '', a: s.area ? (ar(pts) ?? '') : '' };
    });
  });

  // ---- hover / keyboard cursor ----
  const keyed = $derived(visible.filter((s) => !s.ghost));
  const longestSeries = $derived(keyed.reduce<Series | undefined>((m, s) => (!m || s.points.length > m.points.length ? s : m), undefined));

  const hover = $derived.by(() => {
    if (cursor === null || cursor === undefined || !hasData) return null;
    if (cursor < xExtent[0] || cursor > xExtent[1]) return null;
    const rows = keyed
      .map((s) => {
        const i = nearest(s.points, cursor as number);
        const p = s.points[i];
        return p ? { s, p } : null;
      })
      .filter((r): r is { s: Series; p: Pt } => r !== null);
    if (!rows.length) return null;
    const px = x(rows[0]!.p.x);
    return { rows, px, xv: rows[0]!.p.x };
  });

  function onpointermove(e: PointerEvent) {
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
    const px = e.clientX - rect.left;
    if (px < margin.left - 4 || px > width - margin.right + 4) {
      cursor = null;
      return;
    }
    cursor = x.invert(px);
  }

  function onkeydown(e: KeyboardEvent) {
    const s = longestSeries;
    if (!s || !s.points.length) return;
    let i = cursor === null ? s.points.length - 1 : nearest(s.points, cursor);
    const big = e.shiftKey ? 10 : 1;
    if (e.key === 'ArrowLeft') i = Math.max(0, i - big);
    else if (e.key === 'ArrowRight') i = Math.min(s.points.length - 1, i + big);
    else if (e.key === 'Home') i = 0;
    else if (e.key === 'End') i = s.points.length - 1;
    else if (e.key === 'Escape') {
      cursor = null;
      return;
    } else return;
    e.preventDefault();
    cursor = s.points[i]!.x;
  }

  const latest = (s: Series) => {
    for (let i = s.points.length - 1; i >= 0; i--) {
      const p = s.points[i]!;
      if (Number.isFinite(p.y)) return p.y;
    }
    return null;
  };

  const tipLeft = $derived(hover ? hover.px > width * 0.6 : false);
</script>

<div class="chart">
  {#if legend && keyed.length > 0}
    <ul class="legend" aria-hidden="true">
      {#each keyed as s (s.id)}
        {@const v = hover ? hover.rows.find((r) => r.s.id === s.id)?.p.y : latest(s)}
        <li>
          <span class="swatch" class:dashed={s.dashed} style:--c={s.color}></span>
          <span class="name">{s.label}</span>
          <span class="val">{v === null || v === undefined ? '—' : fmtValue(v)}</span>
        </li>
      {/each}
    </ul>
  {/if}

  <div class="plot" bind:clientWidth={width} style:height="{height}px">
    {#if width > 0}
      <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
      <svg
        {width}
        {height}
        role="img"
        aria-label={label}
        tabindex={hasData ? 0 : -1}
        {onpointermove}
        onpointerleave={() => (cursor = null)}
        {onkeydown}
        onblur={() => (cursor = null)}
      >
        <g class="grid">
          {#each yTickValues as t (t)}
            <line x1={margin.left} x2={width - margin.right} y1={y(t)} y2={y(t)} />
          {/each}
        </g>
        <line class="axis" x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} />
        <g class="ticks">
          {#each yTickValues as t (t)}
            <text x={margin.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{yFormat(t)}</text>
          {/each}
          {#each xTickValues as t (t)}
            <text x={x(t)} y={height - 6} text-anchor="middle">{xFormat(t)}</text>
          {/each}
        </g>

        {#if hasData}
          <defs>
            {#each paths as { s, a } (s.id)}
              {#if a}
                <linearGradient id="{uid}-{s.id}" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0" style:stop-color={s.color} stop-opacity="0.28" />
                  <stop offset="1" style:stop-color={s.color} stop-opacity="0" />
                </linearGradient>
              {/if}
            {/each}
          </defs>
          {#each paths as { s, d, a } (s.id)}
            {#if a}<path class="area" d={a} fill="url(#{uid}-{s.id})" />{/if}
            <path class="line" class:dashed={s.dashed} class:ghost={s.ghost} {d} style:stroke={s.color} />
          {/each}

          {#each markers as m (m.label + m.x)}
            {#if m.x >= xExtent[0] && m.x <= xExtent[1]}
              <g class="marker" transform="translate({x(m.x)},{y(m.y)})">
                <circle r="4.5" style:stroke={m.color ?? 'var(--chart-1)'} />
                <text y="-10" text-anchor="middle">{m.label}</text>
              </g>
            {/if}
          {/each}

          {#if hover}
            <line class="cursor" x1={hover.px} x2={hover.px} y1={margin.top} y2={height - margin.bottom} />
            {#each hover.rows as r (r.s.id)}
              {#if Number.isFinite(r.p.y) && (!yLog || r.p.y > 0)}
                <circle class="dot" cx={x(r.p.x)} cy={y(r.p.y)} r="3.5" style:fill={r.s.color} />
              {/if}
            {/each}
          {/if}
        {/if}
      </svg>

      {#if !hasData}
        <p class="empty">{empty}</p>
      {/if}

      {#if hover}
        <div class="tip" class:left={tipLeft} style:--x="{hover.px}px" role="presentation">
          <p class="tip-title">{xTitle ? xTitle(hover.xv) : xFormat(hover.xv)}</p>
          {#each hover.rows as r (r.s.id)}
            <p class="tip-row">
              <span class="swatch" class:dashed={r.s.dashed} style:--c={r.s.color}></span>
              <span class="name">{r.s.label}</span>
              <span class="val">{fmtValue(r.p.y)}</span>
            </p>
          {/each}
        </div>
      {/if}
    {/if}
  </div>
</div>

<style>
  .chart {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
    font-size: var(--type-xs-size);
    line-height: var(--type-xs-line);
  }
  .legend li {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .swatch {
    width: 12px;
    height: 2px;
    border-radius: 1px;
    background: var(--c);
    flex-shrink: 0;
  }
  .swatch.dashed {
    background: repeating-linear-gradient(90deg, var(--c) 0 4px, transparent 4px 6px);
  }
  .name {
    color: var(--text-secondary);
  }
  .val {
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
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
  svg:focus-visible {
    outline: 2px solid var(--focus);
    outline-offset: 4px;
    border-radius: var(--radius-sm);
  }
  .grid line {
    stroke: var(--chart-grid);
    stroke-width: 1;
    shape-rendering: crispEdges;
  }
  .axis {
    stroke: var(--chart-axis);
    shape-rendering: crispEdges;
  }
  .ticks text {
    fill: var(--chart-label);
    font-family: var(--font-mono);
    font-size: 10px;
    font-variant-numeric: tabular-nums;
  }
  .line {
    fill: none;
    stroke-width: 1.75;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .line.dashed {
    stroke-dasharray: 4 3;
    stroke-width: 1.5;
  }
  .line.ghost {
    stroke-width: 1;
    opacity: 0.28;
  }
  .area {
    stroke: none;
  }
  .cursor {
    stroke: var(--chart-cursor);
    stroke-width: 1;
    stroke-dasharray: 2 2;
    shape-rendering: crispEdges;
  }
  .dot {
    stroke: var(--bg-panel);
    stroke-width: 2;
  }
  .marker circle {
    fill: var(--bg-panel);
    stroke-width: 2;
  }
  .marker text {
    fill: var(--text-secondary);
    font-size: 10px;
    font-family: var(--font-mono);
    paint-order: stroke;
    stroke: var(--bg-panel);
    stroke-width: 3px;
  }
  .empty {
    position: absolute;
    inset: 0 0 22px 0;
    display: grid;
    place-items: center;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .tip {
    position: absolute;
    top: 0;
    left: var(--x);
    transform: translateX(12px);
    min-width: 140px;
    padding: 6px 8px;
    border-radius: var(--radius-md);
    background: var(--bg-overlay);
    box-shadow: var(--shadow-md);
    pointer-events: none;
    font-size: var(--type-xs-size);
    line-height: var(--type-xs-line);
    z-index: 2;
    white-space: nowrap;
  }
  .tip.left {
    transform: translateX(calc(-100% - 12px));
  }
  .tip-title {
    color: var(--text-tertiary);
    margin-bottom: 2px;
    font-family: var(--font-mono);
  }
  .tip-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .tip-row .name {
    flex: 1;
  }
  .tip-row .val {
    padding-left: var(--space-3);
  }
</style>
