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
    /** Thicker line drawn above the others (e.g. a mean over many thin per-class lines). */
    emphasis?: boolean;
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
  // When the x domain is fixed (e.g. 0..epochs), prefer a tick step that lands exactly on the end.
  const xTickValues = $derived.by(() => {
    const want = Math.max(2, Math.min(xTicks, Math.floor(width / 70)));
    const [lo, hi] = xExtent;
    if (!xDomain) return niceTicks(lo, hi, want);
    for (let c = want; c >= 2; c--) {
      const t = niceTicks(lo, hi, c);
      if (t[t.length - 1] === hi) return t;
    }
    return niceTicks(lo, hi, want);
  });

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
    const c = cursor as number;
    const rows = keyed
      .map((s) => {
        const pts = s.points;
        const first = pts[0];
        const last = pts[pts.length - 1];
        if (!first || !last) return null;
        // A series only reads out where it has data: never its last value under a cursor past its end.
        const tol = pts.length > 1 ? (last.x - first.x) / (pts.length - 1) / 2 : 0.5;
        if (c < first.x - tol || c > last.x + tol) return null;
        const p = pts[nearest(pts, c)];
        return p ? { s, p } : null;
      })
      .filter((r): r is { s: Series; p: Pt } => r !== null);
    if (!rows.length) return null;
    // Title at the series with the densest sampling (closest to the cursor); rows sampled elsewhere say where.
    const ref = rows.reduce((a, b) => (Math.abs(b.p.x - c) < Math.abs(a.p.x - c) ? b : a));
    const title = xTitle ? xTitle(ref.p.x) : xFormat(ref.p.x);
    const labelled = rows.map((r) => {
      const t = xTitle ? xTitle(r.p.x) : xFormat(r.p.x);
      return { ...r, at: t === title ? null : t };
    });
    return { rows: labelled, px: x(ref.p.x), xv: ref.p.x, title };
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

  // Keyboard scrubbing announces the values under the cursor; pointer hover stays silent.
  let announce = $state(false);
  const spoken = $derived(
    announce && hover
      ? `${xTitle ? xTitle(hover.xv) : xFormat(hover.xv)}: ${hover.rows.map((r) => `${r.s.label} ${fmtValue(r.p.y)}`).join(', ')}`
      : '',
  );

  /** Per-series summary for assistive tech: latest, minimum and maximum. */
  const summary = $derived(
    keyed.map((s) => {
      let lo = Infinity;
      let hi = -Infinity;
      for (const p of s.points) {
        if (!Number.isFinite(p.y)) continue;
        if (p.y < lo) lo = p.y;
        if (p.y > hi) hi = p.y;
      }
      return { s, last: latest(s), lo: Number.isFinite(lo) ? lo : null, hi: Number.isFinite(hi) ? hi : null };
    }),
  );

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
    announce = true;
    cursor = s.points[i]!.x;
  }

  const latest = (s: Series) => {
    for (let i = s.points.length - 1; i >= 0; i--) {
      const p = s.points[i]!;
      if (Number.isFinite(p.y)) return p.y;
    }
    return null;
  };

  let tipWidth = $state(160);
  const tipLeft = $derived(hover ? hover.px + 12 + tipWidth > width - 4 : false);
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

  <p class="sr-only" aria-live="polite">{spoken}</p>
  {#if keyed.length}
    <!-- A table ignores width: 1px, so the visually-hidden box is a wrapper that clips it. -->
    <div class="sr-only"><table>
      <caption>{label}: summary by series. Focus the chart and use the arrow keys to read values point by point.</caption>
      <thead><tr><th scope="col">Series</th><th scope="col">Latest</th><th scope="col">Minimum</th><th scope="col">Maximum</th></tr></thead>
      <tbody>
        {#each summary as row (row.s.id)}
          <tr>
            <th scope="row">{row.s.label}</th>
            <td>{row.last === null ? '—' : fmtValue(row.last)}</td>
            <td>{row.lo === null ? '—' : fmtValue(row.lo)}</td>
            <td>{row.hi === null ? '—' : fmtValue(row.hi)}</td>
          </tr>
        {/each}
      </tbody>
    </table></div>
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
        onpointerdown={() => (announce = false)}
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
            <path class="line" class:dashed={s.dashed} class:ghost={s.ghost} class:emphasis={s.emphasis} {d} style:stroke={s.color} />
          {/each}

          {#each markers as m (m.label + m.x)}
            {#if m.x >= xExtent[0] && m.x <= xExtent[1]}
              <g class="marker" transform="translate({x(m.x)},{y(m.y)})">
                <circle r="4.5" style:stroke={m.color ?? 'var(--chart-1)'} />
                <text
                  y="-10"
                  text-anchor={x(m.x) > width - margin.right - 40 ? 'end' : x(m.x) < margin.left + 40 ? 'start' : 'middle'}
                  dx={x(m.x) > width - margin.right - 40 ? 6 : x(m.x) < margin.left + 40 ? -6 : 0}>{m.label}</text
                >
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
        <div class="tip" class:left={tipLeft} style:--x="{hover.px}px" role="presentation" bind:clientWidth={tipWidth}>
          <p class="tip-title">{hover.title}</p>
          {#each hover.rows as r (r.s.id)}
            <p class="tip-row">
              <span class="swatch" class:dashed={r.s.dashed} style:--c={r.s.color}></span>
              <span class="name">{r.s.label}{#if r.at}<span class="at">{r.at}</span>{/if}</span>
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
  .line.emphasis {
    stroke-width: 2.75;
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
  .tip-row .at {
    margin-left: 6px;
    color: var(--text-tertiary);
    font-family: var(--font-mono);
    font-size: var(--type-micro-size);
  }
  .tip-row .name {
    flex: 1;
  }
  .tip-row .val {
    padding-left: var(--space-3);
  }
</style>
