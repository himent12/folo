<script lang="ts" module>
  import type { Pt } from './downsample';

  export interface Series {
    id: string;
    label: string;
    /** Any CSS colour, usually a token: `var(--chart-1)`. */
    color: string;
    points: Pt[];
    dashed?: boolean;
    /** Line pattern for secondary series of the same hue: dashed, dotted or dash-dot. */
    pattern?: 'dash' | 'dot' | 'dashdot';
    /** Fill the area under the line. */
    area?: boolean;
    /** A 1 px line: a secondary series in ink beside the focal one. */
    thin?: boolean;
    /** Thin, recessive line (e.g. raw values under a smoothed line); hidden from legend and tooltip. */
    ghost?: boolean;
    /**
     * Context line (e.g. each class under their mean): 1 px in `--chart-context`, which holds >= 3:1 on the plot
     * without opacity, so it stays traceable while the emphasised series carry weight and ink. Like a ghost it is
     * hidden from the legend, the readout and end labels; `color` is ignored.
     */
    context?: boolean;
    /** Thicker line drawn above the others (e.g. a mean over many thin per-class lines). */
    emphasis?: boolean;
    /**
     * A band instead of a line: `points` is the upper edge, `lower` the lower edge at the same x positions (e.g. the
     * spread of many curves). Drawn as a soft fill under the lines; not in the legend or the readout.
     */
    lower?: Pt[];
  }

  export interface Marker {
    x: number;
    y: number;
    label: string;
    color?: string;
    /** Ring only: the label is stated elsewhere (e.g. the panel header), not floated over the data. */
    quiet?: boolean;
    /**
     * Bump to pulse the ring once (e.g. a live run just set a new best): a ring grows out of it and fades over the
     * chart duration. Nothing moves under reduced motion. 0 or unset: no pulse.
     */
    pulse?: number;
  }
</script>

<script lang="ts">
  import { tick, untrack, type Snippet } from 'svelte';
  import { scaleLinear, scaleLog } from 'd3-scale';
  import { area as d3area, curveLinear, curveMonotoneX, line as d3line } from 'd3-shape';
  import { downsample, nearest, ticks as niceTicks } from './downsample';
  import { placeEndLabels } from './endLabels';
  import { paintOrder } from './layers';
  import { TouchGesture, type PinAction } from './pin';

  interface Props {
    series: Series[];
    height?: number;
    xDomain?: [number, number];
    yDomain?: [number, number];
    yLog?: boolean;
    /** Y tick values, when the caller fitted the domain itself (./domain): the ticks end exactly on its top. */
    yTickValues?: number[];
    curve?: 'linear' | 'monotone';
    xFormat?: (x: number) => string;
    /** Tick labels on y. Default: as many decimals as the tick step needs ("6", "0.5", "0.25"). */
    yFormat?: (y: number) => string;
    /** Formats values in the tooltip and legend (defaults to yFormat). */
    valueFormat?: (y: number) => string;
    /** Title row for the tooltip, given the hovered x. */
    xTitle?: (x: number) => string;
    markers?: Marker[];
    legend?: boolean;
    yTicks?: number;
    xTicks?: number;
    /** x ticks only on whole numbers (epochs): a 3-epoch run reads 0 1 2 3, not 0 0.5 1 … */
    xInteger?: boolean;
    /** Shared crosshair position in data x (bind it across charts to sync them). */
    cursor?: number | null;
    label: string;
    /** The one line in an empty plot: text, or a snippet (e.g. with a link). The axes stay drawn around it. */
    empty?: string | Snippet;
    /** Label each line where it ends (direct labels), for comparisons where colour alone is not enough. */
    endLabels?: boolean;
    /** Short x-axis caption drawn under the x tick labels, e.g. "epoch". */
    xLabel?: string;
    /** Short y-axis caption drawn above the y tick labels, e.g. "Precision". */
    yLabel?: string;
    /** Leading legend caption saying which value the legend shows, e.g. "Last". Replaced by the x title on hover. */
    legendNote?: string;
    /** A floating tooltip at the cursor. Off by default: the legend is the readout (cursor values on hover). */
    tooltip?: boolean;
    /**
     * Series id to bring forward (e.g. while its legend entry is hovered or pinned): drawn heavier and painted over the
     * rest. The others keep their own contrast-tested colour and weight; nothing is faded.
     */
    highlight?: string | null;
    /** A small dot at each line's last point, so every series stays findable where lines overlap. */
    endDots?: boolean;
  }

  let {
    series,
    height = 200,
    xDomain,
    yDomain,
    yLog = false,
    yTickValues: yTickProp,
    curve = 'linear',
    xFormat: xFormatProp,
    yFormat: yFormatProp,
    valueFormat,
    xTitle,
    markers = [],
    legend = true,
    yTicks = 4,
    xTicks = 6,
    xInteger = false,
    cursor = $bindable(null),
    label,
    empty = 'No data yet',
    endLabels = false,
    xLabel,
    yLabel,
    legendNote,
    tooltip = false,
    highlight = null,
    endDots = false,
  }: Props = $props();
  // A pulse the chart was mounted with is old news (a tab switch, a return to a finished run): only a newer one plays.
  const pulseBase = untrack(() => Math.max(0, ...markers.map((m) => m.pulse ?? 0)));

  const uid = $props.id();
  let width = $state(0);

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
    if (yTickProp) return yTickProp.filter((t) => t >= yExtent[0] - 1e-9 && t <= yExtent[1] + 1e-9);
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

  // Decimals follow the tick step, so a 0..6 axis reads "0 2 4 6" and a 0..1 axis "0 0.5 1"; log ticks use 3 digits.
  const yFormat = $derived.by(() => {
    if (yFormatProp) return yFormatProp;
    if (yLog) return (v: number) => String(+v.toPrecision(3));
    const step = yTickValues.length > 1 ? Math.abs(yTickValues[1]! - yTickValues[0]!) : 1;
    const d = Math.max(0, Math.min(6, Math.ceil(-Math.log10(step) - 1e-9)));
    return (v: number) => v.toFixed(d);
  });
  const fmtValue = $derived(valueFormat ?? yFormatProp ?? ((v: number) => String(+v.toFixed(3))));

  const margin = $derived.by(() => {
    const longest = Math.max(...yTickValues.map((v) => yFormat(v).length), 1);
    // ~7 px per character at 11 px Geist (tabular digits); end labels are 12 px.
    const labelW = endLabels && width >= 560 ? Math.min(130, Math.max(0, ...series.map((s) => Math.min(18, s.label.length))) * 6.8 + 10) : 0;
    const left = Math.ceil(longest * 7) + 14;
    // An x-axis title gets its own line under the tick labels.
    return { top: yLabel ? 24 : 8, right: 12 + labelW, bottom: xLabel ? 36 : 22, left };
  });

  const x = $derived(scaleLinear().domain(xExtent).range([margin.left, Math.max(margin.left + 1, width - margin.right)]));
  const y = $derived(
    (yLog ? scaleLog().domain(yExtent) : scaleLinear().domain(yExtent)).range([height - margin.bottom, margin.top]),
  );
  // When the x domain is fixed (e.g. 0..epochs), prefer a tick step that lands exactly on the end.
  // Default x labels: integers, or one decimal when the domain is too short for distinct integers.
  const xFormat = $derived(
    xFormatProp ?? ((v: number) => (xExtent[1] - xExtent[0] < 5 ? String(+v.toFixed(1)) : String(Math.round(v)))),
  );

  const xTickValues = $derived.by(() => {
    const want = Math.max(2, Math.min(xTicks, Math.floor(width / 70)));
    const [lo, hi] = xExtent;
    if (xInteger && hi - lo <= want) {
      const out: number[] = [];
      for (let v = Math.ceil(lo); v <= Math.floor(hi); v++) out.push(v);
      if (out.length >= 2) return out;
    }
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
      if (s.lower) {
        const lo = s.lower;
        const zipped = s.points.map((p, i) => ({ x: p.x, y: p.y, y0: lo[i]?.y ?? p.y }));
        const band = d3area<{ x: number; y: number; y0: number }>()
          .defined((p) => Number.isFinite(p.y) && Number.isFinite(p.y0))
          .x((p) => x(p.x))
          .y0((p) => y(p.y0))
          .y1((p) => y(p.y))
          .curve(c);
        return { s, d: '', a: '', b: band(zipped) ?? '' };
      }
      const pts = downsample(s.points, buckets);
      return { s, d: ln(pts) ?? '', a: s.area ? (ar(pts) ?? '') : '', b: '' };
    });
  });

  /** Direct labels at each line's last point, each in a spot that covers no curve and no other label (endLabels.ts). */
  // Narrow plots keep their width for data: the legend names the lines instead.
  const showEnds = $derived(endLabels && width >= 560);
  const ends = $derived.by(() => {
    if (!showEnds) return [];
    const drawn = visible.filter((s) => !s.ghost && !s.context);
    const color = new Map(drawn.map((s) => [s.id, s.color]));
    const points = drawn
      .map((s) => {
        const p = s.points[s.points.length - 1];
        const curve = visible.indexOf(s);
        return p && Number.isFinite(p.y) ? { id: s.id, label: s.label.length > 18 ? `${s.label.slice(0, 17)}…` : s.label, x: x(p.x), y: y(p.y), curve } : null;
      })
      .filter((e): e is NonNullable<typeof e> => e !== null)
      // The highlighted line is labelled first, so it never loses its spot; then the shortest lines, which are the
      // easiest to lose among the others.
      .sort((a, b) => Number(b.id === highlight) - Number(a.id === highlight) || a.x - b.x);
    const curves = visible.map((s) => s.points.filter((p) => Number.isFinite(p.y)).map((p) => ({ x: x(p.x), y: y(p.y) })));
    const bounds = { x0: margin.left, y0: margin.top, x1: width - 2, y1: height - margin.bottom };
    return placeEndLabels(points, curves, bounds).map((e) => ({ ...e, color: color.get(e.id) ?? 'var(--text-secondary)' }));
  });

  // ---- hover / keyboard cursor ----
  const keyed = $derived(visible.filter((s) => !s.ghost && !s.context && !s.lower));
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

  const local = (e: PointerEvent) => {
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
    return { px: e.clientX - rect.left, py: e.clientY - rect.top };
  };
  const touchy = (e: PointerEvent) => e.pointerType === 'touch' || e.pointerType === 'pen';
  const inPlot = (px: number) => px >= margin.left - 4 && px <= width - margin.right + 4;

  // Touch: a tap or a sideways scrub pins the cursor and its readout until it is tapped again, a tap lands outside
  // the chart or Escape is pressed (./pin). A mouse reads out on hover, as before.
  let pinned = $state(false);
  let gesture: TouchGesture | null = null;
  let svgEl = $state<SVGSVGElement>();
  function apply(a: PinAction) {
    if (a.kind === 'clear') {
      cursor = null;
      pinned = false;
    } else if (a.kind === 'set' && inPlot(a.px)) {
      cursor = x.invert(Math.min(width - margin.right, Math.max(margin.left, a.px)));
      pinned = true;
    }
  }
  // A cursor cleared elsewhere (a synced chart, blur) is no longer pinned here.
  $effect(() => {
    if (cursor === null) pinned = false;
  });
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
    spoken = '';
    if (!touchy(e)) return;
    const { px, py } = local(e);
    gesture = new TouchGesture(px, py, pinned && hover ? hover.px : null);
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
    spoken = '';
    const { px, py } = local(e);
    if (touchy(e)) {
      if (gesture) apply(gesture.move(px, py));
      return;
    }
    pinned = false;
    cursor = inPlot(px) ? x.invert(px) : null;
  }
  function onpointerleave(e: PointerEvent) {
    if (!touchy(e) && !pinned) cursor = null;
  }

  // Narrow plots: the legend keeps one line of whole items (no leading note), and the cursor's x reads out on the axis.
  const compact = $derived(width > 0 && width < 480);
  const pill = $derived.by(() => {
    if (!compact || !hover) return null;
    const w = Math.ceil(hover.title.length * 6.8) + 12;
    const cx = Math.min(width - w / 2 - 1, Math.max(w / 2 + 1, hover.px));
    return { w, cx, text: hover.title };
  });

  // Keyboard scrubbing announces the values under the cursor. The text is written from the key handler itself, so
  // pointer movement here or on a linked chart never produces announcements.
  let spoken = $state('');

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
      pinned = false;
      spoken = '';
      return;
    } else return;
    e.preventDefault();
    cursor = s.points[i]!.x;
    // `hover` derives from the cursor; read it once the update has settled.
    void tick().then(() => {
      const h = hover;
      spoken = h ? `${h.title}: ${h.rows.map((r) => `${r.s.label} ${fmtValue(r.p.y)}`).join(', ')}` : '';
    });
  }

  // Dots are 0-length dashes drawn with round caps (stroke width), so they read as dots, not a faint hairline.
  // Dots sit 4.5 px apart, so a line they run along (box under dfl in the loss chart) still shows between them.
  const PATTERNS = { dash: [6, 3], dot: [0.01, 4.5], dashdot: [7, 3, 0.01, 3] } as const;
  // The legend swatch (12 px, 2 px stroke, round caps) shows the same rhythm as its line: two dashes, three dots.
  const SWATCH = { dash: [3, 4], dot: [0.01, 4.5], dashdot: [4, 3.5, 0.01, 3.5] } as const;
  const patternOf = (s: Series) => s.pattern ?? (s.dashed ? 'dash' : null);
  /** stroke-dasharray for a series; `k` scales it with a heavier stroke (a highlighted line keeps its rhythm). */
  function dashOf(s: Series, k = 1): string | undefined {
    const p = patternOf(s);
    return p ? PATTERNS[p].map((v) => +(v * k).toFixed(3)).join(' ') : undefined;
  }
  const swatchDash = (s: Series) => {
    const p = patternOf(s);
    return p ? SWATCH[p].join(' ') : undefined;
  };
  /** Stroke width of a series' line in px, as the `.line` classes below draw it. */
  function widthOf(s: Series): number {
    if (s.ghost || s.context || s.thin) return 1;
    const p = patternOf(s);
    if (s.emphasis || (highlight !== null && s.id === highlight)) return p === 'dot' ? 3.25 : 2.75;
    return p === 'dot' ? 2.25 : p ? 1.5 : 1.75;
  }
  /** `items` with the highlighted series moved to the end, so it is painted over every other line. */
  function lifted<T>(items: T[], id: (t: T) => string): T[] {
    if (highlight === null) return items;
    return [...items.filter((t) => id(t) !== highlight), ...items.filter((t) => id(t) === highlight)];
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

{#snippet swatch(s: Series)}
  <svg class="swatch" width="14" height="6" viewBox="0 0 14 6" aria-hidden="true">
    <line x1="1" x2="13" y1="3" y2="3" style:stroke={s.color} style:stroke-dasharray={swatchDash(s)} />
  </svg>
{/snippet}

<div class="chart">
  {#if legend && keyed.length > 0}
    <ul class="legend" aria-hidden="true">
      {#if (legendNote || hover) && !compact}<li class="note" class:at={!!hover}>{hover ? hover.title : legendNote}</li>{/if}
      {#each keyed as s (s.id)}
        {@const v = hover ? hover.rows.find((r) => r.s.id === s.id)?.p.y : latest(s)}
        <li>
          {@render swatch(s)}
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
        bind:this={svgEl}
        {width}
        {height}
        role="img"
        aria-label={label}
        tabindex={hasData ? 0 : -1}
        {onpointermove}
        {onpointerleave}
        {onpointerdown}
        {onpointerup}
        {onpointercancel}
        {onkeydown}
        onblur={() => ((cursor = null), (spoken = ''))}
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
            <text x={x(t)} y={height - margin.bottom + 16} text-anchor="middle">{xFormat(t)}</text>
          {/each}
          {#if xLabel}<text class="x-label" x={(margin.left + width - margin.right) / 2} y={height - 4} text-anchor="middle">{xLabel}</text>{/if}
          {#if yLabel}<text class="x-label" x="0" y="10" dy="0.32em" text-anchor="start">{yLabel}</text>{/if}
        </g>

        {#if hasData}
          <defs>
            {#each paths as { s, a } (s.id)}
              {#if a}
                <linearGradient id="{uid}-{s.id}" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0" style:stop-color={s.color} stop-opacity="0.08" />
                  <stop offset="1" style:stop-color={s.color} stop-opacity="0" />
                </linearGradient>
              {/if}
            {/each}
          </defs>
          {#each paths.filter((p) => p.b) as { s, b } (s.id)}
            <path class="band" d={b} style:fill={s.color} />
          {/each}
          {#each lifted(paintOrder(paths.filter((p) => !p.b), (p) => p.s), (p) => p.s.id) as { s, d, a } (s.id)}
            {#if a}<path class="area" d={a} fill="url(#{uid}-{s.id})" />{/if}
            {#if patternOf(s) && !s.ghost && !s.context}
              <!-- Patterned lines are painted over solid ones (./layers). A 1.5 px halo in the panel colour around each
                   dash or dot keeps the pattern readable where it runs along a solid line, which still shows between
                   them. -->
              <path class="halo" {d} style:stroke-width="{widthOf(s) + 3}px" style:stroke-dasharray={dashOf(s, s.id === highlight ? 1.4 : 1)} />
            {/if}
            <path
              class="line"
              class:dashed={s.dashed || s.pattern === 'dash' || s.pattern === 'dashdot'}
              class:dotted={s.pattern === 'dot'}
              class:thin={s.thin}
              class:ghost={s.ghost}
              class:context={s.context}
              class:emphasis={s.emphasis || (highlight !== null && s.id === highlight)}
              {d}
              style:stroke={s.context ? undefined : s.color}
              style:stroke-dasharray={dashOf(s, s.id === highlight ? 1.4 : 1)}
            />
          {/each}
          {#if endDots}
            {#each lifted(paintOrder(keyed, (s) => s), (s) => s.id) as s (s.id)}
              {@const p = s.points[s.points.length - 1]}
              {#if p && Number.isFinite(p.y)}
                <circle class="end-dot" cx={x(p.x)} cy={y(p.y)} r="3" style:fill={s.color} />
              {/if}
            {/each}
          {/if}

          {#if showEnds}
            {#each ends as e (e.id)}
              <text class="end-label" x={e.x} y={e.y} dy="0.32em" style:fill={e.color}>{e.label}</text>
            {/each}
          {/if}
          {#each markers as m (m.label + m.x)}
            {#if m.x >= xExtent[0] && m.x <= xExtent[1]}
              <g class="marker" transform="translate({x(m.x)},{y(m.y)})">
                {#if m.pulse && m.pulse > pulseBase}
                  {#key m.pulse}<circle class="pulse" r="4.5" style:stroke={m.color ?? 'var(--chart-1)'} />{/key}
                {/if}
                <circle class="ring" r="4.5" style:stroke={m.color ?? 'var(--chart-1)'} />
                {#if !m.quiet}
                  <text
                    y="-10"
                    text-anchor={x(m.x) > width - margin.right - 40 ? 'end' : x(m.x) < margin.left + 40 ? 'start' : 'middle'}
                    dx={x(m.x) > width - margin.right - 40 ? 6 : x(m.x) < margin.left + 40 ? -6 : 0}>{m.label}</text
                  >
                {/if}
              </g>
            {/if}
          {/each}

          {#if hover}
            <line class="cursor" class:pinned x1={hover.px} x2={hover.px} y1={margin.top} y2={height - margin.bottom} />
            {#each hover.rows as r (r.s.id)}
              {#if Number.isFinite(r.p.y) && (!yLog || r.p.y > 0)}
                <circle class="dot" cx={x(r.p.x)} cy={y(r.p.y)} r="3.5" style:fill={r.s.color} />
              {/if}
            {/each}
            {#if pill}
              <!-- Over the x ticks it covers: the cursor's x, which a wide legend states in its note. -->
              <g class="pill" transform="translate({pill.cx},{height - margin.bottom + 4})">
                <rect x={-pill.w / 2} y="0" width={pill.w} height="16" rx="3" />
                <text y="8" dy="0.32em" text-anchor="middle">{pill.text}</text>
              </g>
            {/if}
          {/if}
        {/if}
      </svg>

      {#if !hasData}
        <p class="empty"><span>{#if typeof empty === 'string'}{empty}{:else}{@render empty()}{/if}</span></p>
      {/if}

      {#if hover && tooltip}
        <div class="tip" class:left={tipLeft} style:--x="{hover.px}px" role="presentation" bind:clientWidth={tipWidth}>
          <p class="tip-title">{hover.title}</p>
          {#each hover.rows as r (r.s.id)}
            <p class="tip-row">
              {@render swatch(r.s)}
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
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  /* Items wrap whole: a value never ends up on a line of its own. */
  .legend li {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .legend .note {
    /* Fixed width so the readout does not shift the legend as the cursor moves (phones wrap instead). */
    min-width: min(6.5em, 100%);
    color: var(--text-tertiary);
    font-variant-numeric: tabular-nums;
  }
  .legend .note.at {
    color: var(--text-secondary);
  }
  /* Axis names are labels, not ticks: 12 px. */
  .ticks .x-label {
    fill: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .swatch {
    flex-shrink: 0;
    overflow: visible;
  }
  .swatch line {
    stroke-width: 2;
    stroke-linecap: round;
  }
  .name {
    color: var(--text-secondary);
  }
  .val {
    color: var(--text-primary);
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
    font-size: var(--type-xs-size);
    font-variant-numeric: tabular-nums;
  }
  .line {
    fill: none;
    stroke-width: 1.75;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .line.dashed {
    stroke-width: 1.5;
  }
  /* Dots carry less ink than a line: a heavier stroke keeps a dotted series as findable as a solid one. */
  .line.dotted {
    stroke-width: 2.25;
  }
  .line.thin {
    stroke-width: 1;
  }
  .line.emphasis {
    stroke-width: 2.75;
  }
  .line.dotted.emphasis {
    stroke-width: 3.25;
  }
  .line {
    transition: stroke-width var(--dur-fast) var(--ease-standard);
  }
  .end-dot {
    stroke: var(--bg-panel);
    stroke-width: 1.5;
  }
  .halo {
    fill: none;
    stroke: var(--bg-panel);
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .line.ghost {
    stroke-width: 1;
    opacity: 0.28;
  }
  .line.context {
    stroke: var(--chart-context);
    stroke-width: 1;
  }
  .band {
    stroke: none;
    opacity: 0.16;
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
  /* A pinned (touch) cursor is solid: it stays until tapped again, a tap lands outside or Escape. */
  .cursor.pinned {
    stroke-dasharray: none;
  }
  .dot {
    stroke: var(--bg-panel);
    stroke-width: 2;
  }
  .pill rect {
    fill: var(--bg-overlay);
    stroke: var(--line-default);
  }
  .pill text {
    fill: var(--text-primary);
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
  }
  .end-label {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    paint-order: stroke;
    stroke: var(--bg-panel);
    stroke-width: 3px;
  }
  .marker .ring {
    fill: var(--bg-panel);
    stroke-width: 2;
  }
  /* A new best: a ring grows out of the marker and fades, once, over the chart duration. */
  .marker .pulse {
    fill: none;
    stroke-width: 2;
    opacity: 0;
    transform-box: fill-box;
    transform-origin: center;
    animation: best-pulse var(--dur-chart) var(--ease-out);
  }
  @keyframes best-pulse {
    from {
      opacity: 0.9;
      transform: scale(1);
    }
    to {
      opacity: 0;
      transform: scale(2.4);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .marker .pulse {
      display: none;
    }
  }
  .marker text {
    fill: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
    paint-order: stroke;
    stroke: var(--bg-panel);
    stroke-width: 3px;
  }
  .empty {
    position: absolute;
    /* Clear of the y labels, so a message that wraps on a phone never runs into them. */
    inset: 0 var(--space-4) 24px 40px;
    display: grid;
    place-items: center;
    text-align: center;
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
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    z-index: 2;
    white-space: nowrap;
  }
  .tip.left {
    transform: translateX(calc(-100% - 12px));
  }
  .tip-title {
    color: var(--text-tertiary);
    margin-bottom: 2px;
    font-variant-numeric: tabular-nums;
  }
  .tip-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .tip-row .at {
    margin-left: 6px;
    color: var(--text-tertiary);
    font-variant-numeric: tabular-nums;
    font-size: var(--type-sm-size);
  }
  .tip-row .name {
    flex: 1;
  }
  .tip-row .val {
    padding-left: var(--space-3);
  }
</style>
