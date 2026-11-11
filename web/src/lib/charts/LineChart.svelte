<script lang="ts" module>
  import type { Pt } from './downsample';
  import type { Marker, Series, XMark } from './lineSeries';

  export type { Marker, Series, XMark };
</script>

<script lang="ts">
  import { tick, untrack, type Snippet } from 'svelte';
  import { scaleLinear, scaleLog } from 'd3-scale';
  import { nearest } from './downsample';
  import { logMinors, MAX_TICKS } from './domain';
  import { END_LABELS_MIN, memoPlaceEndLabels } from './endLabels';
  import { prefs } from '$lib/prefs.svelte';
  import { chartHint, claimOnce, TAP_HINT_KEY } from './hint';
  import { paintOrder } from './layers';
  import { TouchGesture, type PinAction } from './pin';
  import { MediaQuery } from 'svelte/reactivity';
  import { Tween } from 'svelte/motion';
  import { cubicOut } from 'svelte/easing';
  import { motion } from '$lib/ui/motion';
  import { media } from '$lib/design/tokens';
  import { patternOf, dashOf, inkOf, seriesSummary, strokeWidth } from './lineSeries';
  import { capInPlotOf, marginOf, markOf, xExtentOf, xTicksOf, yExtentOf, yFormatOf, yTicksOf } from './lineAxes';
  import { mergesOf, pathsOf } from './linePaths';
  import { endRadius, endsOf, placedEndsOf } from './lineEndLabels';
  import { hoverOf, pickNearest, pillOf } from './lineHover';
  import { textMeasurer } from './textWidth';
  import ChartLegend from './ChartLegend.svelte';
  import ChartTooltip from './ChartTooltip.svelte';
  import LineEnds from './LineEnds.svelte';

  interface Props {
    series: Series[];
    height?: number;
    xDomain?: [number, number];
    yDomain?: [number, number];
    yLog?: boolean;
    /** Y tick values, when the caller fitted the domain itself (./domain): the ticks end exactly on its top. */
    yTickValues?: number[];
    /** Decimals of every y label, when the caller fitted the domain (`Fit.decimals`): a short plot that shows every
     *  second tick keeps the step's decimals, so one domain reads the same at every width. */
    yDecimals?: number;
    curve?: 'linear' | 'monotone';
    xFormat?: (x: number) => string;
    /** Tick labels on y. Default: as many decimals as the tick step needs ("6", "0.5", "0.25"). */
    yFormat?: (y: number) => string;
    /** Formats values in the tooltip and legend (defaults to yFormat). */
    valueFormat?: (y: number) => string;
    /** Title row for the tooltip, given the hovered x. */
    xTitle?: (x: number) => string;
    markers?: Marker[];
    /** A phase change on x (XMark). The plot keeps room for its label whenever it is set, in or out of the domain. */
    xMark?: XMark;
    legend?: boolean;
    /** The most y ticks (3–5 are drawn; never more than five gridlines). */
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
    /** Leading legend caption saying which value the legend shows, e.g. "Latest · ep 37". Replaced by the x title on hover. */
    legendNote?: string;
    /** A floating tooltip at the cursor. Off by default: the legend is the readout (cursor values on hover). */
    tooltip?: boolean;
    /**
     * Series id to bring forward (e.g. while its legend entry is hovered or pinned): drawn heavier and painted over the
     * rest. The others keep their own contrast-tested colour and weight; nothing is faded.
     */
    highlight?: string | null;
    /**
     * While a series is highlighted, draw it at 3 px in its hue over a panel-colour halo, painted last, and the others
     * as context curves: 1 px in `--chart-context` (>= 3:1 on the plot, no opacity), keeping their line pattern, with
     * their end dots and labels in ink. The comparison's "keep this line in front"; its legend keeps every hue.
     */
    recede?: boolean;
    /** A small dot at each line's last point, so every series stays findable where lines overlap. */
    endDots?: boolean;
    /**
     * The reading hint ("← → to inspect" on keyboard focus, "Tap to pin a value" once per session on touch), or null.
     * The legend shows it; a chart without one binds this to show it in its own readout row.
     */
    hint?: string | null;
    /**
     * Series the pointer can pick out where they are drawn (PR curves' quiet context lines): `picked` is the one whose
     * line passes nearest the pointer, within PICK_PX (PICK_TOUCH_PX for a tap), and a click on the plot calls `onpick`
     * with it. A tap pins the cursor and picks.
     */
    pick?: (s: Series) => boolean;
    picked?: string | null;
    onpick?: (id: string) => void;
  }

  let {
    series,
    height = 200,
    xDomain,
    yDomain,
    yLog = false,
    yTickValues: yTickProp,
    yDecimals,
    curve = 'linear',
    xFormat: xFormatProp,
    yFormat: yFormatProp,
    valueFormat,
    xTitle,
    markers = [],
    xMark,
    legend = true,
    yTicks = MAX_TICKS,
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
    recede = false,
    endDots = false,
    hint = $bindable(null),
    pick,
    picked = $bindable(null),
    onpick,
  }: Props = $props();
  // A pulse the chart was mounted with is old news (a tab switch, a return to a finished run): only a newer one plays.
  const pulseBase = untrack(() => Math.max(0, ...markers.map((m) => m.pulse ?? 0)));
  const focusBase = untrack(() => Math.max(0, ...markers.map((m) => m.focus ?? 0)));

  const uid = $props.id();
  let width = $state(0);

  const visible = $derived(series.filter((s) => s.points.length > 0));
  const hasData = $derived(visible.length > 0);

  const xExtent = $derived(xExtentOf(xDomain, visible));
  const yExtent = $derived(yExtentOf(yDomain, visible, yLog));
  const yTickValues = $derived(yTicksOf(yTickProp, yExtent, yLog, yTicks));
  // A log axis labels its decades and ends; the 2s and 5s between them stay as unlabelled minor gridlines, so the
  // scale reads evenly (0.2 · 0.5 · 1 · 2 · 5 · 10) without crowding the labels.
  const yMinor = $derived(yLog && !yTickProp ? logMinors(yExtent[0], yExtent[1]) : []);
  const yFormat = $derived(yFormatOf(yFormatProp, yLog, yTickValues, yDecimals));
  const fmtValue = $derived(valueFormat ?? yFormatProp ?? ((v: number) => String(+v.toFixed(3))));
  const textWidth = textMeasurer();
  /** Direct labels at each line's last point, each in a spot that covers no curve and no other label (endLabels.ts). */
  // One rule for every chart (DESIGN_DIRECTION §2.4): a plot under 560 px (every phone) has no direct labels, short
  // names included, and keeps that width for data; its legend carries each line's name, value and how it ended.
  const showEnds = $derived(endLabels && width >= END_LABELS_MIN);
  const margin = $derived.by(() => marginOf(yTickValues, yFormat, xExtent, series, textWidth, showEnds, yLabel, capInPlot, xMark, xLabel));
  const capInPlot = $derived(capInPlotOf(yLabel, xMark, yLog, yTickValues, yExtent, height, xLabel));

  const x = $derived(scaleLinear().domain(xExtent).range([margin.left, Math.max(margin.left + 1, width - margin.right)]));
  // A y domain change eases over 240 ms (callers re-fit at most once per epoch, ./domain holdDomain; values themselves
  // never tween): the lines and gridlines glide to the new scale instead of jumping. Instant under reduced motion and
  // on a log scale.
  const yEase = new Tween<[number, number]>(untrack(() => yExtent), { easing: cubicOut });
  $effect(() => {
    const d = yExtent;
    const log = yLog;
    untrack(() => void yEase.set(d, { duration: log ? 0 : motion(240) }));
  });
  const yShown = $derived(yLog ? yExtent : yEase.current);
  const yScale = (d: [number, number]) => (yLog ? scaleLog().domain(d) : scaleLinear().domain(d)).range([height - margin.bottom, margin.top]);
  const y = $derived(yScale(yShown));
  /** The y scale the axis is easing to (`y` once the ease ends): end labels are placed against it, once per change. */
  const yTo = $derived(yScale(yExtent));
  // Default x labels: integers, or one decimal when the domain is too short for distinct integers.
  const xFormat = $derived(
    xFormatProp ?? ((v: number) => (xExtent[1] - xExtent[0] < 5 ? String(+v.toFixed(1)) : String(Math.round(v)))),
  );

  const xTickValues = $derived(xTicksOf(width, xTicks, xExtent, xInteger, xDomain));
  const paths = $derived(pathsOf(visible, curve, width, margin, height, x, y, yLog));
  const endR = (id: string) => endRadius(id, endDots, receding, highlight, visible, merges);
  const placeEnds = memoPlaceEndLabels();
  const placedEnds = $derived(placedEndsOf(showEnds, visible, x, yTo, textWidth, highlight, margin, width, height, placeEnds));
  const ends = $derived.by(() => endsOf(visible, placedEnds, y, receding, highlight, endR));
  const mark = $derived(markOf(xMark, xExtent, x, width, margin));

  // ---- hover / keyboard cursor ----
  const keyed = $derived(visible.filter((s) => !s.ghost && !s.context && !s.lower));
  const longestSeries = $derived(keyed.reduce<Series | undefined>((m, s) => (!m || s.points.length > m.points.length ? s : m), undefined));

  const hover = $derived(hoverOf(cursor, hasData, xExtent, keyed, xTitle, xFormat, x));

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
      tapHint = false;
    }
  }

  // Reading hints in the readout row (./hint): "← → to inspect" while the plot has keyboard focus, and on a touch
  // screen "Tap to pin a value" on the first chart of the session, until it is tapped. A mouse gets neither, and the
  // phone tier and touch-only devices get no key hint (no arrow keys to press there).
  let keyFocus = $state(false);
  const keyless = new MediaQuery(`${media.phone}, (hover: none) and (pointer: coarse)`, false);
  let tapHint = $state(typeof matchMedia !== 'undefined' && matchMedia('(pointer: coarse)').matches && claimOnce(TAP_HINT_KEY));
  const hintText = $derived(hasData ? chartHint({ reading: hover !== null, keyFocus: keyFocus && !keyless.current, tap: tapHint }) : null);
  $effect(() => {
    hint = hintText;
  });
  function onfocus() {
    try {
      keyFocus = !!svgEl?.matches(':focus-visible');
    } catch {
      keyFocus = false;
    }
  }
  // A cursor cleared elsewhere (a synced chart, blur) is no longer pinned here, and picks nothing.
  $effect(() => {
    if (cursor === null) {
      pinned = false;
      picked = null;
    }
  });

  /** How near a line the pointer picks it out (`pick`), px: a mouse, and a finger. */
  const PICK_PX = 10;
  const PICK_TOUCH_PX = 20;
  function pickAt(px: number, py: number, r: number): string | null {
    return pickNearest(pick, inPlot, visible, x, y, px, py, r);
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
    spoken = '';
    if (!touchy(e)) return;
    const { px, py } = local(e);
    gesture = new TouchGesture(px, py, pinned && hover ? hover.px : null);
  }
  function onpointerup(e: PointerEvent) {
    if (!touchy(e)) {
      // A click on a picked line (`pick`) acts on it.
      if (picked !== null && e.button === 0) onpick?.(picked);
      return;
    }
    if (!gesture) return;
    const { px, py } = local(e);
    apply(gesture.up(px));
    gesture = null;
    picked = pinned ? pickAt(px, py, PICK_TOUCH_PX) : null;
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
    if (pick) picked = pickAt(px, py, PICK_PX);
  }
  function onpointerleave(e: PointerEvent) {
    if (!touchy(e) && !pinned) {
      cursor = null;
      picked = null;
    }
  }

  // Narrow plots: the cursor's x reads out on the axis, not in the legend's note.
  const compact = $derived(width > 0 && width < 480);
  const pill = $derived(pillOf(compact, hover, width));

  // Keyboard scrubbing announces the values under the cursor. The text is written from the key handler itself, so
  // pointer movement here or on a linked chart never produces announcements.
  let spoken = $state('');

  const summary = $derived(seriesSummary(keyed));

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

  const light = $derived(prefs.resolvedTheme === 'light');
  const widthOf = (s: Series): number => strokeWidth(s, receding, highlight, light);
  /** A highlighted series with `recede`: it alone keeps weight and hue. */
  const receding = $derived(recede && highlight !== null && visible.some((s) => s.id === highlight));
  /** Drawn as a context curve: a series of that kind, or a run receded behind the highlighted one. */
  const contextual = (s: Series) => !!s.context || (receding && s.id !== highlight && !s.ghost);
  /** `items` with the highlighted series moved to the end, so it is painted over every other line. */
  function lifted<T>(items: T[], id: (t: T) => string): T[] {
    if (highlight === null) return items;
    return [...items.filter((t) => id(t) !== highlight), ...items.filter((t) => id(t) === highlight)];
  }

  const merges = $derived(mergesOf(endDots, hasData, markers, keyed, x, y, xExtent));
  /** The marker drawn inside series `id`'s end mark, if any. */
  const mergedInto = (id: string): Marker | null => {
    const i = merges.indexOf(id);
    return i < 0 ? null : (markers[i] ?? null);
  };

  let tipWidth = $state(160);
  const tipLeft = $derived(hover ? hover.px + 12 + tipWidth > width - 4 : false);
</script>

<div class="chart">
  {#if legend && keyed.length > 0}
    <ChartLegend {hintText} {hover} {compact} {legendNote} {keyed} {fmtValue} />
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
        class:picking={picked !== null && !!onpick}
        {width}
        {height}
        role="group"
        aria-roledescription="chart"
        aria-label={label}
        tabindex={hasData ? 0 : -1}
        {onpointermove}
        {onpointerleave}
        {onpointerdown}
        {onpointerup}
        {onpointercancel}
        {onkeydown}
        {onfocus}
        onblur={() => ((cursor = null), (spoken = ''), (keyFocus = false))}
      >
        <g class="grid">
          {#each yMinor as t (t)}
            <line class="minor" x1={margin.left} x2={width - margin.right} y1={y(t)} y2={y(t)} />
          {/each}
          {#each yTickValues as t (t)}
            <line x1={margin.left} x2={width - margin.right} y1={y(t)} y2={y(t)} />
          {/each}
        </g>
        <line class="axis" x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} />
        {#if mark}
          <!-- The span after the mark hatched along the axis (the progress bar's hatch), the line, its label above the plot. -->
          <defs>
            <pattern id="{uid}-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line class="hatch" x1="0" y1="0" x2="0" y2="4" />
            </pattern>
          </defs>
          <rect x={mark.px} y={height - margin.bottom - 6} width={Math.max(0, mark.end - mark.px)} height="6" fill="url(#{uid}-hatch)" />
          <line class="mark-line" x1={mark.px} x2={mark.px} y1={margin.top - 4} y2={height - margin.bottom} />
          <text class="mark-label" x={mark.tx} y={margin.top - 8} text-anchor={mark.anchor}>{mark.label}</text>
        {/if}
        <g class="ticks">
          {#each yTickValues as t (t)}
            <text x={margin.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{yFormat(t)}</text>
          {/each}
          {#each xTickValues as t (t)}
            <text x={x(t)} y={height - margin.bottom + 16} text-anchor="middle">{xFormat(t)}</text>
          {/each}
          {#if xLabel}<text class="x-label" x={(margin.left + width - margin.right) / 2} y={height - 4} text-anchor="middle">{xLabel}</text>{/if}
          {#if yLabel}<text class="x-label" x="0" y={capInPlot ? y(Math.max(...yTickValues)) - 14 : 10} dy="0.32em" text-anchor="start">{yLabel}</text>{/if}
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
            <!-- A fitted domain that does not start at 0 (Compare's "Fit") cuts a line's early climb at the baseline. -->
            <clipPath id="{uid}-plot"><rect x="0" y="0" {width} height={Math.max(0, height - margin.bottom + 1)} /></clipPath>
          </defs>
          <g clip-path="url(#{uid}-plot)">
          {#each paths.filter((p) => p.b) as { s, b } (s.id)}
            <path class="band" d={b} style:fill={s.color} />
          {/each}
          {#each lifted(paintOrder(paths.filter((p) => !p.b), (p) => p.s), (p) => p.s.id) as { s, d, a } (s.id)}
            {#if a}<path class="area" d={a} fill="url(#{uid}-{s.id})" />{/if}
            {#if (patternOf(s) || (receding && s.id === highlight)) && !s.ghost && !contextual(s)}
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
              class:context={contextual(s)}
              class:emphasis={s.emphasis || (highlight !== null && s.id === highlight)}
              {d}
              style:stroke={contextual(s) ? undefined : inkOf(s)}
              style:stroke-width={receding && !s.ghost && !s.context ? `${widthOf(s)}px` : undefined}
              style:stroke-dasharray={dashOf(s, s.id === highlight ? 1.4 : 1)}
            />
          {/each}
          </g>
          <LineEnds
            {endDots}
            {keyed}
            {receding}
            {highlight}
            {contextual}
            {mergedInto}
            {lifted}
            {x}
            {y}
            {pulseBase}
            {focusBase}
            {showEnds}
            {ends}
            {markers}
            {merges}
            {xExtent}
            {width}
            {margin}
          />

          {#if hover}
            <line class="cursor" class:pinned x1={hover.px} x2={hover.px} y1={margin.top} y2={height - margin.bottom} />
            {#each hover.rows as r (r.s.id)}
              <!-- None for a value under a fitted floor: the line is cut there, and the legend reads it. -->
              {#if Number.isFinite(r.p.y) && (!yLog || r.p.y > 0) && r.p.y >= yShown[0] - 1e-9}
                <circle class="dot" cx={x(r.p.x)} cy={y(r.p.y)} r="3.5" style:fill={contextual(r.s) ? 'var(--chart-context)' : r.s.color} />
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
          <ChartTooltip {hover} {tipLeft} bind:tipWidth {fmtValue} />
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
  /* Axis names are labels, not ticks: 12 px. */
  .ticks .x-label {
    fill: var(--text-tertiary);
    font-size: var(--type-sm-size);
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
  /* Over a line a click acts on (`pick`, `onpick`). */
  svg.picking {
    cursor: pointer;
  }
  /* Focus: the global :focus-visible ring (base.css), never a halo of its own (DESIGN_DIRECTION §2.9). */
  .grid line {
    stroke: var(--chart-grid);
    stroke-width: 1;
    shape-rendering: crispEdges;
  }
  /* Minor gridlines (a log axis's unlabelled 2s and 5s): the grid's ink, dashed, so they read as subdivisions. */
  .grid line.minor {
    stroke-dasharray: 2 3;
  }
  .axis {
    stroke: var(--chart-axis);
    shape-rendering: crispEdges;
  }
  /* The x mark: a hairline in the axis ink, its span hatched along the axis, its label a 12 px caption. */
  .mark-line {
    stroke: var(--chart-axis);
    stroke-width: 1;
    shape-rendering: crispEdges;
  }
  .hatch {
    stroke: var(--line-strong);
    stroke-width: 1.5;
  }
  .mark-label {
    fill: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .ticks text {
    fill: var(--chart-label);
    font-size: var(--type-xs-size); /* small: chart tick */
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
  /* In light a 1.5 px dash of a series hue reads fainter than its contrast says (the dotted teal fourth run): 2 px. */
  :global([data-theme='light']) .line.dashed:not(.thin) {
    stroke-width: 2;
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
  /* The panel colour behind the line, so a gridline never runs through it. */
  .empty span {
    padding: 0 var(--space-2);
    background: var(--bg-panel);
  }
</style>
