// The line chart behind the run, compare and model screens: series on a linear or log y scale, with ticks, markers,
// a hover tooltip and legend, and series the pointer can pick out.
import { scaleLinear, scaleLog } from 'd3-scale';
import { bindable, computed, createUniqueId, css, cubicOut, defaults, effect, For, MediaQuery, signal, tick, Tween, untrack } from 'fw';
import { prefs } from '$lib/prefs';
import { media } from '$lib/design/tokens';
import { motion } from '$lib/ui/motion';
import { nearest } from './downsample';
import { logMinors, MAX_TICKS } from './domain';
import { END_LABELS_MIN, memoPlaceEndLabels } from './endLabels';
import { chartHint, claimOnce, TAP_HINT_KEY } from './hint';
import { paintOrder } from './layers';
import { TouchGesture, type PinAction } from './pin';
import { patternOf, dashOf, inkOf, seriesSummary, strokeWidth } from './lineSeries';
import { capInPlotOf, eventOf, futureOf, marginOf, markOf, xExtentOf, xTicksOf, yExtentOf, yFormatOf, yTicksOf, type Margin, type Scale } from './lineAxes';
import { mergesOf } from './linePaths';
import { memoPathsOf } from './linePathsMemo';
import { endRadius, endsOf, placedEndsOf } from './lineEndLabels';
import { hoverOf, pickNearest, pillOf } from './lineHover';
import { createRectCache } from './rectCache';
import { textMeasurer } from './textWidth';
import type { Future, Marker, Series, XMark } from './lineSeries';
import ChartLegend from './ChartLegend';
import ChartTooltip from './ChartTooltip';
import LineEnds from './LineEnds';

export type { Future, Marker, Series, XMark };

export interface LineChartProps {
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
  /**
   * An event on x (a failed run's "failed during epoch 3", design/UNIFIED.md §3 Frame): a hairline at `x` and its label
   * in the band above the plot, in the x mark's style but with no span; inside the plot's top where the x mark's label
   * holds its spot. The end mark says how the line ended; this says where.
   */
  xEvent?: XMark;
  /**
   * The part of a live run's planned axis not trained yet (DESIGN_DIRECTION §2.4): shaded from `x` (the latest point)
   * to the axis's end behind a 1 px rule, with `label` (the ETA) over the rule in the band above the plot, beside the x
   * mark's label (inside the plot's top when the two would touch). The plot keeps the band whenever a label is set.
   */
  future?: Future;
  /** Keep the band above the plot without a label in it: charts side by side then share their plot tops. */
  band?: boolean;
  /** The plot as an inset well (`bg.inset`), its marks haloed in the well's colour. Default on. */
  well?: boolean;
  /** A last legend entry after the series: the best epoch, its ring glyph before it ("○ best 0.656 @ 41"). */
  legendBest?: string;
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
  empty?: string | (() => JSX.Element);
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

/** Two numeric pairs (domains, extents) are equal: the same numbers, whatever the array. */
const samePair = (a: readonly number[], b: readonly number[]) => a[0] === b[0] && a[1] === b[1];
const sameMargin = (a: Margin, b: Margin) => a.top === b.top && a.right === b.right && a.bottom === b.bottom && a.left === b.left;

export default function LineChart(props: LineChartProps) {
  const p = defaults(props, {
    height: 200,
    yLog: false,
    curve: 'linear',
    markers: [],
    legend: true,
    yTicks: MAX_TICKS,
    xTicks: 6,
    xInteger: false,
    empty: 'No data yet',
    endLabels: false,
    tooltip: false,
    highlight: null,
    recede: false,
    endDots: false,
    well: true,
  });
  const cursor = bindable(props, 'cursor', null);
  const hint = bindable(props, 'hint', null);
  const picked = bindable(props, 'picked', null);
  // A pulse the chart was mounted with is old news (a tab switch, a return to a finished run): only a newer one plays.
  const pulseBase = untrack(() => Math.max(0, ...p.markers.map((m) => m.pulse ?? 0)));
  const focusBase = untrack(() => Math.max(0, ...p.markers.map((m) => m.focus ?? 0)));

  const uid = createUniqueId();
  const width = signal(0);

  const visible = computed(() => p.series.filter((s) => s.points.length > 0));
  const hasData = computed(() => visible.get().length > 0);

  // Extents compare by value: a poll that hands back equal data in new arrays moves no scale, no axis and no path.
  const xExtent = computed(() => xExtentOf(p.xDomain, visible.get()), {
    equals: samePair,
  });
  const yExtent = computed(() => yExtentOf(p.yDomain, visible.get(), p.yLog), {
    equals: samePair,
  });
  const yTickValues = computed(() => yTicksOf(p.yTickValues, yExtent.get(), p.yLog, p.yTicks));
  // A log axis labels its decades and ends; the 2s and 5s between them stay as unlabelled minor gridlines, so the
  // scale reads evenly (0.2 · 0.5 · 1 · 2 · 5 · 10) without crowding the labels.
  const yMinor = computed(() => (p.yLog && !p.yTickValues ? logMinors(yExtent.get()[0], yExtent.get()[1]) : []));
  const yFormat = computed(() => yFormatOf(p.yFormat, p.yLog, yTickValues.get(), p.yDecimals));
  const fmtValue = computed(() => p.valueFormat ?? p.yFormat ?? ((v: number) => String(+v.toFixed(3))));
  const textWidth = textMeasurer();
  /** Direct labels at each line's last point, each in a spot that covers no curve and no other label (endLabels.ts). */
  // One rule for every chart (DESIGN_DIRECTION §2.4): a plot under 560 px (every phone) has no direct labels, short
  // names included, and keeps that width for data; its legend carries each line's name, value and how it ended.
  const showEnds = computed(() => p.endLabels && width.get() >= END_LABELS_MIN);
  /** The band above the plot is kept for an x mark's label, an event's and the future's (the ETA). */
  const band = computed(() => !!p.band || !!p.xMark || !!p.xEvent || !!p.future?.label);
  const capInPlot = computed(() => capInPlotOf(p.yLabel, band.get(), p.yLog, yTickValues.get(), yExtent.get(), p.height, p.xLabel));
  const margin = computed(
    () => marginOf(yTickValues.get(), yFormat.get(), xExtent.get(), p.series, textWidth, showEnds.get(), p.yLabel, capInPlot.get(), band.get(), p.xLabel),
    { equals: sameMargin },
  );

  const x = computed(() =>
    scaleLinear()
      .domain(xExtent.get())
      .range([margin.get().left, Math.max(margin.get().left + 1, width.get() - margin.get().right)]),
  );
  // A y domain change eases over 240 ms (callers re-fit at most once per epoch, ./domain holdDomain; values themselves
  // never tween): the lines and gridlines glide to the new scale instead of jumping. Instant under reduced motion and
  // on a log scale. Only a domain whose numbers moved starts the ease (the Svelte original restarted it on every
  // recomputation, a 240 ms frame loop per poll on an idle chart), and the loop ends once it is there.
  const yEase = new Tween<[number, number]>(
    untrack(() => yExtent.get()),
    { easing: cubicOut },
  );
  let eased: { d: [number, number]; log: boolean } = untrack(() => ({
    d: yExtent.get(),
    log: p.yLog,
  }));
  effect(() => {
    const d = yExtent.get();
    const log = p.yLog;
    if (samePair(d, eased.d) && log === eased.log) return;
    eased = { d, log };
    untrack(() => void yEase.set(d, { duration: log ? 0 : motion(240) }));
  });
  const yShown = computed(() => (p.yLog ? yExtent.get() : yEase.current));
  const yScale = (d: [number, number]) => (p.yLog ? scaleLog().domain(d) : scaleLinear().domain(d)).range([p.height - margin.get().bottom, margin.get().top]);
  const y = computed<Scale>(() => yScale(yShown.get()));
  /** The y scale the axis is easing to (`y` once the ease ends): end labels are placed against it, once per change. */
  const yTo = computed<Scale>(() => yScale(yExtent.get()));
  // Default x labels: integers, or one decimal when the domain is too short for distinct integers.
  const xFormat = computed(() => p.xFormat ?? ((v: number) => (xExtent.get()[1] - xExtent.get()[0] < 5 ? String(+v.toFixed(1)) : String(Math.round(v)))));

  const xTickValues = computed(() => xTicksOf(width.get(), p.xTicks, xExtent.get(), p.xInteger, p.xDomain));
  // Path strings are built here, once per change of data or scale, and per series: a line whose points did not change
  // keeps its strings (./linePathsMemo). Nothing the cursor does reads this.
  const pathsOf = memoPathsOf();
  const paths = computed(() => pathsOf(visible.get(), p.curve, width.get(), margin.get(), p.height, x.get(), y.get(), p.yLog));
  const endR = (id: string) => endRadius(id, p.endDots, receding.get(), p.highlight, visible.get(), merges.get());
  const placeEnds = memoPlaceEndLabels();
  const placedEnds = computed(() =>
    placedEndsOf(showEnds.get(), visible.get(), x.get(), yTo.get(), textWidth, p.highlight, margin.get(), width.get(), p.height, placeEnds),
  );
  const ends = computed(() => endsOf(visible.get(), placedEnds.get(), y.get(), receding.get(), p.highlight, endR));
  const mark = computed(() => markOf(p.xMark, xExtent.get(), x.get(), width.get(), margin.get()));
  const future = computed(() => futureOf(p.future, mark.get(), xExtent.get(), x.get(), width.get(), margin.get(), p.height));
  const event = computed(() => eventOf(p.xEvent, mark.get(), xExtent.get(), x.get(), width.get(), margin.get()));

  // ---- hover / keyboard cursor ----
  const keyed = computed(() => visible.get().filter((s) => !s.ghost && !s.context && !s.lower));
  const longestSeries = computed(() => keyed.get().reduce<Series | undefined>((m, s) => (!m || s.points.length > m.points.length ? s : m), undefined));

  const hover = computed(() => hoverOf(cursor.get(), hasData.get(), xExtent.get(), keyed.get(), p.xTitle, xFormat.get(), x.get()));

  const rectCache = createRectCache();
  const local = (e: PointerEvent) => {
    const rect = rectCache.get(e.currentTarget as SVGElement);
    return { px: e.clientX - rect.left, py: e.clientY - rect.top };
  };
  const touchy = (e: PointerEvent) => e.pointerType === 'touch' || e.pointerType === 'pen';
  const inPlot = (px: number) => px >= margin.peek().left - 4 && px <= width.peek() - margin.peek().right + 4;

  // Touch: a tap or a sideways scrub pins the cursor and its readout until it is tapped again, a tap lands outside
  // the chart or Escape is pressed (./pin). A mouse reads out on hover, as before.
  const pinned = signal(false);
  let gesture: TouchGesture | null = null;
  const svgEl = signal<SVGSVGElement>();
  function apply(a: PinAction) {
    if (a.kind === 'clear') {
      speakSeq++;
      spoken.set('');
      cursor.set(null);
      pinned.set(false);
    } else if (a.kind === 'set' && inPlot(a.px)) {
      const m = margin.peek();
      cursor.set(x.peek().invert(Math.min(width.peek() - m.right, Math.max(m.left, a.px))));
      pinned.set(true);
      tapHint.set(false);
    }
  }

  // Reading hints in the readout row (./hint): "← → to inspect" while the plot has keyboard focus, and on a touch
  // screen "Tap to pin a value" on the first chart of the session, until it is tapped. A mouse gets neither, and the
  // phone tier and touch-only devices get no key hint (no arrow keys to press there).
  const keyFocus = signal(false);
  const keyless = new MediaQuery(`${media.phone}, (hover: none) and (pointer: coarse)`, false);
  const tapHint = signal(typeof matchMedia !== 'undefined' && matchMedia('(pointer: coarse)').matches && claimOnce(TAP_HINT_KEY));
  const hintText = computed(() =>
    hasData.get()
      ? chartHint({
          reading: hover.get() !== null,
          keyFocus: keyFocus.get() && !keyless.current,
          tap: tapHint.get(),
        })
      : null,
  );
  effect(() => {
    hint.set(hintText.get());
  });
  // Invalidate cached SVG rect when plot size or legend/readout state changes (e.g. a legend above the plot wrapping).
  const hasHover = computed(() => hover.get() !== null);
  effect(() => {
    void width.get();
    void p.height;
    void p.highlight;
    void picked.get();
    void hasHover.get();
    void hintText.get();
    rectCache.clear();
  });
  function onfocus() {
    try {
      keyFocus.set(!!svgEl.peek()?.matches(':focus-visible'));
    } catch {
      keyFocus.set(false);
    }
  }
  // A cursor cleared elsewhere (a synced chart, blur) is no longer pinned here, and picks nothing.
  effect(() => {
    if (cursor.get() === null) {
      pinned.set(false);
      picked.set(null);
    }
  });

  /** How near a line the pointer picks it out (`pick`), px: a mouse, and a finger. */
  const PICK_PX = 10;
  const PICK_TOUCH_PX = 20;
  function pickAt(px: number, py: number, r: number): string | null {
    return pickNearest(p.pick, inPlot, visible.peek(), x.peek(), y.peek(), px, py, r);
  }
  effect(() => {
    if (!pinned.get()) return;
    const outside = (e: PointerEvent) => {
      const el = svgEl.peek();
      if (el && !el.contains(e.target as Node)) apply({ kind: 'clear' });
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !e.defaultPrevented && !document.querySelector('dialog[open]')) {
        e.preventDefault();
        apply({ kind: 'clear' });
      }
    };
    document.addEventListener('pointerdown', outside, true);
    document.addEventListener('keydown', esc);
    return () => {
      document.removeEventListener('pointerdown', outside, true);
      document.removeEventListener('keydown', esc);
    };
  });

  function onpointerenter() {
    rectCache.clear();
  }
  function onpointerdown(e: PointerEvent) {
    speakSeq++;
    spoken.set('');
    if (!touchy(e)) return;
    rectCache.clear();
    const { px, py } = local(e);
    const h = hover.peek();
    gesture = new TouchGesture(px, py, pinned.peek() && h ? h.px : null);
  }
  function onpointerup(e: PointerEvent) {
    if (!touchy(e)) {
      // A click on a picked line (`pick`) acts on it.
      const id = picked.peek();
      if (id !== null && e.button === 0) {
        rectCache.clear();
        p.onpick?.(id);
      }
      return;
    }
    if (!gesture) return;
    const { px, py } = local(e);
    apply(gesture.up(px));
    gesture = null;
    picked.set(pinned.peek() ? pickAt(px, py, PICK_TOUCH_PX) : null);
  }
  function onpointercancel() {
    if (gesture) apply(gesture.cancel());
    gesture = null;
  }
  function onpointermove(e: PointerEvent) {
    speakSeq++;
    spoken.set('');
    const { px, py } = local(e);
    if (touchy(e)) {
      if (gesture) apply(gesture.move(px, py));
      return;
    }
    pinned.set(false);
    cursor.set(inPlot(px) ? x.peek().invert(px) : null);
    if (props.pick) picked.set(pickAt(px, py, PICK_PX));
  }
  function onpointerleave(e: PointerEvent) {
    rectCache.clear();
    if (!touchy(e) && !pinned.peek()) {
      cursor.set(null);
      picked.set(null);
    }
  }

  // Narrow plots: the cursor's x reads out on the axis, not in the legend's note.
  const compact = computed(() => width.get() > 0 && width.get() < 480);
  const pill = computed(() => pillOf(compact.get(), hover.get(), width.get()));

  // Keyboard scrubbing announces the values under the cursor. The text is written from the key handler itself, so
  // pointer movement here or on a linked chart never produces announcements.
  const spoken = signal('');
  let speakSeq = 0;

  const summary = computed(() => seriesSummary(keyed.get()));

  function onkeydown(e: KeyboardEvent) {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    const s = longestSeries.peek();
    if (!s || !s.points.length) return;
    const c = cursor.peek();
    const big = e.shiftKey ? 10 : 1;
    let i: number;
    if (e.key === 'ArrowLeft') i = c === null ? s.points.length - 1 : Math.max(0, nearest(s.points, c) - big);
    else if (e.key === 'ArrowRight') i = c === null ? 0 : Math.min(s.points.length - 1, nearest(s.points, c) + big);
    else if (e.key === 'Home') i = 0;
    else if (e.key === 'End') i = s.points.length - 1;
    else if (e.key === 'Escape') {
      if (c === null && !pinned.peek()) return;
      e.preventDefault();
      speakSeq++;
      cursor.set(null);
      pinned.set(false);
      picked.set(null);
      spoken.set('');
      return;
    } else return;
    e.preventDefault();
    cursor.set(s.points[i]!.x);
    const seq = ++speakSeq;
    // `hover` derives from the cursor; read it once the update has settled.
    void tick().then(() => {
      if (seq !== speakSeq) return;
      const h = hover.peek();
      const fmt = fmtValue.peek();
      spoken.set(h ? `${h.title}: ${h.rows.map((r) => `${r.s.label} ${fmt(r.p.y)}`).join(', ')}` : '');
    });
  }

  const light = computed(() => prefs.resolvedTheme === 'light');
  const widthOf = (s: Series): number => strokeWidth(s, receding.get(), p.highlight, light.get());
  /** A highlighted series with `recede`: it alone keeps weight and hue. */
  const receding = computed(() => p.recede && p.highlight !== null && visible.get().some((s) => s.id === p.highlight));
  /** Drawn as a context curve: a series of that kind, or a run receded behind the highlighted one. */
  const contextual = (s: Series) => !!s.context || (receding.get() && s.id !== p.highlight && !s.ghost);
  /** `items` with the highlighted series moved to the end, so it is painted over every other line. */
  function lifted<T>(items: T[], id: (t: T) => string): T[] {
    const h = p.highlight;
    if (h === null) return items;
    return [...items.filter((t) => id(t) !== h), ...items.filter((t) => id(t) === h)];
  }

  const merges = computed(() => mergesOf(p.endDots, hasData.get(), p.markers, keyed.get(), x.get(), y.get(), xExtent.get()));
  /** The marker drawn inside series `id`'s end mark, if any. */
  const mergedInto = (id: string): Marker | null => {
    const i = merges.get().indexOf(id);
    return i < 0 ? null : (p.markers[i] ?? null);
  };

  const tipWidth = signal(160);
  const tipLeft = computed(() => {
    const h = hover.get();
    return h ? h.px + 12 + tipWidth.get() > width.get() - 4 : false;
  });

  const bands = computed(() => paths.get().filter((q) => q.b));
  const lines = computed(() =>
    lifted(
      paintOrder(
        paths.get().filter((q) => !q.b),
        (q) => q.s,
      ),
      (q) => q.s.id,
    ),
  );

  /** Snippets of the data layer: the lines' ends (./LineEnds), and what the cursor reads (its line, dots and pill). */
  const endMarks = (): JSX.Element => (
    <LineEnds
      endDots={p.endDots}
      keyed={keyed.get()}
      receding={receding.get()}
      highlight={p.highlight}
      contextual={contextual}
      mergedInto={mergedInto}
      lifted={lifted}
      x={x.get()}
      y={y.get()}
      pulseBase={pulseBase}
      focusBase={focusBase}
      showEnds={showEnds.get()}
      ends={ends.get()}
      markers={p.markers}
      merges={merges.get()}
      xExtent={xExtent.get()}
      width={width.get()}
      margin={margin.get()}
    />
  );
  const cursorMarks = (): JSX.Element => (
    <>
      {hover.get() ? (
        <>
          <line
            class="cursor"
            class:pinned={pinned.get()}
            x1={hover.get()!.px}
            x2={hover.get()!.px}
            y1={margin.get().top}
            y2={p.height - margin.get().bottom}
          />
          <For each={hover.get()!.rows} key={(r) => r.s.id}>
            {(r) => (
              // None for a value under a fitted floor: the line is cut there, and the legend reads it.
              <>
                {Number.isFinite(r().p.y) && (!p.yLog || r().p.y > 0) && r().p.y >= yShown.get()[0] - 1e-9 ? (
                  <circle
                    class="dot"
                    cx={x.get()(r().p.x)}
                    cy={y.get()(r().p.y)}
                    r="3.5"
                    style:fill={contextual(r().s) ? 'var(--chart-context)' : r().s.color}
                  />
                ) : null}
              </>
            )}
          </For>
          {pill.get() ? (
            // Over the x ticks it covers: the cursor's x, which a wide legend states in its note.
            <g class="pill" transform={`translate(${pill.get()!.cx},${p.height - margin.get().bottom + 4})`}>
              <rect x={-pill.get()!.w / 2} y="0" width={pill.get()!.w} height="16" rx="3" />
              <text y="8" dy="0.32em" text-anchor="middle">
                {pill.get()!.text}
              </text>
            </g>
          ) : null}
        </>
      ) : null}
    </>
  );

  return (
    <div class="chart" class:well={p.well}>
      {p.legend && keyed.get().length > 0 ? (
        <ChartLegend
          hintText={hintText.get()}
          hover={hover.get()}
          compact={compact.get()}
          legendNote={p.legendNote}
          best={p.legendBest}
          keyed={keyed.get()}
          fmtValue={fmtValue.get()}
          width={width.get()}
        />
      ) : null}

      <p class="sr-only" aria-live="polite">
        {spoken.get()}
      </p>
      {keyed.get().length ? (
        // A table ignores width: 1px, so the visually-hidden box is a wrapper that clips it.
        <div class="sr-only">
          <table>
            <caption>{`${p.label}: summary by series. Focus the chart and use the arrow keys to read values point by point.`}</caption>
            <thead>
              <tr>
                <th scope="col">Series</th>
                <th scope="col">Latest</th>
                <th scope="col">Minimum</th>
                <th scope="col">Maximum</th>
              </tr>
            </thead>
            <tbody>
              <For each={summary.get()} key={(row) => row.s.id}>
                {(row) => (
                  <tr>
                    <th scope="row">{row().s.label}</th>
                    <td>{row().last === null ? '—' : fmtValue.get()(row().last!)}</td>
                    <td>{row().lo === null ? '—' : fmtValue.get()(row().lo!)}</td>
                    <td>{row().hi === null ? '—' : fmtValue.get()(row().hi!)}</td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </div>
      ) : null}

      <div class="plot" bind:clientWidth={width} style:height={`${p.height}px`}>
        {width.get() > 0 ? (
          <>
            <svg
              ref={svgEl}
              class:picking={picked.get() !== null && !!p.onpick}
              width={width.get()}
              height={p.height}
              role="group"
              aria-roledescription="chart"
              aria-label={p.label}
              tabindex={hasData.get() ? 0 : -1}
              onpointerenter={onpointerenter}
              onpointermove={onpointermove}
              onpointerleave={onpointerleave}
              onpointerdown={onpointerdown}
              onpointerup={onpointerup}
              onpointercancel={onpointercancel}
              onkeydown={onkeydown}
              onfocus={onfocus}
              onblur={() => {
                speakSeq++;
                spoken.set('');
                keyFocus.set(false);
                if (!pinned.peek()) cursor.set(null);
              }}
            >
              {p.well ? (
                // The plot as an inset well: the data's own surface, under the gridlines.
                <rect
                  class="well"
                  x={margin.get().left}
                  y={margin.get().top}
                  width={Math.max(0, width.get() - margin.get().right - margin.get().left)}
                  height={Math.max(0, p.height - margin.get().bottom - margin.get().top)}
                />
              ) : null}
              {future.get() ? (
                // The epochs still to train: a faint shade behind a 1 px rule at the latest point.
                <>
                  <rect class="future" x={future.get()!.px} y={future.get()!.top} width={future.get()!.end - future.get()!.px} height={Math.max(0, future.get()!.bottom - future.get()!.top)} />
                  <line class="future-rule" x1={future.get()!.px} x2={future.get()!.px} y1={future.get()!.top} y2={future.get()!.bottom} />
                </>
              ) : null}
              <g class="grid">
                <For each={yMinor.get()} key={(t) => t}>
                  {(t) => <line class="minor" x1={margin.get().left} x2={width.get() - margin.get().right} y1={y.get()(t())} y2={y.get()(t())} />}
                </For>
                <For each={yTickValues.get()} key={(t) => t}>
                  {(t) => <line x1={margin.get().left} x2={width.get() - margin.get().right} y1={y.get()(t())} y2={y.get()(t())} />}
                </For>
              </g>
              <line
                class="axis"
                x1={margin.get().left}
                x2={width.get() - margin.get().right}
                y1={p.height - margin.get().bottom}
                y2={p.height - margin.get().bottom}
              />
              {mark.get() ? (
                // The span after the mark hatched along the axis (the progress bar's hatch), the line, its label above the plot.
                <>
                  <defs>
                    <pattern id={`${uid}-hatch`} width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                      <line class="hatch" x1="0" y1="0" x2="0" y2="4" />
                    </pattern>
                  </defs>
                  <rect
                    x={mark.get()!.px}
                    y={p.height - margin.get().bottom - 6}
                    width={Math.max(0, mark.get()!.end - mark.get()!.px)}
                    height="6"
                    fill={`url(#${uid}-hatch)`}
                  />
                  <line class="mark-line" x1={mark.get()!.px} x2={mark.get()!.px} y1={margin.get().top - 4} y2={p.height - margin.get().bottom} />
                  <text class="mark-label" x={mark.get()!.tx} y={margin.get().top - 8} text-anchor={mark.get()!.anchor as 'start' | 'end'}>
                    {mark.get()!.label}
                  </text>
                </>
              ) : null}
              {event.get() ? (
                <>
                  <line class="mark-line" x1={event.get()!.px} x2={event.get()!.px} y1={margin.get().top - 4} y2={p.height - margin.get().bottom} />
                  <text class="mark-label" x={event.get()!.tx} y={event.get()!.ty} text-anchor={event.get()!.anchor as 'start' | 'end'}>
                    {event.get()!.label}
                  </text>
                </>
              ) : null}
              {future.get()?.label ? (
                <text class="mark-label" x={future.get()!.tx} y={future.get()!.ty}>
                  {future.get()!.label}
                </text>
              ) : null}
              <g class="ticks">
                <For each={yTickValues.get()} key={(t) => t}>
                  {(t) => (
                    <text x={margin.get().left - 8} y={y.get()(t())} dy="0.32em" text-anchor="end">
                      {yFormat.get()(t())}
                    </text>
                  )}
                </For>
                <For each={xTickValues.get()} key={(t) => t}>
                  {(t) => (
                    <text x={x.get()(t())} y={p.height - margin.get().bottom + 16} text-anchor="middle">
                      {xFormat.get()(t())}
                    </text>
                  )}
                </For>
                {p.xLabel ? (
                  <text class="x-label" x={(margin.get().left + width.get() - margin.get().right) / 2} y={p.height - 4} text-anchor="middle">
                    {p.xLabel}
                  </text>
                ) : null}
                {p.yLabel ? (
                  <text class="x-label" x="0" y={capInPlot.get() ? y.get()(Math.max(...yTickValues.get())) - 14 : 10} dy="0.32em" text-anchor="start">
                    {p.yLabel}
                  </text>
                ) : null}
              </g>

              {hasData.get() ? (
                <>
                  <defs>
                    <For each={paths.get()} key={(q) => q.s.id}>
                      {(q) => (
                        <>
                          {q().a ? (
                            <linearGradient id={`${uid}-${q().s.id}`} x1="0" x2="0" y1="0" y2="1">
                              <stop offset="0" class="area-top" style:stop-color={q().s.color} />
                              <stop offset="1" style:stop-color={q().s.color} stop-opacity="0" />
                            </linearGradient>
                          ) : null}
                        </>
                      )}
                    </For>
                    {/* A fitted domain that does not start at 0 (Compare's "Fit") cuts a line's early climb at the baseline. */}
                    <clipPath id={`${uid}-plot`}>
                      <rect x="0" y="0" width={width.get()} height={Math.max(0, p.height - margin.get().bottom + 1)} />
                    </clipPath>
                  </defs>
                  <g clip-path={`url(#${uid}-plot)`}>
                    <For each={bands.get()} key={(q) => q.s.id}>
                      {(q) => <path class="band" d={q().b} style:fill={q().s.color} />}
                    </For>
                    <For each={lines.get()} key={(q) => q.s.id}>
                      {(q) => {
                        const s = () => q().s;
                        const hl = () => s().id === p.highlight;
                        return (
                          <>
                            {q().a ? <path class="area" d={q().a} fill={`url(#${uid}-${s().id})`} /> : null}
                            {(patternOf(s()) || (receding.get() && hl())) && !s().ghost && !contextual(s()) ? (
                              // Patterned lines are painted over solid ones (./layers). A 1.5 px halo in the panel colour
                              // around each dash or dot keeps the pattern readable where it runs along a solid line, which
                              // still shows between them.
                              <path class="halo" d={q().d} style:stroke-width={`${widthOf(s()) + 3}px`} style:stroke-dasharray={dashOf(s(), hl() ? 1.4 : 1)} />
                            ) : null}
                            <path
                              class="line"
                              class:dashed={s().dashed || s().pattern === 'dash' || s().pattern === 'dashdot'}
                              class:dotted={s().pattern === 'dot'}
                              class:thin={s().thin}
                              class:ghost={s().ghost}
                              class:context={contextual(s())}
                              class:emphasis={s().emphasis || (p.highlight !== null && hl())}
                              d={q().d}
                              style:stroke={contextual(s()) ? undefined : inkOf(s())}
                              style:stroke-width={receding.get() && !s().ghost && !s().context ? `${widthOf(s())}px` : undefined}
                              style:stroke-dasharray={dashOf(s(), hl() ? 1.4 : 1)}
                            />
                          </>
                        );
                      }}
                    </For>
                  </g>
                  {endMarks()}

                  {cursorMarks()}
                </>
              ) : null}
            </svg>

            {!hasData.get() ? (
              <p class="empty">
                <span>{p.empty}</span>
              </p>
            ) : null}

            {hover.get() && p.tooltip ? <ChartTooltip hover={hover.get()!} tipLeft={tipLeft.get()} bind:tipWidth={tipWidth} fmtValue={fmtValue.get()} /> : null}
          </>
        ) : null}
      </div>
    </div>
  );
}

css`
  /* What a mark's halo, a ring's fill and the empty line's plate take: the surface under the plot. */
  .chart {
    --plot-bg: var(--bg-panel);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .chart.well {
    --plot-bg: var(--bg-inset);
  }
  .well {
    fill: var(--bg-inset);
  }
  /* The planned epochs: the hover overlay's faint tint (it reads on the well in both themes) and the rule at now. */
  .future {
    fill: var(--bg-hover);
  }
  .future-rule {
    stroke: var(--line-default);
    stroke-width: 1;
    shape-rendering: crispEdges;
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
    /* A press and drag over the plot reads values; it never selects the tick labels (runs F18). */
    user-select: none;
  }
  /* Over a line a click acts on (\`pick\`, \`onpick\`). */
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
    stroke: var(--plot-bg);
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
  /* The theme's chart.area_opacity, relative to its mode's default: the default themes draw 0.08 as before. */
  .area-top {
    stop-opacity: calc(var(--chart-area-opacity) / var(--chart-area-opacity-default) * 0.08);
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
    stroke: var(--plot-bg);
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
    background: var(--plot-bg);
  }
`;
