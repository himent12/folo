/**
 * Precision-recall curves with the per-class table as their legend. Every class is a 1 px context line
 * (`--chart-context`, >= 3:1 on the panel) and the mean over all classes a heavier line in ink; many classes are a
 * percentile band instead, so 80 classes read as one shape instead of spaghetti. Hovering a table row gives its curve
 * the class hue, clicking keeps it highlighted (the selection rule: bg.active and the 2 px ink bar), and a line
 * above the table says so and offers Clear. The legend row above the plot is the readout. Phones hide the precision
 * and recall columns: each row has two controls there, the class name traces its curve (`aria-pressed`) and the
 * chevron opens a details row with those two at their operating point (`aria-expanded`); "View curve"
 * (`onviewcurve`) is the selection bar's, once a class is traced. `view` shows only the curves or only the table: the model page renders them as two panels (the
 * curves beside the Details rail, the table in a row of its own; on a phone, two sections) that share `selected` and
 * `hovered` (bindable), so a class picked or pointed at in the table is traced in the other panel's curves, and
 * `onback` puts a way back to the table in the curves' selection bar. The legend row is a readout only (names and
 * values); on a phone, what acts on the selection (the way back, Clear) sits in a selection bar held under the section
 * picker, as on the Classes section (DESIGN_DIRECTION §2.12), with "Trace a class": the classes as chips right under
 * the bar, traced first (a tap removes one), so a phone traces a class without leaving the graph. `fill` lets the plot grow to the panel's height
 * (a panel stretched to its row), from a 300 px plot.
 */
import ArrowLeft from '$icons/arrow-left';
import ArrowRight from '$icons/arrow-right';
import ChevronDown from '$icons/chevron-down';
import Plus from '$icons/plus';
import Search from '$icons/search';
import { bindable, computed, createUniqueId, css, defaults, effect, For, MediaQuery, signal, untrack } from 'fw';
import type { Evaluation } from '$lib/api/evaluation';
import LineChart, { type Series } from '$lib/charts/LineChart';
import { classInk, classStroke } from '$lib/design/classInk';
import { imagesOf } from '$lib/evaluated';
import { metric, num } from '$lib/format';
import Input from '$lib/ui/Input';
import Segmented from '$lib/ui/Segmented';
import { classMetrics, matchesClass, prContext, prEnvelope, trimPrTail } from './pr';
import PrTracer from './PrTracer';
import { prDomain, PR_ZOOM, UNIT_ZOOM_NOTE } from '$lib/charts/domain';
import { media } from '$lib/design/tokens';

export interface PrCurvesProps {
  classes: readonly string[];
  ev: Evaluation;
  map50: number | null;
  view?: 'all' | 'curves' | 'classes';
  /** A details row's "View curve": the class is traced, then the page shows the curves. */
  onviewcurve?: (cls: number) => void;
  /** Back to the class table, offered in the curves' selection bar (after "View curve"). */
  onback?: () => void;
  /** The plot takes the height left in its panel (at least 300 px). */
  fill?: boolean;
  /** Classes traced in their hue (clicked rows), shared between a curves view and a table view. */
  selected?: number[];
  /** The class under the pointer or keyboard focus in the table. */
  hovered?: number | null;
  /** Many classes: every curve instead of the 10th–90th percentile band. */
  allCurves?: boolean;
  /** The band / all-curves switch sits in the panel's header (the page renders it from 768 px up; on phones this
   *  renders it at the end of the Curves bar), not at the legend row's end. */
  modeInHeader?: boolean;
}

/** The plot itself is 300 px (DESIGN_DIRECTION, Model detail); the chart is that plus its axes, LineChart's margins:
 *  the y axis name above (24) and the x ticks and name below (36). Round 22 drew 300 for the whole chart, so the plot
 *  was 240 and read smaller than the class table under it. Both axes run 0–1, so each is named at the axis
 *  ("Recall", "Precision"), with that room kept at every width, phones included (§2.4). */
const PLOT_H = 300;
/** A phone's plot: shorter, so the table (the legend) starts on the first screen. */
const PHONE_PLOT_H = 190;
const AXES_H = 24 + 36;
// Phones show four columns: a selected (highlighted) class also opens a details row with all five metrics.
const COLS_NARROW = 4;
// Large class sets: a filter and 16 rows until the reader asks for all of them.
const ROWS_MAX = 16;
/** The narrowest panel that holds the plot and the table side by side. */
const SIDE_MIN = 1100;
const classOf = (id: string | null) => (id !== null && id.startsWith('c') ? Number(id.slice(1)) : null);

export default function PrCurves(props: PrCurvesProps) {
  const p = defaults(props, { view: 'all', fill: false, modeInHeader: false });
  const selected = bindable(props, 'selected', []);
  const hovered = bindable(props, 'hovered', null);
  const allCurves = bindable(props, 'allCurves', false);
  /** The plot's box when it fills its panel: the chart is drawn at that height. */
  const plotH = signal(0);

  // Phones get a shorter plot so the table (the legend) starts on the first screen.
  const phoneQuery = new MediaQuery(media.phone, false);
  const narrow = () => phoneQuery.current;
  // A wide panel is one body: the plot on seven twelfths and the table, its legend, on five, scrolling inside itself at
  // the plot's height when the classes do not fit (80 COCO classes). The panel's own width decides, not the viewport's
  // tier: the table's six columns need about 460 px, which the panel of a 1440 px window has (and of a laptop's wider
  // windows, whose sidebar is icons) and of a 1280 px one does not; there the table follows the plot.
  const width = signal(0);
  const side = () => p.view === 'all' && width.get() >= SIDE_MIN;

  // Few classes: a leaderboard by AP. Many: by frequency.
  const many = computed(() => p.classes.length > 10);
  const order = computed(() => {
    const m = many.get();
    return p.ev.perClass.map((c) => c).sort((a, b) => (m ? b.instances - a.instances : b.ap50 - a.ap50));
  });

  const toggle = (i: number) => {
    const s = selected.peek();
    selected.set(s.includes(i) ? s.filter((x) => x !== i) : [...s, i]);
  };
  /** Phones: the classes whose details row is open (apart from tracing). */
  const opened = signal<number[]>([]);
  const disclose = (i: number) => opened.set((o) => (o.includes(i) ? o.filter((x) => x !== i) : [...o, i]));
  function viewCurve(i: number) {
    if (!selected.peek().includes(i)) selected.set([...selected.peek(), i]);
    p.onviewcurve?.(i);
  }
  /** Classes drawn in their hue, in legend order: the selection, then the hovered row. */
  const lit = computed(() => {
    const h = hovered.get();
    const s = selected.get();
    return h !== null && !s.includes(h) ? [...s, h] : s;
  });

  const name = (i: number) => p.classes[i] ?? `class ${i}`;
  // Each class's curve and the mean without their zero tails, trimmed once per evaluation (legacy trimmed every curve
  // again in the envelope, the zoom, every series rebuild and every cursor readout). The series then hand LineChart the
  // same arrays on a hover or a trace, so its path memo matches them by identity instead of comparing every point.
  const trimmed = computed(() => p.classes.map((_, i) => trimPrTail(p.ev.curves[i] ?? [])));
  const trimmedMean = computed(() => trimPrTail(p.ev.mean));
  // Curves end at their last recall point with precision > 0; the zero padding up to recall 1 is not data.
  // Many classes: by default a band (10th–90th percentile of the classes' precision) instead of 80 lines; every
  // curve is one click away. Few classes: every curve, as context lines.
  const banded = computed(() => many.get() && !allCurves.get());
  const envelope = computed(() => (many.get() ? prEnvelope(trimmed.get()) : null));
  const series = computed<Series[]>(() => {
    const env = envelope.get();
    const l = lit.get();
    return [
      ...(banded.get() && env
        ? [{ id: 'band', label: 'P10–P90 of classes', color: 'var(--text-tertiary)', points: env.upper, lower: env.lower }]
        : p.classes
            .map((_, i) => i)
            .filter((i) => !l.includes(i))
            .map((i) => ({ id: `c${i}`, label: name(i), color: 'var(--chart-context)', context: true, points: trimmed.get()[i]! }))),
      { id: 'mean', label: 'All classes', color: 'var(--text-primary)', points: trimmedMean.get(), emphasis: true },
      ...l.map((i) => ({ id: `c${i}`, label: name(i), color: classStroke(i), points: trimmed.get()[i]!, emphasis: true })),
    ];
  });

  // A near-perfect model (every class above 0.9 precision under recall 0.95) is a flat line on a 0–1 frame's top edge:
  // its axis is zoomed to 0.8–1, says so, and the plot is shorter (./domain prDomain, DESIGN_DIRECTION §2.4).
  const zoom = computed(() => prDomain(trimmed.get()));
  const plotPx = computed(() => (narrow() ? PHONE_PLOT_H : zoom.get() ? PR_ZOOM.plotPx : PLOT_H));

  // Cursor on the shared recall axis (bound to the chart); the legend reads each curve's precision there.
  const cursor = signal<number | null>(null);
  // A quiet grey curve is identified where it is drawn: the pointer near it picks out the nearest class (LineChart
  // `pick`), which takes its hue and its name and value in the readout, as a hovered table row does; a click traces it.
  // A tap picks it too, and the readout offers Trace (a tap elsewhere in the plot lets it go).
  const picked = signal<string | null>(null);
  /** Whether `hovered` was set by a pick on the chart (so letting the pick go clears it, not a table row's hover). */
  let pickedLit = false;
  effect(() => {
    const c = classOf(picked.get());
    untrack(() => {
      if (c !== null || (hovered.get() !== null && pickedLit)) hovered.set(c);
      pickedLit = c !== null;
    });
  });
  /** The class a tap picked, kept for the readout's Trace button until it is traced or a tap picks nothing. */
  const tapped = signal<number | null>(null);
  function onplotup(e: PointerEvent) {
    if (e.pointerType === 'mouse') return;
    // After the chart has handled the tap (it sets `picked`).
    queueMicrotask(() => tapped.set(classOf(picked.peek())));
  }
  const traceTapped = () => {
    const t = tapped.peek();
    if (t !== null && !selected.peek().includes(t)) selected.set([...selected.peek(), t]);
    tapped.set(null);
  };
  /** The precision at the cursor on a curve already trimmed of its tail. */
  function at(pts: { x: number; y: number }[]): string {
    const c = cursor.get();
    if (c === null || !pts.length) return '—';
    const last = pts[pts.length - 1]!;
    if (c > last.x + 1e-9) return '—';
    let best = pts[0]!;
    for (const pt of pts) if (Math.abs(pt.x - c) < Math.abs(best.x - c)) best = pt;
    return metric(best.y);
  }

  const meanAp = computed(() => p.map50 ?? p.ev.perClass.reduce((a, c) => a + c.ap50, 0) / Math.max(1, p.ev.perClass.length));
  const apOf = computed(() => new Map(p.ev.perClass.map((c) => [c.class, c.ap50])));

  const uid = createUniqueId();
  const tracing = signal(false);

  const query = signal('');
  const all = signal(false);
  const listed = computed(() => {
    const q = query.get();
    return order.get().filter((c) => matchesClass(name(c.class), q));
  });
  const rows = computed(() => (all.get() || query.get() || side() ? listed.get() : listed.get().slice(0, ROWS_MAX)));

  /** The band / every-curve switch (Segmented's value). */
  const curvesMode = { get: () => (allCurves.get() ? 'all' : 'band') as 'all' | 'band', set: (v: 'all' | 'band') => allCurves.set(v === 'all') };
  const n = () => selected.get().length;
  const traceButton = (cls: string) => (
    <button type="button" class={cls} aria-expanded={tracing.get()} aria-controls={`${uid}-trace`} onclick={() => tracing.set(!tracing.get())}>
      {n() ? (
        <>
          {n()} {n() === 1 ? 'class' : 'classes'} traced
        </>
      ) : (
        <>
          <Plus size={14} aria-hidden="true" />
          Trace a class
        </>
      )}
      <span class="chev" class:open={tracing.get()} aria-hidden="true">
        <ChevronDown size={14} />
      </span>
    </button>
  );

  return (
    <div class="pr" class:side={side()} bind:clientWidth={width}>
      <div class="part" class:off={p.view === 'classes'}>
        {/* Phones' Curves section has no table beside it: its selection bar, held under the section picker, says what is
            traced and carries the way back and Clear, so the legend below stays a readout. */}
        {p.view === 'curves' && narrow() ? (
          <div class="tools bar">
            {/* The picker's disclosure says what is traced ("2 classes traced"), so the bar keeps one line. */}
            <p class="pick num" aria-live="polite">
              {traceButton('view trace')}
              {p.onback ? (
                <button type="button" class="view back" onclick={p.onback}>
                  <ArrowLeft size={14} aria-hidden="true" />
                  Back to Classes
                </button>
              ) : null}
              {n() ? (
                <button type="button" class="clear" onclick={() => selected.set([])}>
                  Clear
                </button>
              ) : null}
            </p>
            {/* Many classes: the band / every-curve switch the page's header holds from 768 px up, here at the bar's end. */}
            {many.get() && p.modeInHeader ? (
              <span class="bar-mode">
                <Segmented
                  size="sm"
                  label="Curves shown"
                  bind:value={curvesMode}
                  options={[
                    { value: 'band', label: 'Band' },
                    { value: 'all', label: 'All curves' },
                  ]}
                />
              </span>
            ) : null}
          </div>
        ) : null}
        {/* At rest the legend states each curve's AP50 (a summary of the whole curve); under the cursor it switches to a
            readout, marked by the cursor's own glyph: the precision at that recall. */}
        <div class="legend" class:reading={cursor.get() !== null}>
          {cursor.get() === null ? (
            <span class="what">
              AP50 <span class="what-sub">area under each curve</span>
            </span>
          ) : (
            <span class="what at num">
              <span class="cur" aria-hidden="true"></span>
              Precision at recall {cursor.get()!.toFixed(2)}
            </span>
          )}
          <span class="item">
            <span class="line mean" aria-hidden="true"></span>
            All classes <span class="num">{cursor.get() === null ? metric(meanAp.get()) : at(trimmedMean.get())}</span>
          </span>
          <For each={lit.get()} key={(i) => i}>
            {(i) => (
              <span class="item">
                <span class="line" style:background={classStroke(i())} aria-hidden="true"></span>
                {name(i())} <span class="num">{cursor.get() === null ? metric(apOf.get().get(i())) : at(trimmed.get()[i()] ?? [])}</span>
              </span>
            )}
          </For>
          {/* The class picked out on the plot: a click traces it (or lets it go); a tap offers Trace here. */}
          {tapped.get() !== null && !selected.get().includes(tapped.get()!) ? (
            <button type="button" class="clear trace-pick" onclick={traceTapped}>
              <Plus size={14} aria-hidden="true" />
              Trace {name(tapped.get()!)}
            </button>
          ) : classOf(picked.get()) !== null ? (
            <span class="hint">Click to {selected.get().includes(classOf(picked.get())!) ? 'stop tracing' : 'trace'}</span>
          ) : null}
          {banded.get() ? (
            <span class="item band-key">
              <span class="swatch-band" aria-hidden="true"></span>
              10th–90th percentile of classes
            </span>
          ) : null}
          {p.view !== 'classes' && !narrow() ? (
            // The class picker, as on phones: the classes as chips under this row, the traced first.
            <>
              {traceButton('view trace mode')}
              {n() ? (
                <button type="button" class="clear" onclick={() => selected.set([])}>
                  Clear
                </button>
              ) : null}
            </>
          ) : null}
          {many.get() && !p.modeInHeader ? (
            <button type="button" class="clear mode" aria-pressed={allCurves.get()} onclick={() => allCurves.set(!allCurves.get())}>
              {allCurves.get() ? 'Show band' : 'All curves'}
            </button>
          ) : null}
        </div>
        {p.view !== 'classes' ? (
          <PrTracer id={`${uid}-trace`} open={tracing.get()} order={order.get().map((c) => c.class)} name={name} many={many.get()} bind:selected={selected} />
        ) : null}
        <div class="plot" class:fill={p.fill} bind:clientHeight={plotH} onpointerup={onplotup}>
          <LineChart
            series={series.get()}
            height={p.fill && !narrow() ? Math.max(plotPx.get() + AXES_H, plotH.get() - 24) : plotPx.get() + AXES_H}
            xDomain={[0, 1]}
            yDomain={zoom.get()?.domain ?? [0, 1]}
            yTickValues={zoom.get()?.ticks}
            yDecimals={zoom.get()?.decimals}
            pick={(s) => s.id.startsWith('c')}
            bind:picked={picked}
            onpick={(id) => toggle(classOf(id)!)}
            xFormat={(x) => x.toFixed(1)}
            valueFormat={(y) => y.toFixed(3)}
            xTicks={6}
            yTicks={5}
            xLabel="Recall"
            yLabel={zoom.get() ? `Precision · ${UNIT_ZOOM_NOTE}` : 'Precision'}
            legend={false}
            bind:cursor={cursor}
            empty="No correct detections yet: precision is 0 at every recall."
            label={`Precision-recall curves at IoU 0.5 for ${p.classes.length} classes and the mean over all classes`}
          />
        </div>
      </div>

      <div class="part legend-t" class:off={p.view === 'curves'}>
        {/* Beside the plot (side): out of flow, so the plot's column alone sets the panel's height. */}
        <div class="t-in">
          {/* The rows are the curves' controls: say so, and how many are traced, with a way back. Phones: with classes
              traced, this row is the selection's action bar ("View curve"), held under the section picker while the
              table scrolls. */}
          <div class="tools" class:bar={narrow() && !!p.onviewcurve && n() > 0}>
            {many.get() ? (
              <div class="filter">
                <Input bind:value={query} icon={Search} size="sm" type="search" placeholder="Find a class" aria-label="Find a class" />
              </div>
            ) : null}
            <p class="pick num" aria-live="polite">
              {n() ? (
                <>
                  <span>
                    {n()} {n() === 1 ? 'class' : 'classes'} traced
                  </span>
                  {p.onviewcurve ? (
                    <button type="button" class="view to-curve" onclick={() => viewCurve(selected.peek()[selected.peek().length - 1]!)}>
                      {n() === 1 ? 'View curve' : 'View curves'}
                      <ArrowRight size={14} aria-hidden="true" />
                    </button>
                  ) : null}
                  <button type="button" class="clear" onclick={() => selected.set([])}>
                    Clear
                  </button>
                </>
              ) : (
                <span class="hint">{narrow() ? 'Tap a class to trace its curve, the arrow for its details' : side() ? 'Hover a class to see its curve, click to trace it' : 'Select classes to trace their curves'}</span>
              )}
            </p>
          </div>
          <div class="scroll">
            <table class="table dense" class:fixed={narrow() || side()}>
              {/* Phones: one column definition for the header and every row (fixed layout), so the headings sit over their
                  numbers and the class column ends a clear gutter before AP50. */}
              {narrow() ? (
                <colgroup>
                  <col />
                  <col class="c-ap" />
                  <col class="c-ap95" />
                  <col class="c-inst" />
                </colgroup>
              ) : side() ? (
                // Beside the plot: each figure as wide as its heading and an 8 px gutter; the class takes the rest (a long
                // name truncates, its button keeps the whole name).
                <colgroup>
                  <col />
                  <col class="s-ap" />
                  <col class="s-ap95" />
                  <col class="s-p" />
                  <col class="s-r" />
                  <col class="s-inst" />
                </colgroup>
              ) : null}
              {/* Wider screens: bounded columns (the class 224 px, each number 96 px) and an empty last column that takes
                  what is left, so 3 or 80 classes read as one compact block on the left, never figures strewn across the
                  panel. */}
              <thead>
                <tr>
                  <th scope="col" class="cls-h" style:width={narrow() || side() ? undefined : '224px'}>
                    Class
                  </th>
                  <th scope="col" class="r" style:width={narrow() || side() ? undefined : '96px'}>
                    AP50
                  </th>
                  <th scope="col" class="r" style:width={narrow() || side() ? undefined : '96px'}>
                    AP50-95
                  </th>
                  <th scope="col" class="r hide-sm" style:width={narrow() || side() ? undefined : '96px'}>
                    Precision
                  </th>
                  <th scope="col" class="r hide-sm" style:width={narrow() || side() ? undefined : '96px'}>
                    Recall
                  </th>
                  <th scope="col" class="r" style:width={narrow() || side() ? undefined : '96px'}>
                    Instances
                  </th>
                  {!narrow() && !side() ? <th class="rest" aria-hidden="true"></th> : null}
                </tr>
              </thead>
              <tbody>
                <For
                  each={rows.get()}
                  key={(c) => c.class}
                  fallback={
                    <tr>
                      <td colspan={narrow() ? COLS_NARROW : side() ? 6 : 7} class="none">
                        No class matches “{query.get().trim()}”
                      </td>
                    </tr>
                  }
                >
                  {(c) => {
                    const on = () => selected.get().includes(c().class);
                    const open = () => narrow() && opened.get().includes(c().class);
                    return (
                      <>
                        <tr
                          class="link"
                          class:on={on()}
                          class:hot={hovered.get() === c().class}
                          onpointerenter={() => hovered.set(c().class)}
                          onpointerleave={() => hovered.set(null)}
                          onclick={() => toggle(c().class)}
                        >
                          <td>
                            {/* The class name traces its curve (a toggle). Phones: the chevron beside it is a separate
                                control that opens the details row. */}
                            <div class="cls-cell">
                              <button
                                type="button"
                                class="cls"
                                data-cls={c().class}
                                aria-pressed={on()}
                                aria-label={`Trace ${name(c().class)}`}
                                onclick={(e) => (e.stopPropagation(), toggle(c().class))}
                                onfocus={() => hovered.set(c().class)}
                                onblur={() => hovered.set(null)}
                              >
                                <span class="sw" style:background={classInk(c().class)} aria-hidden="true"></span>
                                <span class="truncate">{name(c().class)}</span>
                              </button>
                              {narrow() ? (
                                <button
                                  type="button"
                                  class="disc"
                                  data-cls={c().class}
                                  aria-expanded={open()}
                                  aria-controls={`pr-detail-${c().class}`}
                                  aria-label={`${name(c().class)} details`}
                                  onclick={(e) => (e.stopPropagation(), disclose(c().class))}
                                >
                                  <span class="chev" class:open={open()} aria-hidden="true">
                                    <ChevronDown size={14} />
                                  </span>
                                </button>
                              ) : null}
                            </div>
                          </td>
                          <td class="r">{metric(c().ap50)}</td>
                          <td class="r muted">{metric(c().ap50_95)}</td>
                          <td class="r hide-sm muted">{metric(c().precision)}</td>
                          <td class="r hide-sm muted">{metric(c().recall)}</td>
                          <td class="r muted">{num(c().instances)}</td>
                          {!narrow() && !side() ? <td class="rest" aria-hidden="true"></td> : null}
                        </tr>
                        {open() ? (
                          <tr class="detail" class:on={on()} id={`pr-detail-${c().class}`}>
                            <td colspan={COLS_NARROW}>
                              <dl>
                                <For each={classMetrics(c())} key={(d) => d.label}>
                                  {(d) => (
                                    <div>
                                      <dt>{d().label}</dt>
                                      <dd class="num">{d().value}</dd>
                                    </div>
                                  )}
                                </For>
                              </dl>
                              {/* The way to the curve is the selection bar's "View curve" alone (one action, one place). */}
                              <p class="ctx">{prContext(p.ev.bestF1Conf)}</p>
                            </td>
                          </tr>
                        ) : null}
                      </>
                    );
                  }}
                </For>
              </tbody>
            </table>
          </div>
          {!query.get() && !side() && listed.get().length > ROWS_MAX ? (
            <button type="button" class="more" aria-expanded={all.get()} onclick={() => all.set(!all.get())}>
              {all.get() ? `Show the first ${ROWS_MAX}` : `Show all ${listed.get().length}`}
            </button>
          ) : null}
          {/* The split actually evaluated, never "validation" by default: the page's notice says why when it is not. */}
          <p class="foot">
            Evaluated on {imagesOf(p.ev.source, p.ev.images)} at IoU 0.50. Precision and recall {prContext(p.ev.bestF1Conf, false)}; instances are
            ground-truth boxes per class.
          </p>
        </div>
      </div>
    </div>
  );
}

css`
  .pr {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  /* The two halves lay out as direct children of .pr (the foot keeps pinning to the bottom); \`view\` hides one. */
  .part,
  .t-in {
    display: contents;
  }
  .part.off {
    display: none;
  }
  /* From the laptop tier (side): the plot and its readout on the left seven twelfths, the table on the right five,
     parted by a hairline. The table's column is as tall as the plot's: its content sits out of flow and scrolls
     inside it (its head held at the top), so 3 classes or 80 never change the panel's height. */
  .pr.side {
    display: grid;
    grid-template-columns: minmax(0, 7fr) minmax(0, 5fr);
  }
  .side > .part {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .side > .legend-t {
    position: relative;
    border-left: 1px solid var(--line-subtle);
  }
  .side .t-in {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
  }
  .side .scroll {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    overscroll-behavior: contain;
  }
  .side .scroll thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: var(--bg-panel);
  }
  .side .cls {
    max-width: 100%;
  }
  .side .fixed {
    table-layout: fixed;
  }
  .s-ap {
    width: 56px;
  }
  .s-ap95 {
    width: 72px;
  }
  .s-p {
    width: 76px;
  }
  .s-r {
    width: 60px;
  }
  .s-inst {
    width: 84px;
  }
  /* Six columns in five twelfths: 8 px gutters between the figures. */
  .side .scroll :is(th, td):not(:first-child) {
    padding-left: var(--space-2);
  }
  .side .scroll :is(th, td):not(:last-child) {
    padding-right: var(--space-2);
  }
  .swatch-band {
    width: 14px;
    height: 8px;
    border-radius: 2px;
    background: var(--text-tertiary);
    opacity: 0.35;
  }
  .mode {
    margin-left: auto;
  }
  .trace-pick {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    color: var(--text-primary);
  }
  .what {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 13em;
    color: var(--text-secondary);
  }
  .what-sub {
    color: var(--text-tertiary);
  }
  /* The readout state: the cursor's dashed rule as its key, values in place of the AP summary. */
  .what.at {
    color: var(--text-primary);
  }
  .cur {
    width: 1px;
    height: 12px;
    background: repeating-linear-gradient(to bottom, var(--chart-cursor) 0 2px, transparent 2px 4px);
    flex-shrink: 0;
  }
  .legend {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-4);
    min-height: 24px;
    padding: 0 var(--space-4);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .item {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    white-space: nowrap;
  }
  .item .num {
    color: var(--text-primary);
  }
  .line {
    width: 12px;
    height: 2px;
    border-radius: 1px;
    flex-shrink: 0;
  }
  .line.mean {
    background: var(--text-primary);
  }
  .hint {
    color: var(--text-tertiary);
  }
  .clear {
    height: 20px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
  }
  .clear:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  /* Class curves are 1 px hairlines; the mean and the highlighted classes 2 px. */
  .plot {
    padding: var(--space-2) var(--space-4) var(--space-4);
  }
  /* Filling a stretched panel: the box takes the height left (its 8 + 16 px padding around the chart), and the chart,
     drawn at that height, sits out of flow so it never feeds back into the box it measures. */
  .plot.fill {
    flex: 1;
    position: relative;
    /* The chart (plot and axes, 360) and the box's 8 + 16 px padding. */
    min-height: 384px;
  }
  .plot.fill > :global(*) {
    position: absolute;
    inset: var(--space-2) var(--space-4) var(--space-4);
  }
  .plot :global(path.line) {
    stroke-width: 1;
  }
  .plot :global(path.line.emphasis) {
    stroke-width: 2;
  }
  .tools {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2) var(--space-4);
    padding: 0 var(--space-4) var(--space-2);
  }
  /* Under the sticky section picker (the page sets --pick-bottom, 8 px under it), on the panel's fill. It starts where
     the header above it ends, never pulled up over it: a bar that paints its own fill must not cover a neighbour's
     control (round 27: the header's Band / All curves lost its bottom 6 px to a -8 px margin here). One row: what is
     traced on the left, the curves' mode on the right. */
  .tools.bar {
    position: sticky;
    top: var(--pick-bottom, var(--size-topbar));
    z-index: 1;
    flex-wrap: nowrap;
    justify-content: space-between;
    background: var(--bg-panel);
  }
  .bar-mode {
    display: flex;
    flex-shrink: 0;
  }
  /* What is traced and its ways back wrap inside their own part, so the mode keeps its place at the row's end. */
  .tools.bar .pick {
    flex: 1;
    flex-wrap: wrap;
    min-width: 0;
  }
  .view.to-curve,
  .view.trace {
    margin-top: 0;
  }
  .filter {
    width: 240px;
    max-width: 100%;
  }
  .pick {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 28px;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .scroll {
    overflow-x: auto;
    border-top: 1px solid var(--line-subtle);
  }
  /* The selection rule: a neutral bg.active fill and the 2 px ink bar on the row's leading edge. Hover is a lighter
     neutral step, and adds to the fill of a selected row, so the two stay distinct. */
  tr.hot {
    background: var(--bg-hover);
  }
  tr.on:not(.detail) {
    background: var(--bg-active);
  }
  tr.on.hot {
    background: linear-gradient(var(--bg-hover), var(--bg-hover)), var(--bg-active);
  }
  tr.on td:first-child {
    box-shadow: var(--select-bar);
  }
  /* The heading starts where the names do, past the swatch (8 px) and its gap (8 px). */
  .cls-h {
    padding-left: calc(var(--space-4) + var(--space-4));
  }
  .rest {
    padding: 0;
  }
  .cls-cell {

    display: flex;
    align-items: center;
    min-width: 0;
  }
  .cls {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    max-width: 200px;
    text-align: left;
  }
  .disc {
    display: grid;
    place-items: center;
    flex-shrink: 0;
    width: 28px;
    height: 28px;
    margin-left: auto;
    border-radius: var(--radius-sm);
    color: var(--text-tertiary);
  }
  .disc:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }
  .view.back {
    margin-top: 0;
  }
  .view {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    min-height: 28px;
    margin-top: var(--space-2);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-primary);
  }
  .view:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--line-strong);
  }
  .cls:focus-visible {
    outline-offset: 2px;
  }
  .chev {
    display: inline-flex;
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .chev.open {
    transform: rotate(180deg);
  }
  /* Phones: every metric of an opened class, under its row (precision and recall have no column there). Indented to
     the class names (past the swatch), like the Class heading. One layer in both themes: a raised well under a hairline,
     never a second selection fill; a traced class's leading bar runs on through it. */
  tr.detail td {
    height: auto;
    padding: var(--space-2) var(--space-4) var(--space-3) calc(var(--space-4) + var(--space-4));
    background: var(--bg-raised);
    box-shadow: inset 0 1px 0 var(--line-subtle);
  }
  tr.detail.on td {
    box-shadow:
      var(--select-bar),
      inset 0 1px 0 var(--line-subtle);
  }
  .detail dl {
    display: grid;
    grid-template-columns: repeat(2, max-content);
    gap: var(--space-2) var(--space-8);
  }
  .ctx {
    margin-top: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .detail dt {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .detail dd {
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    color: var(--text-primary);
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .none {
    text-align: center;
    color: var(--text-tertiary);
  }
  .more {
    display: block;
    width: 100%;
    height: var(--control-lg);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    color: var(--text-secondary);
    transition:
      background-color var(--dur-fast) var(--ease-standard),
      color var(--dur-fast) var(--ease-standard);
  }
  .more:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .more:focus-visible {
    outline-offset: -2px;
  }
  .foot {
    margin-top: auto;
    padding: var(--space-3) var(--space-4);
    border-top: 1px solid var(--line-subtle);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  @media (pointer: coarse) {
    .more {
      height: 44px;
    }
    .clear {
      height: 44px;
    }
    /* Touch: every class row is a 44 px target (fine pointers keep the dense 28 px rows). */
    tbody td {
      height: 44px;
    }
    .cls {
      min-height: 44px;
    }
    .disc {
      width: 44px;
      height: 44px;
      margin-right: calc(var(--space-3) * -1);
    }
    .view {
      min-height: 44px;
      margin-top: 0;
    }
  }
  @media (max-width: 767px) {
    /* The trace toggle takes the class cell's width, so every row's chevron sits on one edge. */
    .cls {
      flex: 1;
      max-width: none;
    }
    .fixed {
      table-layout: fixed;
    }
    /* Each number column is as wide as its heading plus a 12 px gutter on both sides (16 at the table's edge). */
    .c-ap {
      width: 64px;
    }
    .c-ap95 {
      width: 80px;
    }
    .c-inst {
      width: 88px;
    }
  }
`;
