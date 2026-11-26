/**
 * The plot the benchmark screens draw their results on: lines and dots on two linear scales, with what reading them
 * needs as first-class marks: intervals (whiskers with caps), a band across the plot (a 95% interval every point is
 * read against), reference levels and the diagonal, labelled points, and a strip of counts under the plot. It keeps
 * Studio's chart conventions (LineChart, Histogram): recessive grid and axis, 11 px ticks, both axes named at the axes
 * where both run 0–1, the readout row above the plot as the legend (the caller's snippet, given the cursor), no
 * floating tooltip; a mouse reads where it points, touch pins (./pin), the keyboard steps through `stops` with the
 * arrows, Home and End. Assistive tech gets the chart's own table (`table`).
 */
import { scaleLinear } from 'd3-scale';
import { bindable, computed, css, defaults, effect, For, Index, MediaQuery, signal } from 'fw';
import { media } from '$lib/design/tokens';
import { seriesText } from './ink';
import { canDismissChartOnEscape, TouchGesture, type PinAction } from './pin';
import { chartHint, claimOnce, TAP_HINT_KEY } from './hint';
import { createRectCache } from './rectCache';
import { spokenAt, type XYMark, type XYPoint, type XYRef, type XYSeries, type XYStrip } from './xy';

export interface XYChartProps {
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
  readout: (cursor: () => number | null, hint: () => string | null) => JSX.Element;
  table: { caption: string; head: string[]; rows: string[][] };
  /** What the keyboard's cursor says aloud at a stop (default: ./xy spokenAt, from the table or the series). */
  speak?: (x: number) => string;
}

const STRIP_H = 28;
const DASH = { dash: '6 3', dot: '0.01 4.5' } as const;
const fmt = (v: number) => String(v);
const samePair = (a: readonly [number, number], b: readonly [number, number]) => a[0] === b[0] && a[1] === b[1];
const sameNums = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((v, i) => v === b[i]);
const sameMargin = (a: { top: number; right: number; bottom: number; left: number }, b: { top: number; right: number; bottom: number; left: number }) =>
  a.top === b.top && a.right === b.right && a.bottom === b.bottom && a.left === b.left;

export default function XYChart(props: XYChartProps) {
  const p = defaults(props, { xFormat: fmt, yFormat: fmt, height: 240, refs: [] as XYRef[], band: null, marks: [] as XYMark[], strip: null, highlight: null });
  const cursor = bindable(props, 'cursor', null);

  const width = signal(0);
  const xDom = computed(() => p.xDomain, { equals: samePair });
  const yDom = computed(() => p.yDomain, { equals: samePair });
  const stops = computed(
    () => (p.stops ? p.stops.slice() : [...new Set(p.series.flatMap((s) => s.points.filter((q) => q.y !== null).map((q) => q.x)))]).sort((a, b) => a - b),
    { equals: sameNums },
  );
  const m = computed(
    () => {
      const strip = p.strip;
      const longest = Math.max(1, ...p.yTicks.map((t) => p.yFormat(t).length), strip ? String(Math.max(0, ...strip.items.map((i) => i.v))).length : 0);
      return {
        top: p.yLabel ? 24 : 10,
        right: 14,
        bottom: 22 + (p.xLabel ? 16 : 0) + (strip ? STRIP_H + 12 : 0),
        left: Math.ceil(longest * 7) + 14,
      };
    },
    { equals: sameMargin },
  );
  const plotBottom = computed(() => p.height - m.get().bottom);
  const x = computed(() =>
    scaleLinear()
      .domain(xDom.get())
      .range([m.get().left, Math.max(m.get().left + 1, width.get() - m.get().right)]),
  );
  const y = computed(() => scaleLinear().domain(yDom.get()).range([plotBottom.get(), m.get().top]));
  const clampY = (v: number) => {
    const [lo, hi] = yDom.get();
    return Math.max(lo, Math.min(hi, v));
  };
  const inkOf = (s: XYSeries) => (s.pattern ? seriesText(s.color) : s.color);
  const activeHighlight = computed(() => (p.highlight !== null && p.series.some((t) => t.id === p.highlight) ? p.highlight : null));
  const receded = (s: XYSeries) => activeHighlight.get() !== null && s.id !== activeHighlight.get();

  /** Each series' line as an SVG path, broken where a point has no value: memoized per series & scale. */
  const pathCache = new Map<string, { points: readonly XYPoint[]; d: string }>();
  let pathScaleKey = '';
  const paths = computed(() => {
    const xx = x.get();
    const yy = y.get();
    const [yLo, yHi] = yDom.get();
    const [xLo, xHi] = xDom.get();
    const mg = m.get();
    const key = `${xLo},${xHi}|${yLo},${yHi}|${width.get()}|${p.height}|${mg.left},${mg.right},${mg.top},${mg.bottom}`;
    if (key !== pathScaleKey) {
      pathScaleKey = key;
      pathCache.clear();
    }
    const out = new Map<string, string>();
    const seen = new Set<string>();
    for (const s of p.series) {
      seen.add(s.id);
      const cached = pathCache.get(s.id);
      if (cached && cached.points === s.points) {
        out.set(s.id, cached.d);
        continue;
      }
      let d = '';
      let pen = false;
      for (const q of s.points) {
        if (q.y === null || !Number.isFinite(q.y)) {
          pen = false;
          continue;
        }
        d += `${pen ? 'L' : 'M'}${xx(q.x).toFixed(1)} ${yy(Math.max(yLo, Math.min(yHi, q.y))).toFixed(1)}`;
        pen = true;
      }
      pathCache.set(s.id, { points: s.points, d });
      out.set(s.id, d);
    }
    for (const id of pathCache.keys()) if (!seen.has(id)) pathCache.delete(id);
    return out;
  });
  /** Painted last: the highlighted series, then patterned ones over solid ones. */
  const ordered = computed(() => {
    const hl = p.highlight;
    return [...p.series].sort((a, b) => Number(a.id === hl) - Number(b.id === hl) || Number(!!a.pattern) - Number(!!b.pattern));
  });

  const stripMax = computed(() => (p.strip ? Math.max(1, ...p.strip.items.map((i) => i.v)) : 1));

  // ---- reading: the mouse, a touch pin, the keyboard ----
  const nearestStop = (px: number) => {
    const st = stops.peek();
    const xx = x.peek();
    if (!st.length) return null;
    let best = st[0]!;
    for (const s of st) if (Math.abs(xx(s) - px) < Math.abs(xx(best) - px)) best = s;
    return best;
  };
  const pinned = signal(false);
  let gesture: TouchGesture | null = null;
  const svgEl = signal<SVGSVGElement>();
  const readoutH = signal(0);
  const rectCache = createRectCache();
  effect(() => {
    void width.get();
    void p.height;
    void readoutH.get();
    rectCache.clear();
  });
  const local = (e: PointerEvent) => {
    const el = e.currentTarget as SVGElement;
    let r = rectCache.get(el);
    let px = e.clientX - r.left;
    let py = e.clientY - r.top;
    // If an external banner shifted the plot while hovered, an in-SVG pointer event can appear outside the stale rect.
    if (py < 0 || py > p.height || px < 0 || px > width.peek()) {
      rectCache.clear();
      r = rectCache.get(el);
      px = e.clientX - r.left;
      py = e.clientY - r.top;
    }
    return { px, py };
  };
  const touchy = (e: PointerEvent) => e.pointerType === 'touch' || e.pointerType === 'pen';
  const inPlot = (px: number) => px >= m.peek().left - 8 && px <= width.peek() - m.peek().right + 8;
  function apply(a: PinAction) {
    if (a.kind === 'clear') {
      spoken.set('');
      cursor.set(null);
      pinned.set(false);
    } else if (a.kind === 'set' && inPlot(a.px)) {
      const next = nearestStop(a.px);
      if (next !== null) {
        cursor.set(next);
        pinned.set(true);
        tapHint.set(false);
      }
    }
  }
  // A cursor cleared elsewhere drops the pin state and stale speech too.
  effect(() => {
    if (cursor.get() === null) {
      spoken.set('');
      pinned.set(false);
    }
  });
  effect(() => {
    if (!pinned.get()) return;
    const outside = (e: PointerEvent) => {
      const svg = svgEl.peek();
      if (svg && !svg.contains(e.target as Node)) apply({ kind: 'clear' });
    };
    const esc = (e: KeyboardEvent) => {
      if (canDismissChartOnEscape(e, true)) {
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
    spoken.set('');
    if (!touchy(e)) return;
    rectCache.clear();
    const { px, py } = local(e);
    const c = cursor.peek();
    gesture = new TouchGesture(px, py, pinned.peek() && c !== null ? x.peek()(c) : null);
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
    spoken.set('');
    const { px, py } = local(e);
    if (touchy(e)) {
      if (gesture) apply(gesture.move(px, py));
      return;
    }
    pinned.set(false);
    cursor.set(inPlot(px) && py >= 0 && py <= p.height ? nearestStop(px) : null);
  }
  function onpointerleave(e: PointerEvent) {
    rectCache.clear();
    if (!touchy(e) && !pinned.peek()) cursor.set(null);
  }
  const spoken = signal('');
  function onkeydown(e: KeyboardEvent) {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
    const st = stops.peek();
    if (!st.length) return;
    const c = cursor.peek();
    const i = c === null ? -1 : st.indexOf(c);
    let j = i;
    if (e.key === 'ArrowRight') j = i < 0 ? 0 : Math.min(st.length - 1, i + (e.shiftKey ? 10 : 1));
    else if (e.key === 'ArrowLeft') j = i < 0 ? st.length - 1 : Math.max(0, i - (e.shiftKey ? 10 : 1));
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = st.length - 1;
    else if (e.key === 'Escape') {
      if ((c === null && !pinned.peek()) || !canDismissChartOnEscape(e)) return;
      e.preventDefault();
      cursor.set(null);
      pinned.set(false);
      spoken.set('');
      return;
    } else return;
    e.preventDefault();
    const next = st[j]!;
    cursor.set(next);
    spoken.set(spokenAt(next, { series: p.series, table: p.table, xFormat: p.xFormat, yFormat: p.yFormat, xLabel: p.xLabel, speak: p.speak }));
  }
  // The reading hints of every chart (./hint): "← → to inspect" on keyboard focus (hidden on phones and touch-only
  // devices, matching LineChart), "Tap to pin a value" once on touch.
  const keyFocus = signal(false);
  const keyless = new MediaQuery(`${media.phone}, (hover: none) and (pointer: coarse)`, false);
  const tapHint = signal(typeof matchMedia !== 'undefined' && matchMedia('(pointer: coarse)').matches && claimOnce(TAP_HINT_KEY));
  const hint = computed(() =>
    chartHint({ reading: cursor.get() !== null, keyFocus: keyFocus.get() && !keyless.current, tap: tapHint.get() }),
  );
  function onfocus() {
    try {
      keyFocus.set(!!svgEl.peek()?.matches(':focus-visible'));
    } catch {
      keyFocus.set(false);
    }
  }

  /** The point of `s` at stop `v`, if it has one there. */
  const at = (s: XYSeries, v: number) => s.points.find((q) => q.x === v && q.y !== null);
  const markPos = (k: XYMark) => {
    const px = x.get()(k.x);
    const py = y.get()(clampY(k.y));
    const side = k.side ?? 'right';
    if (side === 'left') return { px, py, tx: px - 9, ty: py, anchor: 'end' as const };
    if (side === 'above') return { px, py, tx: px, ty: py - 12, anchor: 'middle' as const };
    if (side === 'below') return { px, py, tx: px, ty: py + 15, anchor: 'middle' as const };
    return { px, py, tx: px + 9, ty: py, anchor: 'start' as const };
  };

  const ref = (r: () => XYRef) => {
    const lo = () => Math.max(xDom.get()[0], yDom.get()[0]);
    const hi = () => Math.min(xDom.get()[1], yDom.get()[1]);
    return (
      <>
        {r().axis === 'diagonal' ? (
          <>
            <line class="ref" x1={x.get()(lo())} y1={y.get()(lo())} x2={x.get()(hi())} y2={y.get()(hi())} />
            {r().label ? (
              <text class="ref-label" x={x.get()(hi()) - 4} y={y.get()(hi()) + 14} text-anchor="end">
                {r().label}
              </text>
            ) : null}
          </>
        ) : r().axis === 'y' ? (
          <>
            <line class="ref" x1={m.get().left} x2={width.get() - m.get().right} y1={y.get()((r() as { value: number }).value)} y2={y.get()((r() as { value: number }).value)} />
            {r().label ? (
              <text class="ref-label" x={width.get() - m.get().right - 2} y={y.get()((r() as { value: number }).value) + ((r() as { below?: boolean }).below ? 15 : -5)} text-anchor="end">
                {r().label}
              </text>
            ) : null}
          </>
        ) : (
          <>
            <line class="ref" x1={x.get()((r() as { value: number }).value)} x2={x.get()((r() as { value: number }).value)} y1={m.get().top} y2={plotBottom.get()} />
            {r().label ? (
              <text class="ref-label" x={x.get()((r() as { value: number }).value) + 5} y={plotBottom.get() - 6}>
                {r().label}
              </text>
            ) : null}
          </>
        )}
      </>
    );
  };

  /** A point's whiskers and dot (the data layer: never reads the cursor). */
  const point = (s: () => XYSeries, q: () => XYPoint) => {
    const px = () => x.get()(q().x);
    const lo = () => y.get()(clampY(q().lo ?? q().y!));
    const hi = () => y.get()(clampY(q().hi ?? q().y!));
    return (
      <>
        {q().y !== null && (q().lo !== undefined || q().hi !== undefined) ? (
          <g class="whisker" class:context={receded(s())} style:stroke={receded(s()) ? undefined : inkOf(s())}>
            <line x1={px()} x2={px()} y1={lo()} y2={hi()} />
            <line x1={px() - 4} x2={px() + 4} y1={lo()} y2={lo()} />
            <line x1={px() - 4} x2={px() + 4} y1={hi()} y2={hi()} />
          </g>
        ) : null}
        {q().y !== null && (s().dots || q().lo !== undefined || q().r !== undefined || q().hollow) ? (
          <circle
            class="pt"
            class:hollow={q().hollow}
            cx={px()}
            cy={y.get()(clampY(q().y!))}
            r={q().r ?? 3.5}
            style:fill={q().hollow ? undefined : receded(s()) ? 'var(--chart-context)' : inkOf(s())}
            style:stroke={q().hollow ? (receded(s()) ? 'var(--chart-context)' : inkOf(s())) : undefined}
          />
        ) : null}
      </>
    );
  };

  const stripBase = () => plotBottom.get() + 12 + STRIP_H;

  return (
    <div class="xy">
      <div class="readout num" bind:clientHeight={readoutH} aria-hidden="true">
        {p.readout(() => cursor.get(), () => hint.get())}
      </div>
      <p class="sr-only" aria-live="polite">
        {spoken.get()}
      </p>
      <div class="sr-only">
        <table>
          <caption>{p.table.caption}</caption>
          <thead>
            <tr>
              <Index each={p.table.head}>{(h) => <th scope="col">{h()}</th>}</Index>
            </tr>
          </thead>
          <tbody>
            <Index each={p.table.rows}>
              {(r) => (
                <tr>
                  <Index each={r()}>{(c, k) => <>{k === 0 ? <th scope="row">{c()}</th> : <td>{c()}</td>}</>}</Index>
                </tr>
              )}
            </Index>
          </tbody>
        </table>
      </div>
      <div class="plot" bind:clientWidth={width} style:height={`${p.height}px`}>
        {width.get() > 0 ? (
          <svg
            ref={svgEl}
            width={width.get()}
            height={p.height}
            role="group"
            aria-roledescription="chart"
            aria-label={`${p.label}. Focus and use the arrow keys to read it point by point.`}
            tabindex={0}
            onpointerenter={onpointerenter}
            onpointermove={onpointermove}
            onpointerleave={onpointerleave}
            onpointerdown={onpointerdown}
            onpointerup={onpointerup}
            onpointercancel={onpointercancel}
            onkeydown={onkeydown}
            onfocus={onfocus}
            onblur={() => {
              spoken.set('');
              keyFocus.set(false);
              if (!pinned.peek()) cursor.set(null);
            }}
          >
            {p.band ? (
              <rect
                class="band"
                x={m.get().left}
                width={Math.max(0, width.get() - m.get().right - m.get().left)}
                y={y.get()(clampY(p.band.hi))}
                height={Math.max(1, y.get()(clampY(p.band.lo)) - y.get()(clampY(p.band.hi)))}
              />
            ) : null}
            <g class="grid">
              <Index each={p.yTicks}>{(t) => <line x1={m.get().left} x2={width.get() - m.get().right} y1={y.get()(t())} y2={y.get()(t())} />}</Index>
            </g>
            <line class="axis" x1={m.get().left} x2={width.get() - m.get().right} y1={plotBottom.get()} y2={plotBottom.get()} />
            <g class="ticks">
              <Index each={p.yTicks}>
                {(t) => (
                  <text x={m.get().left - 8} y={y.get()(t())} dy="0.32em" text-anchor="end">
                    {p.yFormat(t())}
                  </text>
                )}
              </Index>
              <Index each={p.xTicks}>
                {(t) => (
                  <text x={x.get()(t())} y={p.height - (p.xLabel ? 20 : 4)} text-anchor="middle">
                    {p.xFormat(t())}
                  </text>
                )}
              </Index>
              {p.xLabel ? (
                <text class="axis-name" x={(m.get().left + width.get() - m.get().right) / 2} y={p.height - 2} text-anchor="middle">
                  {p.xLabel}
                </text>
              ) : null}
              {p.yLabel ? (
                <text class="axis-name" x="0" y="8" dy="0.32em" text-anchor="start">
                  {p.yLabel}
                </text>
              ) : null}
            </g>
            {p.band ? (
              /* Inside the band at its lower right: clear of the first point's whiskers, which start at the left. */
              <text class="band-label" x={width.get() - m.get().right - 6} y={y.get()(clampY(p.band.lo)) - 6} text-anchor="end">
                {p.band.label}
              </text>
            ) : null}
            <Index each={p.refs}>{(r) => ref(r)}</Index>
            {p.strip ? (
              <>
                {/* What each span stands on, on the plot's x scale: bars from the strip's base, the most at its top. */}
                <line class="strip-base" x1={m.get().left} x2={width.get() - m.get().right} y1={stripBase()} y2={stripBase()} />
                <text class="tick-t" x={m.get().left - 8} y={plotBottom.get() + 12} dy="0.7em" text-anchor="end">
                  {stripMax.get()}
                </text>
                <Index each={p.strip.items}>
                  {(it) => {
                    const h = () => (it().v > 0 ? Math.max(1, (it().v / stripMax.get()) * STRIP_H) : 0);
                    const gap = 1;
                    return (
                      <rect
                        class="strip-bar"
                        class:on={cursor.get() !== null && cursor.get()! >= it().x0 && cursor.get()! < it().x1 + 1e-9}
                        x={x.get()(it().x0) + gap}
                        width={Math.max(0, x.get()(it().x1) - x.get()(it().x0) - 2 * gap)}
                        y={stripBase() - h()}
                        height={h()}
                        rx="1"
                      />
                    );
                  }}
                </Index>
              </>
            ) : null}
            <g class="data">
              <For each={ordered.get()} key={(s) => s.id}>
                {(s) => (
                  <>
                    {s().line !== false ? (
                      <path
                        class="line"
                        class:thin={s().thin}
                        class:context={receded(s())}
                        class:hl={p.highlight === s().id}
                        class:dotted={s().pattern === 'dot'}
                        d={paths.get().get(s().id)}
                        style:stroke={receded(s()) ? undefined : inkOf(s())}
                        style:stroke-dasharray={s().pattern ? DASH[s().pattern!] : undefined}
                      />
                    ) : null}
                    <Index each={s().points}>{(q) => point(s, q)}</Index>
                  </>
                )}
              </For>
            </g>
            <Index each={p.marks}>
              {(k) => {
                const q = () => markPos(k());
                return (
                  <g class="mark">
                    <circle class="ring" cx={q().px} cy={q().py} r="5" />
                    <text x={q().tx} y={q().ty} dy="0.32em" text-anchor={q().anchor}>
                      {k().label}
                    </text>
                  </g>
                );
              }}
            </Index>
            {cursor.get() !== null ? (
              <>
                <line class="cursor" class:pinned={pinned.get()} x1={x.get()(cursor.get()!)} x2={x.get()(cursor.get()!)} y1={m.get().top} y2={plotBottom.get()} />
                <For each={p.series} key={(s) => s.id}>
                  {(s) => {
                    const q = () => {
                      const c = cursor.get();
                      return c === null ? undefined : at(s(), c);
                    };
                    return <>{q() && q()!.y !== null ? <circle class="dot" cx={x.get()(q()!.x)} cy={y.get()(clampY(q()!.y!))} r="4" style:fill={inkOf(s())} /> : null}</>;
                  }}
                </For>
              </>
            ) : null}
          </svg>
        ) : null}
      </div>
    </div>
  );
}

css`
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
    /* A press and drag over the plot reads values; it never selects the tick labels (runs F18). */
    user-select: none;
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
`;
