/**
 * Vertical bars for a histogram (`edges` + `counts`): each bin is one evenly spaced category named by `binLabel`, and
 * a bar's height is its count, so unequal bins (finer near a threshold, an open-ended last one) read at a glance
 * without a density axis. The readout row above the plot is the legend, as on the line charts: the total at rest, the
 * bin under the cursor on hover. Touch: a tap or a sideways scrub pins a bin until it is tapped again, a tap lands
 * outside or Escape (./pin). Keyboard: arrow keys, Home and End.
 */
import { scaleLinear } from 'd3-scale';
import { computed, css, defaults, effect, Index, signal } from 'fw';
import { ticks as niceTicks } from './downsample';
import { binAt, categoryBars, edgeX } from './histogram';
import { TouchGesture, type PinAction } from './pin';
import { createRectCache } from './rectCache';

export interface HistogramProps {
  edges: number[];
  counts: number[];
  height?: number;
  color?: string;
  /** Fill of the hovered bar. Without it, hovering dims the other bars instead. */
  hoverColor?: string;
  /** Label for bin i (defaults to "a–b"). */
  binLabel?: (lo: number, hi: number, i: number) => string;
  /** A shorter axis label for bin i (e.g. without the unit the panel names); defaults to the bin label. */
  tickLabel?: (lo: number, hi: number, i: number) => string;
  label: string;
  unit?: string;
  /** A dashed vertical reference line at a value (e.g. a threshold on a bin edge), labelled on its lower side. */
  marker?: { x: number; label: string };
  /** What the bar height counts, written above the y axis. */
  yLabel?: string;
}

export default function Histogram(props: HistogramProps) {
  const p = defaults(props, { height: 160, color: 'var(--chart-2)', unit: 'boxes' });

  const width = signal(0);
  const hovered = signal<number | null>(null);

  const m = computed(() => ({ top: p.yLabel ? 24 : 8, right: 4, bottom: 20, left: 36 }));
  const max = computed(() => Math.max(1e-9, ...p.counts));
  const y = computed(() =>
    scaleLinear()
      .domain([0, max.get()])
      .nice(4)
      .range([p.height - m.get().bottom, m.get().top]),
  );
  const yt = computed(() => niceTicks(0, y.get().domain()[1] ?? max.get(), 3));
  const bars = computed(() => categoryBars(p.counts.length, m.get().left, width.get() - m.get().right));
  const bw = computed(() => bars.get()[0]?.w ?? 0);
  const name = (i: number) => {
    const edges = p.edges;
    return p.binLabel ? p.binLabel(edges[i] ?? 0, edges[i + 1] ?? 0, i) : `${edges[i]}–${edges[i + 1]}`;
  };
  const tickName = (i: number) => {
    const edges = p.edges;
    return p.tickLabel ? p.tickLabel(edges[i] ?? 0, edges[i + 1] ?? 0, i) : name(i);
  };
  const total = computed(() => p.counts.reduce((a, b) => a + b, 0));
  const firstBin = computed(() => Math.max(0, p.counts.findIndex((c) => c > 0)));
  const shareOf = (c: number) => {
    const t = total.get();
    return t ? ((c / t) * 100).toFixed(1) : '0';
  };
  const readout = computed(() => {
    const h = hovered.get();
    return h === null ? '' : `${name(h)}: ${(p.counts[h] ?? 0).toLocaleString('en-US')} ${p.unit}, ${shareOf(p.counts[h] ?? 0)}%`;
  });
  // Every category is named while each has 34 px; past that, every k-th.
  const labelEvery = computed(() => (bw.get() > 34 ? 1 : Math.ceil(34 / Math.max(bw.get(), 1))));

  // The bars' geometry, once per change of the data or the plot's size: hovering only toggles a bar's dim or fill.
  const geo = computed(() => {
    const { height } = p;
    const mm = m.get();
    const yy = y.get();
    const bs = bars.get();
    const every = labelEvery.get();
    return p.counts.map((c, i) => {
      const h = Math.max(c > 0 ? 2 : 0, height - mm.bottom - yy(c));
      const b = bs[i] ?? { x: mm.left, w: 0 };
      const gap = Math.min(2, b.w / 6);
      const r = Math.min(2, Math.max(0, (b.w - 2 * gap) / 2));
      return {
        d: `M${b.x + gap},${height - mm.bottom} v${-Math.max(0, h - r)} q0,-${r} ${r},-${r} h${Math.max(0, b.w - 2 * gap - 2 * r)} q${r},0 ${r},${r} v${Math.max(0, h - r)} z`,
        tick: i % every === 0 ? { x: b.x + b.w / 2, text: tickName(i) } : null,
      };
    });
  });

  // Pointer readout: a mouse reads the bin under it; touch pins a bin (./pin) until dismissed.
  const pinned = signal(false);
  let gesture: TouchGesture | null = null;
  const svgEl = signal<SVGSVGElement>();
  const rectCache = createRectCache();
  effect(() => {
    void width.get();
    void p.height;
    rectCache.clear();
  });
  const local = (e: PointerEvent) => {
    const rect = rectCache.get(e.currentTarget as SVGElement);
    return { px: e.clientX - rect.left, py: e.clientY - rect.top };
  };
  const touchy = (e: PointerEvent) => e.pointerType === 'touch' || e.pointerType === 'pen';
  function apply(a: PinAction) {
    if (a.kind === 'clear') {
      hovered.set(null);
      pinned.set(false);
    } else if (a.kind === 'set') {
      const i = binAt(bars.peek(), a.px);
      if (i !== null) {
        hovered.set(i);
        pinned.set(true);
      }
    }
  }
  effect(() => {
    if (!pinned.get()) return;
    const outside = (e: PointerEvent) => {
      const svg = svgEl.peek();
      if (svg && !svg.contains(e.target as Node)) apply({ kind: 'clear' });
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
  function onpointerenter() {
    rectCache.clear();
  }
  function onpointerdown(e: PointerEvent) {
    if (!touchy(e)) return;
    rectCache.clear();
    const { px, py } = local(e);
    const h = hovered.peek();
    const at = pinned.peek() && h !== null ? bars.peek()[h] : undefined;
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
    pinned.set(false);
    const mm = m.peek();
    hovered.set(py >= mm.top && py <= p.height - mm.bottom ? binAt(bars.peek(), px) : null);
  }
  function onpointerleave(e: PointerEvent) {
    rectCache.clear();
    if (!touchy(e) && !pinned.peek()) hovered.set(null);
  }
  function onkeydown(e: KeyboardEvent) {
    const n = p.counts.length;
    if (!n) return;
    const i = hovered.peek() ?? firstBin.peek();
    let j = i;
    if (e.key === 'ArrowRight') j = Math.min(n - 1, i + 1);
    else if (e.key === 'ArrowLeft') j = Math.max(0, i - 1);
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = n - 1;
    else if (e.key === 'Escape') {
      hovered.set(null);
      pinned.set(false);
      return;
    } else return;
    e.preventDefault();
    hovered.set(j);
  }
  // One notation per axis: all ticks in thousands once the axis reaches 1,000; zero is always plain "0".
  const tickFmt = (v: number) => (v !== 0 && (y.get().domain()[1] ?? 0) >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(+v.toFixed(2)));
  const markerX = () => Math.round(edgeX(p.edges, bars.get(), p.marker!.x)) + 0.5;

  return (
    <>
      {/* The readout: the legend row of the chart frame, in one stable place (no floating tooltip). */}
      <p class="readout num" class:at={hovered.get() !== null} aria-hidden="true">
        {hovered.get() !== null ? (
          <>
            <span class="t">{name(hovered.get()!)}</span>
            <span class="v">{`${(p.counts[hovered.get()!] ?? 0).toLocaleString('en-US')} ${p.unit}`}</span>
            <span class="p">{`${shareOf(p.counts[hovered.get()!] ?? 0)}%`}</span>
          </>
        ) : (
          <span class="v">{`${total.get().toLocaleString('en-US')} ${p.unit}`}</span>
        )}
      </p>
      <div class="hist" bind:clientWidth={width} style:height={`${p.height}px`}>
        {width.get() > 0 ? (
          <svg
            ref={svgEl}
            width={width.get()}
            height={p.height}
            role="group"
            aria-roledescription="chart"
            aria-label={`${p.label}. Focus and use the arrow keys to read each bin.`}
            tabindex={0}
            onkeydown={onkeydown}
            onpointerenter={onpointerenter}
            onpointermove={onpointermove}
            onpointerleave={onpointerleave}
            onpointerdown={onpointerdown}
            onpointerup={onpointerup}
            onpointercancel={onpointercancel}
            onfocus={() => hovered.set((h) => h ?? firstBin.peek())}
            onblur={() => {
              if (!pinned.peek()) hovered.set(null);
            }}
          >
            {p.yLabel ? (
              <text class="y-label" x="0" y="8" dy="0.32em" text-anchor="start">
                {p.yLabel}
              </text>
            ) : null}
            <Index each={yt.get()}>
              {(t) => (
                <>
                  <line class="grid" x1={m.get().left} x2={width.get() - m.get().right} y1={y.get()(t())} y2={y.get()(t())} />
                  <text class="tick" x={m.get().left - 8} y={y.get()(t())} dy="0.32em" text-anchor="end">
                    {tickFmt(t())}
                  </text>
                </>
              )}
            </Index>
            <Index each={geo.get()}>
              {(g, i) => (
                <>
                  <path
                    class="bar"
                    class:dim={!p.hoverColor && hovered.get() !== null && hovered.get() !== i}
                    style:fill={p.hoverColor && hovered.get() === i ? p.hoverColor : p.color}
                    d={g().d}
                  />
                  {g().tick ? (
                    <text class="tick" x={g().tick!.x} y={p.height - 6} text-anchor="middle">
                      {g().tick!.text}
                    </text>
                  ) : null}
                </>
              )}
            </Index>
            {p.marker ? (
              <>
                <line class="marker" x1={markerX()} x2={markerX()} y1={m.get().top} y2={p.height - m.get().bottom} />
                {/* The label sits on the threshold's lower side: "tiny" names what lies left of the line. */}
                <text class="marker-label" x={markerX() - 4} y={m.get().top + 8} text-anchor="end">
                  {p.marker.label}
                </text>
              </>
            ) : null}
            <line class="axis" x1={m.get().left} x2={width.get() - m.get().right} y1={p.height - m.get().bottom} y2={p.height - m.get().bottom} />
          </svg>
        ) : null}
      </div>
      {/* Read by assistive tech whether or not the plot has laid out yet. */}
      <p class="sr-only" aria-live="polite">
        {readout.get()}
      </p>
      <div class="sr-only">
        <table>
          <caption>{p.label}</caption>
          <thead>
            <tr>
              <th scope="col">Bin</th>
              <th scope="col">{p.unit}</th>
              <th scope="col">Share</th>
            </tr>
          </thead>
          <tbody>
            <Index each={p.counts}>
              {(c, i) => (
                <tr>
                  <th scope="row">{name(i)}</th>
                  <td>{c().toLocaleString('en-US')}</td>
                  <td>{`${shareOf(c())}%`}</td>
                </tr>
              )}
            </Index>
          </tbody>
        </table>
      </div>
    </>
  );
}

css`
  .hist {
    position: relative;
    width: 100%;
  }
  svg {
    display: block;
    overflow: visible;
    border-radius: var(--radius-sm);
    touch-action: pan-y;
    /* A press and drag over the plot reads values; it never selects the tick labels (runs F18). */
    user-select: none;
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
  /* Focus: the global :focus-visible ring (base.css), never a halo of its own (DESIGN_DIRECTION §2.9). */
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
    font-size: var(--type-xs-size); /* small: chart tick */
    font-variant-numeric: tabular-nums;
  }
  .marker {
    stroke: var(--text-secondary);
    stroke-dasharray: 3 3;
    shape-rendering: crispEdges;
  }
  .marker-label {
    fill: var(--text-secondary);
    font-size: var(--type-sm-size);
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
`;
