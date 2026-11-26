/**
 * Confusion matrix: rows are true classes, columns predicted, plus a "background" row/column for misses and false
 * positives. Cells are row-normalised and shaded on one sequential luminance ramp in ink (the diagonal is told apart
 * by position, not hue), every cell with objects labelled in whole percentages ("<1" for small shares,
 * ./confusionInk) where the number fits, and marked with a dot where it does not; counts on hover or focus. An
 * empty cell is the panel with a hairline grid; the ramp's floor is stepped (under 1 %, 1 %, 5 %, then even from
 * 10 %, ./confusionInk), so a 2 % and an 8 % confusion never look alike. `reveal(r, c)` (the `expose` prop's api) lets a
 * summary elsewhere point at one cell: it scrolls the cell into view and makes it the active cell, whose counts the
 * readout states. A matrix that scrolls sideways (phones) pages through whole columns: its cells are sized so a page
 * fills the width exactly and the scroll snaps to column starts, so no column is half in view and "Columns 1–6 of 9"
 * is what is seen (./cmOverview pageFit); the rotated label row is as tall as the longest label in view. Its navigation
 * is one 44 px row above the grid: a small whole-matrix overview (every cell as a square, a band over the columns in
 * view; a tap brings those columns into view) beside the column-range control and a jump to the background column
 * (the misses). The readout, a tap hint at rest, sits under the grid. Cells are picked in the full matrix, large
 * enough to hit.
 */
import ChevronLeft from '$icons/chevron-left';
import ChevronRight from '$icons/chevron-right';
import ChevronsLeft from '$icons/chevrons-left';
import ChevronsRight from '$icons/chevrons-right';
import { computed, createUniqueId, css, defaults, effect, Index, onCleanup, signal, tick, untrack } from 'fw';
import { classColor } from '$lib/design/tokens';
import { prefs } from '$lib/prefs';
import { overviewGeometry, overviewHit, overviewWindow, pageFit, pageRange } from './cmOverview';
import { FLOOR, labelHigh, rampShare, sharePct } from './confusionInk';

/** What a ConfusionMatrix offers its owner (port of the Svelte component's exported `reveal`), through `expose`. */
export interface ConfusionMatrixApi {
  /**
   * Shows the cell at row `r`, column `c` (indices into `classes`; `classes.length` is background) and makes it the
   * active cell: scrolled into view past the pinned row labels, focused, marked and stated by the readout.
   */
  reveal(r: number, c: number, block?: ScrollLogicalPosition): void;
}

export interface ConfusionMatrixProps {
  classes: string[];
  /** (n+1)×(n+1) counts; the last row/column is background. */
  matrix: number[][];
  label?: string;
  /**
   * Minimum cell size in px. Wide matrices then scroll sideways with the row labels pinned, instead of squeezing
   * cells; on touch screens cells are at least 44 px.
   */
  minCell?: number;
  /** Largest cell in px (fixed matrices), so a wide panel does not turn a small matrix into a wall. */
  maxCell?: number;
  /** Show a labelled 0–100% colour scale under the matrix. */
  scale?: boolean;
  /**
   * What a cell's number says: its share of the true class (`share`, the default) or the objects counted (`count`,
   * "1.2k" from a thousand). The shade is the share either way.
   */
  values?: 'share' | 'count';
  /**
   * The palette index of each entry of `classes` (defaults to its position), for reduced matrices whose rows are a
   * subset of the model's classes; a negative index draws a neutral dot (e.g. "other classes").
   */
  classIds?: number[];
  /** Receives the matrix's api (`reveal`) once, at creation (port of `bind:this={view}` + `view.reveal(...)`). */
  expose?: (api: ConfusionMatrixApi) => void;
}

type At = { r: number; c: number };

/** A cell's count in at most four characters: 7, 526, 1.2k, 12k. */
function countText(n: number): string {
  if (n < 1000) return String(n);
  return n < 9950 ? `${(n / 1000).toFixed(1)}k` : `${Math.round(n / 1000)}k`;
}

/** Row labels (96 px) plus the grid gap: where the columns start. */
const HEAD = 98;

export default function ConfusionMatrix(props: ConfusionMatrixProps) {
  const p = defaults(props, { label: 'Confusion matrix', maxCell: 72, scale: false });
  // Cells take their ink share from the ramp; labels are pure white or black, switched near the ramp's luminance
  // crossover (see ./confusionInk).
  const theme = computed(() => prefs.resolvedTheme);

  const uid = createUniqueId();
  const hovered = signal<At | null>(null);
  /** A cell chosen by keyboard or tap keeps the readout when the pointer leaves. */
  const pinned = signal<At | null>(null);
  /** Roving tabindex: one cell is in the tab order; arrow keys move it. */
  const focus = signal<At>({ r: 0, c: 0 });
  let root!: HTMLDivElement;
  /** A scrolling matrix fades its right edge while there are more columns to scroll to. */
  const moreRight = signal(false);
  /** Fully visible prediction columns (1-based) while the matrix scrolls sideways; null when every column fits. */
  const range = signal<{ first: number; last: number } | null>(null);
  const fixed = computed(() => p.minCell !== undefined);
  /**
   * Pages of whole columns while the matrix scrolls sideways (./cmOverview pageFit): the cell size that fits exactly
   * `cols` columns beside the row labels, so no column is ever half in view and the counter names what is seen. The
   * scroll snaps to column starts. Null when every column fits.
   */
  const fit = signal<{ cols: number; cell: number } | null>(null);
  /** Height of the longest column label in view, px: the rotated label row is sized to it, not to one off-screen. */
  const labelH = signal<number | null>(null);

  const names = computed(() => [...p.classes, 'background']);
  const n = computed(() => names.get().length);
  // Keep the roving tab stop and active readout inside the grid if the class list shrinks.
  effect(() => {
    const max = Math.max(0, n.get() - 1);
    untrack(() => {
      const f = focus.peek();
      if (f.r > max || f.c > max) focus.set({ r: Math.min(f.r, max), c: Math.min(f.c, max) });
      const h = hovered.peek();
      if (h && (h.r > max || h.c > max)) hovered.set(null);
      const pin = pinned.peek();
      if (pin && (pin.r > max || pin.c > max)) pinned.set(null);
    });
  });
  const rowSums = computed(() => p.matrix.map((row) => row.reduce((a, b) => a + b, 0)));
  const norm = (r: number, c: number) => {
    const s = rowSums.get()[r] ?? 0;
    return s ? (p.matrix[r]?.[c] ?? 0) / s : 0;
  };
  /** Every cell's share, ink and name: once per change of the matrix or the theme, never on hover. */
  const cells = computed(() => {
    const t = theme.get();
    const nm = names.get();
    const sums = rowSums.get();
    return nm.map((rname, r) =>
      nm.map((cname, c) => {
        const count = p.matrix[r]?.[c] ?? 0;
        const v = norm(r, c);
        const share = sharePct(v);
        return {
          zero: !count,
          v,
          t: p.values === 'count' ? countText(count) : share,
          hi: labelHigh(v, t),
          s: rampShare(v, t),
          label: `${rname} predicted as ${cname}: ${count.toLocaleString('en-US')} of ${(sums[r] ?? 0).toLocaleString('en-US')}, ${share || '0'}%`,
        };
      }),
    );
  });

  function measure() {
    const min = parseFloat(getComputedStyle(root).getPropertyValue('--min-cell')) || p.minCell || 22;
    const next = pageFit(root.clientWidth, HEAD, min, 2, n.peek());
    const cur = fit.peek();
    if (next?.cols !== cur?.cols || next?.cell !== cur?.cell) fit.set(next);
  }
  function measureScroll() {
    // A deferred call (the next tick, an observer, a rAF) can land after the matrix has gone.
    if (!root || !root.isConnected) return;
    const f = fit.peek();
    if (!f) {
      moreRight.set(root.scrollLeft + root.clientWidth < root.scrollWidth - 1);
      range.set(null);
      labelH.set(null);
      return;
    }
    const count = n.peek();
    moreRight.set(false);
    const step = f.cell + 2;
    const next = pageRange(root.scrollLeft, step, f.cols, count);
    const cur = range.peek();
    if (!cur || cur.first !== next.first || cur.last !== next.last) range.set(next);
    // Partly visible columns count while a page slides in, so a label coming into view is never cut at the top.
    const lo = Math.max(0, Math.floor(root.scrollLeft / step));
    const hi = Math.min(count - 1, Math.ceil((root.scrollLeft + root.clientWidth - HEAD) / step) - 1);
    const lbls = root.querySelectorAll<HTMLElement>('.colhead .lbl');
    let maxH = 0;
    for (let i = lo; i <= hi && i < lbls.length; i++) {
      const h = Math.min(lbls[i]!.scrollHeight, 72);
      if (h > maxH) maxH = h;
    }
    labelH.set(maxH);
  }
  let scrollRaf = 0;
  onCleanup(() => {
    if (scrollRaf) cancelAnimationFrame(scrollRaf);
  });
  function onscroll() {
    if (scrollRaf) return;
    scrollRaf = requestAnimationFrame(() => {
      scrollRaf = 0;
      measureScroll();
    });
  }
  const behavior = (): ScrollBehavior => (matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth');
  /** Pages the columns: the next (or previous) page of whole columns, sideways only. */
  function page(dir: 1 | -1) {
    const f = fit.peek();
    const r = range.peek();
    if (!f || !r) return;
    const step = f.cell + 2;
    root.scrollTo({ left: Math.max(0, (r.first - 1 + dir * f.cols) * step), behavior: behavior() });
  }
  /** Scrolls to the last columns (the background column, the misses) or back to the first. */
  function toEdge(dir: 1 | -1) {
    root.scrollTo({ left: dir > 0 ? root.scrollWidth : 0, behavior: behavior() });
  }

  function reveal(r: number, c: number, block: ScrollLogicalPosition = 'center') {
    const cell = root?.querySelector<HTMLElement>(`[data-r="${r}"][data-c="${c}"]`);
    if (!cell) return;
    focus.set({ r, c });
    cell.focus({ preventScroll: true });
    const at = { r, c };
    hovered.set(at);
    pinned.set(at);
    // scroll-padding keeps the cell clear of the sticky row labels and the faded right edge.
    cell.scrollIntoView({ block, inline: 'nearest', behavior: behavior() });
  }
  props.expose?.({ reveal });

  effect(() => {
    if (!fixed.get()) return;
    void n.get();
    measure();
    const ro = new ResizeObserver(() => measure());
    ro.observe(root);
    return () => ro.disconnect();
  });
  // A new page size lays the grid out again; the range and the label row follow once it has.
  effect(() => {
    if (!fixed.get()) return;
    void fit.get();
    void n.get();
    let live = true;
    void tick().then(() => live && measureScroll());
    return () => (live = false);
  });
  // A label's length changes when the font settles: the label row is measured again, so it never cuts a name short.
  effect(() => {
    if (!fixed.get()) return;
    void n.get();
    const ro = new ResizeObserver(() => measureScroll());
    for (const l of root.querySelectorAll('.colhead .lbl')) ro.observe(l);
    return () => ro.disconnect();
  });

  function cellCoords(el: EventTarget | null): { r: number; c: number } | null {
    const node = el instanceof Element ? el : el instanceof Node ? el.parentElement : null;
    const cell = node?.closest<HTMLElement>('.cell[data-r][data-c]');
    if (!cell || !root?.contains(cell)) return null;
    const r = Number(cell.dataset.r);
    const c = Number(cell.dataset.c);
    return Number.isInteger(r) && Number.isInteger(c) ? { r, c } : null;
  }
  function onpointerover(e: PointerEvent) {
    const to = cellCoords(e.target);
    if (!to) return;
    const from = cellCoords(e.relatedTarget);
    if (from && from.r === to.r && from.c === to.c) return;
    hovered.set(to);
  }
  function onpointerout(e: PointerEvent) {
    if (!cellCoords(e.target)) return;
    if (!cellCoords(e.relatedTarget)) hovered.set(pinned.peek());
  }
  function onfocusin(e: FocusEvent) {
    const at = cellCoords(e.target);
    if (!at) return;
    focus.set(at);
    hovered.set(at);
    pinned.set(at);
  }
  function onfocusout(e: FocusEvent) {
    if (!cellCoords(e.target)) return;
    if (!cellCoords(e.relatedTarget)) {
      hovered.set(null);
      pinned.set(null);
    }
  }

  function onkeydown(e: KeyboardEvent) {
    if (e.defaultPrevented || e.altKey) return;
    const max = n.peek() - 1;
    let { r, c } = focus.peek();
    const mod = e.ctrlKey || e.metaKey;
    if (e.key === 'Escape') {
      if (mod || (pinned.peek() === null && hovered.peek() === null)) return;
      e.preventDefault();
      pinned.set(null);
      hovered.set(null);
      return;
    }
    if (!mod && e.key === 'ArrowRight') c = Math.min(max, c + 1);
    else if (!mod && e.key === 'ArrowLeft') c = Math.max(0, c - 1);
    else if (!mod && e.key === 'ArrowDown') r = Math.min(max, r + 1);
    else if (!mod && e.key === 'ArrowUp') r = Math.max(0, r - 1);
    else if (e.key === 'Home') {
      c = 0;
      if (mod) r = 0;
    } else if (e.key === 'End') {
      c = max;
      if (mod) r = max;
    } else return;
    e.preventDefault();
    focus.set({ r, c });
    root.querySelector<HTMLElement>(`[data-r="${r}"][data-c="${c}"]`)?.focus();
  }
  /** Swatch colour of row/column `i`: the class palette hue, neutral for background and folded rows. */
  const dotColor = (i: number) => {
    const id = i < p.classes.length ? (p.classIds?.[i] ?? i) : -1;
    return id >= 0 ? classColor(id) : 'var(--text-tertiary)';
  };

  // Small: a thumbnail at the start of the pager row (one 44 px row), not a second figure.
  const geo = computed(() => overviewGeometry(n.get(), 44, 2));
  const band = computed(() => {
    const r = range.get();
    return r ? overviewWindow(r.first, r.last, geo.get()) : null;
  });
  /**
   * A tap on the overview centres the tapped column in the full matrix (sideways only, so the page does not jump).
   * Coarse on purpose: the map's squares are far below a touch target, so a near miss only moves the view a column.
   */
  function onmap(e: MouseEvent) {
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
    const hit = overviewHit(e.clientX - rect.left, e.clientY - rect.top, geo.peek(), n.peek());
    const cell = hit && root?.querySelector<HTMLElement>(`[data-r="0"][data-c="${hit.c}"]`);
    if (!cell) return;
    const box = root.getBoundingClientRect();
    const at = cell.getBoundingClientRect();
    root.scrollTo({ left: root.scrollLeft + at.left - box.left - (box.width - at.width) / 2, behavior: behavior() });
  }

  const readout = () => (
    <p class="readout" class:faint={!hovered.get()} class:below={!!range.get()} id={`${uid}-readout`}>
      {hovered.get() ? (
        <>
          <b>{names.get()[hovered.get()!.r]}</b> predicted as <b>{names.get()[hovered.get()!.c]}</b>: <span class="num">{(p.matrix[hovered.get()!.r]?.[hovered.get()!.c] ?? 0).toLocaleString('en-US')}</span> of <span class="num">{(rowSums.get()[hovered.get()!.r] ?? 0).toLocaleString('en-US')}</span> (<span class="num">{`${sharePct(norm(hovered.get()!.r, hovered.get()!.c)) || '0'}%`}</span>)
        </>
      ) : range.get() ? (
        'Tap a cell for its counts.'
      ) : (
        'Hover, tap or arrow through the cells for counts.'
      )}
    </p>
  );

  const cellAt = (r: number, c: number) => {
    const cell = () => cells.get()[r]?.[c];
    const on = () => {
      const h = hovered.get();
      return !!h && (h.r === r || h.c === c);
    };
    return (
      <div
        class="cell"
        class:zero={!!cell()?.zero}
        class:hl={on()}
        class:pin={pinned.get() !== null && pinned.get()!.r === r && pinned.get()!.c === c}
        role="gridcell"
        tabindex={focus.get().r === r && focus.get().c === c ? 0 : -1}
        data-r={r}
        data-c={c}
        aria-label={cell()?.label}
        style={{ '--s': cell()?.s }}
      >
        {(cell()?.v ?? 0) > 0 ? (
          <>
            {/* The share where it fits; else a dot, so no shaded cell carries its value by shade alone. Either way the
                cell's name and the readout (hover, focus) state the counts. */}
            <span class="v" class:hi={cell()!.hi} class:long={cell()!.t.length > 2} class:longer={cell()!.t.length > 3} aria-hidden="true">
              {cell()!.t}
            </span>
            <i class="mark" class:hi={cell()!.hi} aria-hidden="true"></i>
          </>
        ) : null}
      </div>
    );
  };

  return (
    <>
      {range.get() && band.get() ? (
        /* Above the matrix, so it is in reach before the tall grid, and one 44 px row so the first cell starts soon:
           the whole matrix as a thumbnail (a pointer shortcut; the grid stays the keyboard and screen-reader path), the
           columns in view with a way to page through them, and a direct jump to the background column (the misses,
           last). The active cell's figures are under the grid. */
        <div class="ov">
          <svg class="ov-map" width={geo.get().size} height={geo.get().size} viewBox={`0 0 ${geo.get().size} ${geo.get().size}`} aria-hidden="true" onclick={onmap}>
            <Index each={names.get()}>
              {(_, r) => (
                <Index each={names.get()}>
                  {(_c, c) => (
                    <rect
                      class="ov-cell"
                      class:zero={!p.matrix[r]?.[c]}
                      x={c * (geo.get().cell + geo.get().gap)}
                      y={r * (geo.get().cell + geo.get().gap)}
                      width={geo.get().cell}
                      height={geo.get().cell}
                      rx="1"
                      style={{ '--s': cells.get()[r]?.[c]?.s }}
                    />
                  )}
                </Index>
              )}
            </Index>
            <rect class="ov-band" x={band.get()!.x - 1.5} y="-1.5" width={band.get()!.w + 3} height={geo.get().size + 3} rx="2" />
            {pinned.get() ? (
              <rect
                class="ov-pin"
                x={pinned.get()!.c * (geo.get().cell + geo.get().gap) - 1}
                y={pinned.get()!.r * (geo.get().cell + geo.get().gap) - 1}
                width={geo.get().cell + 2}
                height={geo.get().cell + 2}
                rx="2"
              />
            ) : null}
          </svg>
          <div class="ov-side">
            <div class="range num" role="group" aria-label="Columns in view">
              <button type="button" class="pg" aria-label="Previous columns" disabled={range.get()!.first <= 1} onclick={() => page(-1)}>
                <ChevronLeft size={16} aria-hidden="true" />
              </button>
              <span aria-live="polite">{`Columns ${range.get()!.first}–${range.get()!.last} of ${n.get()}`}</span>
              <button
                type="button"
                class="pg"
                aria-label={`Next columns${range.get()!.last < n.get() ? ', up to background' : ''}`}
                disabled={range.get()!.last >= n.get()}
                onclick={() => page(1)}
              >
                <ChevronRight size={16} aria-hidden="true" />
              </button>
            </div>
            <span class="sep" aria-hidden="true">
              ·
            </span>
            {range.get()!.last < n.get() ? (
              <button type="button" class="jump" aria-label="Jump to the background column (misses)" onclick={() => toEdge(1)}>
                Background
                <ChevronsRight size={14} aria-hidden="true" />
              </button>
            ) : (
              <button type="button" class="jump" onclick={() => toEdge(-1)}>
                <ChevronsLeft size={14} aria-hidden="true" />
                First columns
              </button>
            )}
          </div>
        </div>
      ) : null}
      {/* The grid's keys are the grid's (arrows, Home, End, on the root): the focused cell is the roving tab stop, and
          focus pins its readout, so a click (which focuses the cell) and a key reach the same state with no handler per
          cell. A paged matrix is a deliberate scroller (`data-scrolls`): whole columns, with the pager and the overview
          saying what is beyond the edge instead of a fade over a half column. */}
      <div
        class="cm"
        data-scrolls={fit.get() ? true : undefined}
        class:fixed={fixed.get()}
        class:paged={!!fit.get()}
        class:more-r={moreRight.get()}
        style={{
          '--fit': fit.get() ? `${fit.get()!.cell}px` : undefined,
          '--label-h': labelH.get() !== null ? `${labelH.get()}px` : undefined,
          '--n': n.get(),
          '--min-cell': p.minCell !== undefined ? `${p.minCell}px` : undefined,
          '--max-cell': `${p.maxCell}px`,
        }}
        role="grid"
        aria-label={p.label}
        aria-describedby={`${uid}-readout`}
        ref={(e) => (root = e)}
        onkeydown={onkeydown}
        onscroll={onscroll}
        onpointerover={onpointerover}
        onpointerout={onpointerout}
        onfocusin={onfocusin}
        onfocusout={onfocusout}
      >
        <div class="row" role="row">
          <div class="corner" role="columnheader">
            <span>Predicted →</span>
            <span>True ↓</span>
          </div>
          <Index each={names.get()}>
            {(name, c) => (
              <div class="colhead" role="columnheader" title={name()}>
                <span class="dot" style:background={dotColor(c)}></span>
                <span class="lbl">{name()}</span>
              </div>
            )}
          </Index>
        </div>
        <Index each={names.get()}>
          {(rname, r) => (
            <div class="row" role="row">
              <div class="rowhead" role="rowheader" title={rname()}>
                <span class="dot" style:background={dotColor(r)}></span>
                <span class="lbl">{rname()}</span>
              </div>
              <Index each={names.get()}>{(_c, c) => cellAt(r, c)}</Index>
            </div>
          )}
        </Index>
      </div>
      {range.get() ? readout() : null}
      {p.scale ? (
        <div class="scale" aria-hidden="true">
          <span class="scale-name">Share of the true class</span>
          {/* Empty cells first, a swatch of their own (they are not on the ramp); then the stepped floor (under 1 %, 1 %,
              5 %) as three blocks and the even ramp from 10 % to 100 %. Every block is labelled the same way, as a
              share. */}
          <span class="ramp-key">
            <span class="none"></span>
            <Index each={[0.005, FLOOR[0], FLOOR[1]]}>{(v) => <span class="step" style={{ '--s': rampShare(v(), theme.get()) }}></span>}</Index>
            <span class="ramp" style={{ '--s0': rampShare(FLOOR[2], theme.get()), '--s1': rampShare(1, theme.get()) }}></span>
            <span class="tick none-t">None</span>
            <Index each={['<1%', '1%', '5%']}>{(t) => <span class="tick">{t()}</span>}</Index>
            <span class="tick ramp-ticks">
              <span>10%</span>
              <span>100%</span>
            </span>
          </span>
        </div>
      ) : null}
      {/* Under the scale where every column fits; directly under the grid while it pages, so the controls above the
          cells are one row and the hint at rest sits where the counts will be. */}
      {!range.get() ? readout() : null}
    </>
  );
}

css`
  .row {
    display: contents;
  }
  .cm {
    display: grid;
    grid-template-columns: minmax(64px, 96px) repeat(var(--n), minmax(22px, 1fr));
    gap: 2px;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    overflow-x: auto;
  }
  /* Fixed minimum cells: the matrix scrolls sideways under pinned row labels. */
  .cm.fixed {
    /* Cells grow with their column up to \`maxCell\`, so a wide panel does not turn the matrix into a wall. */
    grid-template-columns: 96px repeat(var(--n), minmax(var(--min-cell), var(--max-cell)));
    overscroll-behavior-x: contain;
    /* Scrolling a cell into view stops short of the pinned labels (96 + 2 px gap) and the 24 px fade. */
    scroll-padding-inline: 98px 24px;
  }
  /* Pages of whole columns (pageFit): cells sized so the page fills the width, the scroll snapping to column starts,
     and the rotated label row as tall as the longest label in view (an off-screen "background" no longer leaves an
     empty band over short labels). */
  .cm.paged {
    grid-template-columns: 96px repeat(var(--n), var(--fit));
    scroll-snap-type: x mandatory;
    scroll-padding-inline: 98px 0;
  }
  .cm.paged .colhead {
    scroll-snap-align: start;
    min-height: 0;
    height: calc(var(--label-h, 48px) + 16px);
  }
  /* A label keeps its own length (never squeezed to an ellipsis by a row measured a moment before). */
  .cm.paged .colhead .lbl {
    flex-shrink: 0;
  }
  .cm.paged .cell {
    min-width: 0;
  }
  .cm.more-r {
    mask-image: linear-gradient(to left, transparent, #000 24px);
  }
  .fixed .corner,
  .fixed .rowhead {
    position: sticky;
    left: 0;
    z-index: 1;
    background: var(--bg-panel);
    /* Covers the 2 px gap so scrolled cells don't show between the label and the first column. */
    box-shadow: 2px 0 0 var(--bg-panel);
  }
  .fixed .cell {
    min-width: var(--min-cell);
  }
  @media (pointer: coarse) {
    .cm.fixed {
      --min-cell: 44px !important;
    }
  }
  /* Orientation, at the helper size: what the columns and the rows are. */
  .corner {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    gap: 2px;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    padding: 0 0 4px;
    white-space: nowrap;
  }
  .colhead {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    gap: 4px;
    min-height: 64px;
    padding-bottom: 4px;
    color: var(--text-secondary);
  }
  .colhead .lbl {
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    max-height: 72px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .rowhead {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    color: var(--text-secondary);
    padding-right: 4px;
  }
  .rowhead .lbl {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  /* The raw palette hue, edged so pale classes hold up on white. */
  :global([data-theme='light']) .dot {
    box-shadow: inset 0 0 0 1px #0a0e1433;
  }
  .cell {
    aspect-ratio: 1;
    display: grid;
    place-items: center;
    border-radius: 3px;
    /* One ramp in ink over the panel, perceptually even (oklab), from a visible first step to 94 %. Empty cells
       override it below. */
    background: color-mix(in oklab, var(--text-primary) calc(var(--s) * 100%), var(--bg-panel));
    font-size: var(--type-sm-size);
    font-variant-numeric: tabular-nums;
    color: var(--text-secondary);
    transition: box-shadow var(--dur-fast) var(--ease-standard);
    min-width: 22px;
  }
  /* Each cell is a size container, so a number too wide for its cell gives way to a dot: up to two characters ("<1",
     "42") from 20 px, "100" from 28 px. */
  .cell {
    container-type: inline-size;
  }
  .mark {
    display: none;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: currentColor;
  }
  @container (max-width: 19.98px) {
    .cell .v {
      display: none;
    }
    .cell .mark {
      display: block;
    }
  }
  @container (max-width: 27.98px) {
    .cell .v.long {
      display: none;
    }
    .cell .v.long + .mark {
      display: block;
    }
  }
  /* A four-character count ("1.2k") from 36 px. */
  @container (max-width: 35.98px) {
    .cell .v.longer {
      display: none;
    }
    .cell .v.longer + .mark {
      display: block;
    }
  }
  /* No objects at all is the panel itself, edged by the hairline grid (both themes); the ramp begins at the first
     object. */
  .cell.zero {
    background: var(--bg-panel);
    box-shadow: inset 0 0 0 1px var(--line-subtle);
  }
  .cell.hl,
  .cell.zero.hl {
    box-shadow: inset 0 0 0 1px var(--line-strong);
  }
  /* The active cell (tapped, or pointed at from a summary): the selection's 2 px ink stroke (never lime, §2.7);
     keyboard focus has its own ring. */
  .cell.pin:not(:focus-visible) {
    box-shadow:
      inset 0 0 0 2px var(--text-primary),
      inset 0 0 0 3px var(--bg-panel);
  }
  /* Low cells sit near the well, high cells near the ink; the label takes the far extreme (labelHigh in ./confusionInk). */
  :global([data-theme='dark']) .cell :is(.v, .mark) {
    color: #ffffff;
  }
  :global([data-theme='dark']) .cell :is(.v, .mark).hi,
  :global([data-theme='light']) .cell :is(.v, .mark) {
    color: #000000;
  }
  :global([data-theme='light']) .cell :is(.v, .mark).hi {
    color: #ffffff;
  }
  .cell .v.hi {
    font-weight: var(--weight-medium);
  }
  /* The column-range control: the columns in view between two paging buttons. */
  .range {
    display: flex;
    align-items: center;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .pg {
    position: relative;
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
  }
  /* A 44 × 44 px hit area around the 28 px chevron (as ui/Button does), on any pointer. */
  .pg::after {
    content: '';
    position: absolute;
    inset: -8px;
  }
  .pg:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .pg:disabled {
    color: var(--text-disabled);
  }
  .readout {
    margin-top: var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  /* One row (it wraps only where the jump does not fit): thumbnail, pager, jump; the readout under it. */
  .ov {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 44px;
  }
  /* Under a paging grid: two lines kept, so tapping a cell never moves what follows. */
  .readout.below {
    min-height: calc(2 * var(--type-sm-line));
  }
  .ov {
    margin-bottom: var(--space-3);
  }
  .ov .sep {
    color: var(--text-tertiary);
  }
  /* A text button, aligned with the range's label (past the paging button's inset). */
  .jump {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    justify-self: start;
    min-height: 28px;
    margin-left: calc(var(--space-2) * -1);
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-primary);
  }
  .jump:hover {
    background: var(--bg-hover);
  }
  /* A 44 px tall hit area around the 28 px link on any pointer, as the chevrons have (its label makes it wider than
     44); touch makes the box itself 44 px. */
  .jump {
    position: relative;
  }
  .jump::after {
    content: '';
    position: absolute;
    inset: -8px 0;
  }
  @media (pointer: coarse) {
    .jump {
      min-height: 44px;
    }
    .jump::after {
      content: none;
    }
  }
  .ov-side {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0 var(--space-1);
    min-width: 0;
  }
  .ov-side .jump {
    margin-left: 0;
    padding: 0 var(--space-1);
  }
  .ov-map {
    display: block;
    overflow: visible;
    cursor: pointer;
    touch-action: manipulation;
  }
  .ov-cell {
    fill: color-mix(in oklab, var(--text-primary) calc(var(--s) * 100%), var(--bg-panel));
  }
  .ov-cell.zero {
    fill: var(--bg-panel);
    stroke: var(--line-subtle);
    stroke-width: 1;
  }
  .ov-band {
    fill: none;
    stroke: var(--text-secondary);
    stroke-width: 1;
  }
  .ov-pin {
    fill: none;
    stroke: var(--accent-line);
    stroke-width: 2;
  }
  .readout b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .num {
    font-variant-numeric: tabular-nums;
  }
  .scale {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-start;
    gap: var(--space-1) var(--space-3);
    max-width: 440px;
    margin-top: var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  /* Empty cells are not on the ramp: their key stands apart (16 px further), an edged block like the cells. */
  .ramp-key {
    flex: 1 1 240px;
    display: grid;
    grid-template-columns: 48px 32px 32px 32px minmax(96px, 1fr);
    column-gap: 4px;
    row-gap: var(--space-1);
    padding-top: 4px;
  }
  .none,
  .step,
  .ramp {
    height: 8px;
    border-radius: 2px;
  }
  .none {
    width: 32px;
    background: var(--bg-panel);
    box-shadow: inset 0 0 0 1px var(--line-default);
  }
  /* Each block's label centred under it; the ramp's two ends at its ends. None's label sits under its own block. */
  .tick {
    white-space: nowrap;
    text-align: center;
  }
  .tick.none-t {
    width: 32px;
  }
  .step {
    background: color-mix(in oklab, var(--text-primary) calc(var(--s) * 100%), var(--bg-panel));
  }
  .ramp {
    background: linear-gradient(
      to right in oklab,
      color-mix(in oklab, var(--text-primary) calc(var(--s0) * 100%), var(--bg-panel)),
      color-mix(in oklab, var(--text-primary) calc(var(--s1) * 100%), var(--bg-panel))
    );
  }
  .tick {
    font-variant-numeric: tabular-nums;
  }
  .ramp-ticks {
    display: flex;
    justify-content: space-between;
  }
`;
