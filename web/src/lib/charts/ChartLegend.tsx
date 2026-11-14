/**
 * A line chart's legend, its readout: each series' swatch, name and value (at the cursor, else its latest). It is one
 * line wherever that line fits the chart's width; where it does not, the items wrap whole into even rows (`legendColumns`:
 * the fewest rows that fit, the items spread over them as evenly as they go), each column as wide as its widest item.
 */
import { computed, css, For, MediaQuery } from 'fw';
import { media } from '$lib/design/tokens';
import type { Pt } from './downsample';
import { latest, type Series } from './lineSeries';
import SeriesSwatch from './SeriesSwatch';
import { textMeasurer } from './textWidth';

/** The legend's geometry (its CSS below): the gap between items, inside an item, and the swatch. */
const GAP = 12;
const INNER = 6;
const SWATCH = 14;
const RING = 10;

/** Columns for `n` items when at most `fit` fit on a line: the fewest rows, as evenly filled as they go (4 → 2 + 2). */
export function legendColumns(n: number, fit: number): number {
  const rows = Math.ceil(n / Math.max(1, fit));
  return Math.ceil(n / rows);
}

export interface ChartLegendProps {
  hintText: string | null;
  hover: {
    rows: { s: Series; p: Pt; at: string | null }[];
    title: string;
  } | null;
  /** A narrow plot: the cursor's x reads out on the axis, not in the note. */
  compact: boolean;
  legendNote: string | undefined;
  keyed: Series[];
  fmtValue: (y: number) => string;
  /** The best epoch after the series: the ring glyph (the plot's hollow ink ring), "best" and its value. */
  best?: string;
  /** The chart's width, px (0 before it is measured). */
  width?: number;
}

export default function ChartLegend(props: ChartLegendProps) {
  const measure = textMeasurer('regular');
  const phone = new MediaQuery(media.phone, false);
  // A narrow plot beside a wider one (Training loss in its two fifths): its values at rest are the latest, as the wide
  // chart's note beside it says, so the note gives its room to the readout. Phones keep it (their legends wrap).
  const restNote = computed(() => (props.compact && !phone.current ? undefined : props.legendNote));
  /** Columns when the legend must wrap, measured at rest (the latest values), so the cursor never reflows it. */
  const cols = computed(() => {
    const w = props.width ?? 0;
    if (w <= 0) return null;
    const fmt = props.fmtValue;
    const items = props.keyed.map((s) => {
      const v = latest(s);
      return SWATCH + INNER + measure(s.label) + INNER + measure(v === null || v === undefined ? '—' : fmt(v));
    });
    if (props.best) items.push(RING + INNER + measure('best') + INNER + measure(props.best));
    if (!items.length) return null;
    const note = restNote.get() ? measure(restNote.get()!) + GAP : 0;
    const line = items.reduce((a, b) => a + b, 0) + GAP * (items.length - 1) + note;
    if (line <= w) return null;
    const widest = Math.max(...items);
    return legendColumns(items.length, Math.floor((w + GAP) / (widest + GAP)));
  });
  return (
    <ul class="legend" class:wrapped={cols.get() !== null} style={{ '--legend-cols': cols.get() ?? undefined }} aria-hidden="true">
      {/* The note says which value the legend shows: the cursor's x, or at rest `legendNote` (narrow plots too, where
          the cursor's x reads out on the axis instead). */}
      {props.hintText ? (
        <li class="note hint">{props.hintText}</li>
      ) : props.hover && !props.compact ? (
        <li class="note at">{props.hover.title}</li>
      ) : restNote.get() && !props.hover ? (
        <li class="note">{restNote.get()}</li>
      ) : null}
      <For each={props.keyed} key={(s) => s.id}>
        {(s) => {
          const v = () => {
            const h = props.hover;
            return h ? h.rows.find((r) => r.s.id === s().id)?.p.y : latest(s());
          };
          return (
            <li>
              <SeriesSwatch s={s()} />
              <span class="name">{s().label}</span>
              <span class="val">{v() === null || v() === undefined ? '—' : props.fmtValue(v()!)}</span>
            </li>
          );
        }}
      </For>
      {props.best ? (
        <li class="best">
          <svg class="ring" width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
            <circle cx="5" cy="5" r="3.75" />
          </svg>
          <span class="name">best</span>
          <span class="val">{props.best}</span>
        </li>
      ) : null}
    </ul>
  );
}

css`
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-3);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  /* Too wide for one line: even rows of whole items, the note on a row of its own above them. */
  .legend.wrapped {
    display: grid;
    grid-template-columns: repeat(var(--legend-cols), max-content);
  }
  .legend.wrapped .note {
    grid-column: 1 / -1;
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
  /* A reading hint takes the note's place for as long as it applies, on narrow plots too. */
  .legend .note.hint {
    min-width: 0;
    color: var(--text-secondary);
  }
  .name {
    color: var(--text-secondary);
  }
  /* The best epoch's key: the plot's hollow ink ring, 1.5 px. */
  .ring circle {
    fill: none;
    stroke: var(--text-primary);
    stroke-width: 1.5;
  }
  .val {
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
  }
`;
