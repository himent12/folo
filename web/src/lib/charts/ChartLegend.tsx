/** A line chart's legend, its readout: each series' swatch, name and value (at the cursor, else its latest). */
import { css, For } from 'fw';
import type { Pt } from './downsample';
import { latest, type Series } from './lineSeries';
import SeriesSwatch from './SeriesSwatch';

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
}

export default function ChartLegend(props: ChartLegendProps) {
  return (
    <ul class="legend" aria-hidden="true">
      {/* The note says which value the legend shows: the cursor's x, or at rest `legendNote` (narrow plots too, where
          the cursor's x reads out on the axis instead). */}
      {props.hintText ? (
        <li class="note hint">{props.hintText}</li>
      ) : props.hover && !props.compact ? (
        <li class="note at">{props.hover.title}</li>
      ) : props.legendNote && !props.hover ? (
        <li class="note">{props.legendNote}</li>
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
