/**
 * A run's metrics (design/UNIFIED.md §4 Run, rows 1–3): the KPI strip, then Validation mAP (3/5) beside Training loss
 * (2/5) with 220 px plots, then Precision / Recall beside Learning rate with 140 px plots on the same 3/5 · 2/5 split, so
 * the gap between the panels does not move sideways from one row to the next (as Desktop's). Below 1024 px the rows
 * stack; on phones the second row moves into the section picker (Run's `pr` and `lr` parts, ./RunChart there too).
 * A history that failed to load says so once, with Retry, in place of the strip and the charts (whose readings it
 * holds); a run that never trained says that in one line (Outcome above says how it ended).
 */
import { css } from 'fw';
import type { EvalSource } from '$lib/evaluated';
import ErrorState from '$lib/ui/ErrorState';
import RunChart from './RunChart';
import RunKpis from './RunKpis';
import type { RunModel } from './runModel';

/** The charts' heights, the band and the x ticks included (the plots are 46 px less: 220 and 140, phones 180). */
export const TALL = 266;
export const SHORT = 186;
export const PHONE = 226;

export interface RunChartsProps {
  model: RunModel;
  loading: boolean;
  evalOn: EvalSource | null;
  /** The run's history failed to load: one error with Retry, never "Waiting for the first validation…". */
  error?: unknown;
  retry: () => void;
  /** It ended before its first training step: one line in place of the charts, and tiles with nothing to say. */
  never: boolean;
  phone: boolean;
}

export default function RunCharts(props: RunChartsProps) {
  const m = props.model;
  return (
    <section class="metrics" aria-label="Run metrics" data-tip-bounds>
      {props.error !== undefined ? (
        <div class="failed">
          <ErrorState error={props.error} title="Could not load this run's metrics" retry={props.retry} compact />
        </div>
      ) : props.never ? (
        <p class="never">No metrics: it ended before its first training step.</p>
      ) : (
        <>
          <RunKpis model={m} loading={props.loading} />
          <div class="row wide">
            <RunChart kind="map" model={m} height={props.phone ? PHONE : TALL} loading={props.loading} evalOn={props.evalOn} />
            <RunChart kind="loss" model={m} height={props.phone ? PHONE : TALL} loading={props.loading} />
          </div>
          {!props.phone ? (
            <div class="row wide">
              <RunChart kind="pr" model={m} height={SHORT} loading={props.loading} />
              <RunChart kind="lr" model={m} height={SHORT} loading={props.loading} />
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}

css`
  .metrics {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  .row {
    display: grid;
    gap: var(--space-4);
    align-items: stretch;
    min-width: 0;
  }
  .row.wide {
    grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
  }
  .failed,
  .never {
    padding: var(--space-4);
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
  }
  .never {
    margin: 0;
    color: var(--text-tertiary);
  }
  @media (max-width: 1023px) {
    .row.wide {
      grid-template-columns: minmax(0, 1fr);
    }
  }
`;
