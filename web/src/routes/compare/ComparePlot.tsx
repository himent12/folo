// Plot section, status notices, and phone disclosure for Compare runs (`../RunCompare.tsx`).
import ChevronRight from '$icons/chevron-right';
import { bindable, css, fade, For, Key } from 'fw';
import type { Resource } from '$lib/api/resource';
import type { Run, RunMetrics } from '$lib/api/types';
import ChartSkeleton from '$lib/charts/ChartSkeleton';
import { fitFloor, type Fit } from '$lib/charts/domain';
import LineChart, { type Series } from '$lib/charts/LineChart';
import { metric } from '$lib/format';
import Button from '$lib/ui/Button';
import ErrorState from '$lib/ui/ErrorState';
import Ident from '$lib/ui/Ident';
import { motion } from '$lib/ui/motion';
import Segmented from '$lib/ui/Segmented';
import type { Axis, CompareMetricDef, MetricId, YAxis } from './config';

export interface MissingRunsNoticeProps {
  dropped: string[];
  onDismiss: () => void;
}

/** Announces missing run IDs that were reconciled out of `?ids=` after the runs list loaded. */
export function MissingRunsNotice(props: MissingRunsNoticeProps) {
  return (
    <>
      {props.dropped.length ? (
        <p class="caveat stale-note" role="status" data-missing-runs>
          <span>
            {props.dropped.length === 1 ? 'Removed missing run from this comparison: ' : `Removed ${props.dropped.length} missing runs from this comparison: `}
            <For each={props.dropped} key={(id) => id}>
              {(id, i) => (
                <>
                  {i() > 0 ? ', ' : null}
                  <Ident value={id()} />
                </>
              )}
            </For>
            .
          </span>{' '}
          <Button size="sm" variant="ghost" onclick={props.onDismiss}>
            Dismiss
          </Button>
        </p>
      ) : null}
    </>
  );
}

export interface ComparePlotProps {
  runs: Resource<Run[]>;
  metrics: Resource<Map<string, RunMetrics>>;
  failedRuns: Run[];
  failedWithCache: boolean;
  retryingMetric: Record<string, boolean>;
  onRetryMetric: (id: string) => void;
  metricId: MetricId;
  metricDef: CompareMetricDef;
  series: Series[];
  height: number;
  x: Axis;
  xDomain: [number, number];
  fit: Fit | undefined;
  yCaption: string | undefined;
  cursor?: number | null;
  hint?: string | null;
  highlight: string | null;
}

/** The compare line plot with stale/per-run retry banners, loading skeleton, and total-error fallback. */
export default function ComparePlot(props: ComparePlotProps) {
  const cursor = bindable(props, 'cursor', null);
  const hint = bindable(props, 'hint', null);

  return (
    <>
      {props.metrics.stale !== undefined && props.metrics.data ? (
        <p class="caveat stale-note" role="status">
          Could not refresh these runs' metrics: the curves may be behind. Trying again…{' '}
          <Button size="sm" variant="ghost" onclick={() => void props.metrics.refresh()}>
            Retry
          </Button>
        </p>
      ) : props.failedRuns.length && props.metrics.data ? (
        <p class="caveat stale-note" role="status">
          <span>
            {props.failedWithCache
              ? `Could not refresh these runs' metrics (${props.failedRuns.map((r) => r.name).join(', ')}): the curves may be behind. Trying again…`
              : `Could not load metrics for ${props.failedRuns.map((r) => r.name).join(', ')}. Other runs are shown.`}
          </span>{' '}
          <For each={props.failedRuns} key={(r) => r.id}>
            {(r) => (
              <Button size="sm" variant="ghost" loading={!!props.retryingMetric[r().id]} onclick={() => props.onRetryMetric(r().id)}>
                {props.failedRuns.length === 1 ? 'Retry' : `Retry ${r().name}`}
              </Button>
            )}
          </For>
        </p>
      ) : null}
      {props.metrics.error ? (
        <ErrorState error={props.metrics.error} compact retry={() => props.metrics.refresh()} title="Could not load these runs' metrics" />
      ) : !props.metrics.data || !props.runs.data ? (
        <ChartSkeleton height={props.height} legend={false} />
      ) : (
        // A metric swap fades the new chart in over 160 ms (0 under reduced motion); within a metric the domain eases.
        <Key when={props.metricId}>
          <div class="swap" in:fade={{ duration: motion(160) }}>
            <LineChart
              series={props.series}
              height={props.height}
              xDomain={props.xDomain}
              xInteger={props.x === 'epoch'}
              yDomain={props.fit?.domain}
              yTickValues={props.fit?.ticks}
              yDecimals={props.fit?.decimals}
              curve="monotone"
              valueFormat={(v) => metric(v)}
              xFormat={(v) => (props.x === 'epoch' ? String(Math.round(v)) : `${Math.round(v)}%`)}
              xTitle={(v) => (props.x === 'epoch' ? `Epoch ${Math.round(v)}` : `${v.toFixed(0)}% through`)}
              yLabel={props.yCaption}
              bind:cursor={cursor}
              bind:hint={hint}
              legend={false}
              highlight={props.highlight}
              recede
              endDots
              endLabels
              label={`${props.metricDef.label} for the compared runs`}
            />
          </div>
        </Key>
      )}
    </>
  );
}

export interface PhoneAboutProps {
  floorNote: string | null;
  fit: Fit | undefined;
  yAxis: YAxis;
  setY: (v: YAxis) => void;
  x: Axis;
}

/** Phones: collapsible legend explanation and y-axis mode switch under the plot. */
export function PhoneAbout(props: PhoneAboutProps) {
  return (
    <details class="about">
      <summary>
        <ChevronRight size={14} aria-hidden="true" />{' '}
        <span>How to read this</span>
      </summary>
      <div class="about-body">
        {/* The y axis's choice: the control row has no room for it on a phone. */}
        <div class="about-axis">
          <p>{props.floorNote ? `The y axis starts at ${fitFloor(props.fit)}, under where the runs spend most of their length.` : 'The y axis starts at 0.'}</p>
          <Segmented
            size="sm"
            label="Y axis"
            value={props.yAxis}
            onchange={props.setY}
            options={[
              { value: 'fit', label: 'Fit' },
              { value: 'zero', label: 'From 0' },
            ]}
          />
        </div>
        {props.x === 'progress' ? <p>By progress puts each run over its planned epochs, so a short plan and a long one compare by shape.</p> : null}
        <p>
          Each line ends in its run's status: a lime dot while it trains, a square where it was stopped, a cross where it
          failed.
        </p>
        <p>
          Colours follow the run, not its rank. Tap a run to keep its line in front; Edit shows how each run started
          (fine-tuned, or from scratch). The table's baseline stays until you make another run the baseline.
        </p>
        <p>Tap or drag across the chart to read every run at one point; tap it again or anywhere else to let go.</p>
      </div>
    </details>
  );
}

css`
  .caveat {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  /* A metrics refresh failed: the curves stay, this line says they may be behind, its Retry beside it. */
  .stale-note {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
  }
  .about {
    border-top: 1px solid var(--line-subtle);
  }
  .about summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: 44px;
    list-style: none;
    cursor: pointer;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .about summary::-webkit-details-marker {
    display: none;
  }
  .about summary :global(svg) {
    flex-shrink: 0;
    color: var(--text-tertiary);
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .about[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .about summary:focus-visible {
    outline-offset: -2px;
  }
  .about-body {
    display: grid;
    gap: var(--space-2);
    padding-bottom: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .about-axis {
    display: grid;
    gap: var(--space-2);
    justify-items: start;
  }
  /* The fading wrapper of a swapped chart: it takes the chart's place in the layout. */
  .swap {
    min-width: 0;
  }
`;
