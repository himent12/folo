/**
 * One of a run's four charts (design/UNIFIED.md §3, §4 Run): Validation mAP, Training loss, Precision / Recall and
 * Learning rate, each a panel whose header holds the title, the qualifier (what x is, what the values are) and the
 * chart's controls, then the legend as the readout and the plot in its inset well. They share one x axis and one
 * cursor (./runModel): hovering or arrowing through one reads every one, and "So far / All N", in the first chart's
 * header while the run trains, fits all of them.
 */
import { computed, css, For } from 'fw';
import ChartSkeleton from '$lib/charts/ChartSkeleton';
import LineChart from '$lib/charts/LineChart';
import LossOptions from '$lib/charts/LossOptions';
import { UNIT_ZOOM_NOTE, unitZoomed } from '$lib/charts/domain';
import { scoresTitle, type EvalSource } from '$lib/evaluated';
import { conf, metric, num, pow10 } from '$lib/format';
import InfoButton from '$lib/ui/InfoButton';
import Panel from '$lib/ui/Panel';
import Segmented from '$lib/ui/Segmented';
import { LOSS_TERMS } from './readings';
import type { RunModel } from './runModel';

export type ChartKind = 'map' | 'loss' | 'pr' | 'lr';

export interface RunChartProps {
  kind: ChartKind;
  model: RunModel;
  /** The chart's height, its band and x ticks included (the plot is 46 px less). */
  height: number;
  loading: boolean;
  /** A queued run with nothing yet: the frame a running run fills, its plot empty. */
  queued?: boolean;
  /** What the epochs' scores are on (lib/evaluated `runEvalSource`): the title names the split when it is not validation. */
  evalOn?: EvalSource | null;
}

const epochTitle = (x: number) => `Epoch ${x.toFixed(x % 1 === 0 ? 0 : 1)}`;

export default function RunChart(props: RunChartProps) {
  const m = props.model;
  const kind = props.kind;
  const height = () => props.height;
  const unit = kind === 'map' || kind === 'pr';
  const series = kind === 'map' ? m.mapSeries : kind === 'pr' ? m.prSeries : kind === 'loss' ? m.lossSeries : m.lrSeries;
  const fit = kind === 'loss' ? m.lossFit(height) : kind === 'lr' ? m.lrFit(height) : m.unitFit(() => series.get(), height);
  const zoomed = computed(() => unit && unitZoomed(fit.get()));
  const live = computed(() => m.live.get() && m.run().status !== 'queued');

  const title = computed(() => (kind === 'map' ? `${scoresTitle(props.evalOn ?? null)} mAP` : kind === 'loss' ? 'Training loss' : kind === 'pr' ? 'Precision / Recall' : 'Learning rate'));
  /** The header's qualifier: what x is and what the values are; the loss's last raw step beside its smoothed total. */
  const qualifier = computed(() => {
    if (kind === 'map') return 'by epoch';
    if (kind === 'pr') {
      const pc = m.prConf.get();
      return pc != null ? `by epoch · at best-F1 conf (latest ${conf(pc)})` : "by epoch · at each epoch's best-F1 conf";
    }
    if (kind === 'lr') return m.lrExp.get() ? `${m.schedule.get()} · × ${pow10(m.lrExp.get())}` : m.schedule.get();
    const raw = m.lastStep.get()?.loss.total;
    const sl = m.smoothedLast.get();
    // The scale and the smoothing are the options button's own label ("EMA 0.98 · Log"), beside this.
    if (raw === undefined) return 'by epoch';
    return sl !== undefined ? `raw ${num(raw, 3)} · smoothed ${num(sl, 3)}` : `raw ${num(raw, 3)}`;
  });
  const empty = computed(() =>
    props.queued
      ? 'Curves appear here once training starts'
      : !unit
        ? 'Waiting for the first step…'
        : m.unvalidated.get()
          ? 'No validation: it ended before its first epoch finished.'
          : 'Waiting for the first validation…',
  );
  const lrScale = () => 10 ** m.lrExp.get();

  // The loss terms in full, behind the ⓘ's short line ("How it's computed").
  const lossHow = () => (
    <span class="terms-list">
      <For each={LOSS_TERMS} key={(t) => t[0]}>
        {(t) => (
          <span>
            <b>{t()[0]}</b>: {t()[1]}
          </span>
        )}
      </For>
    </span>
  );

  return (
    <Panel
      title={title.get()}
      meta={qualifier.get()}
      id={`chart-${kind}`}
      class="run-chart"
      actions={
        kind === 'map' && live.get()
          ? () => (
              <Segmented
                size="sm"
                label="Chart range"
                bind:value={m.horizon}
                options={[
                  { value: 'observed', label: 'So far' },
                  { value: 'full', label: `All ${m.run().epochs}` },
                ]}
              />
            )
          : kind === 'loss'
            ? () => (
                // One control for the loss view (smoothing and y scale behind one button), so the header keeps at most
                // one segmented group; the ⓘ says what the terms mean (on phones it opens in place under the heading).
                <>
                  <InfoButton chart label="What the loss terms mean" text="What training lowers: total, the weighted sum of box (placement), cls (class) and dfl (edges)." how={lossHow} />
                  <LossOptions bind:smoothing={m.smoothing} bind:scale={m.scale} />
                </>
              )
            : undefined
      }
    >
      {props.loading ? (
        <ChartSkeleton height={height()} />
      ) : (
        <>
          {/* No point yet: the legend's row is held empty, so the plot does not drop when it appears. */}
          {!props.queued && !series.get().some((x) => x.points.length > 0 && !x.ghost) ? <div class="legend-slot" aria-hidden="true"></div> : null}
          <LineChart
            series={series.get()}
            height={height()}
            xDomain={m.xDomain.get()}
            xInteger
            bind:cursor={m.cursor}
            band
            future={m.future.get() && kind === 'map' ? m.future.get() : m.future.get() ? { x: m.future.get()!.x } : undefined}
            yDomain={fit.get()?.domain}
            yTickValues={fit.get() && !fit.get()!.log ? fit.get()!.ticks : undefined}
            yDecimals={fit.get() && !fit.get()!.log ? fit.get()!.decimals : undefined}
            yLog={!!fit.get()?.log}
            yFormat={kind === 'lr' ? (v) => String(+(v / lrScale()).toFixed(2)) : undefined}
            valueFormat={kind === 'lr' ? (v) => (v / lrScale()).toFixed(2) : (v) => metric(v)}
            xTitle={epochTitle}
            curve={unit ? 'monotone' : 'linear'}
            markers={kind === 'map' ? m.best.get() : undefined}
            legendBest={kind === 'map' ? m.bestLegend.get() : undefined}
            endDots={unit}
            xMark={m.xMark.get()}
            yLabel={zoomed.get() ? UNIT_ZOOM_NOTE : undefined}
            legendNote={unit ? m.latestEp.get() : m.stepEp.get()}
            label={`${title.get()} by epoch`}
            endLabels={kind === 'loss'}
            empty={empty.get()}
          />
        </>
      )}
    </Panel>
  );
}

css`
  .terms-list {
    display: grid;
    gap: var(--space-1);
  }
  /* The legend's line and the chart's gap under it (lib/charts LineChart, ChartLegend). */
  .legend-slot {
    height: calc(var(--type-sm-line) + var(--space-2));
  }
`;
