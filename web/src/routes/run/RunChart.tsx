/**
 * One of a run's four charts (design/UNIFIED.md §3, §4 Run): Validation mAP, Training loss, Precision / Recall and
 * Learning rate (or, picked in its header, the throughput over the whole run), each a panel whose header holds the
 * title, the qualifier (what x is, what the values are) and the chart's controls, then the legend as the readout and
 * the plot in its inset well. They share one x axis (./runModel), and "So far / All N", in the first chart's header
 * while the run trains, fits all of them; each has its own cursor: hovering or arrowing through one reads that chart
 * only, the others keep their latest values.
 */
import { computed, css, For, MediaQuery, signal } from 'fw';
import ChartSkeleton from '$lib/charts/ChartSkeleton';
import LineChart from '$lib/charts/LineChart';
import LossOptions from '$lib/charts/LossOptions';
import { UNIT_ZOOM_NOTE, unitZoomed } from '$lib/charts/domain';
import { media } from '$lib/design/tokens';
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
  /** This chart's cursor (data x, epochs): the pointer or the arrows on this chart move it, no other chart's. */
  const cursor = signal<number | null>(null);
  const kind = props.kind;
  const height = () => props.height;
  const unit = kind === 'map' || kind === 'pr';
  /** The fourth chart showing the throughput instead of the learning rate. */
  const ips = () => kind === 'lr' && m.lrView.get() === 'ips';
  const series = kind === 'map' ? m.mapSeries : kind === 'pr' ? m.prSeries : kind === 'loss' ? m.lossSeries : computed(() => (ips() ? m.ipsSeries : m.lrSeries).get());
  const lrFits = kind === 'lr' ? { lr: m.lrFit(height), ips: m.ipsFit(height) } : null;
  const fit = kind === 'loss' ? m.lossFit(height) : lrFits ? computed(() => (ips() ? lrFits.ips : lrFits.lr).get()) : m.unitFit(() => series.get(), height);
  const zoomed = computed(() => unit && unitZoomed(fit.get()));
  const live = computed(() => m.live.get() && m.run().status !== 'queued');
  /** Phones: the qualifiers in fewer words, so the header's line holds them whole. */
  const phone = new MediaQuery(media.phone, false);
  /**
   * Laptops (1024–1279 px): the two-fifths panels' headers hold the qualifier's essential words only, never cut: the
   * loss's raw step (its smoothed total is the legend's), the learning rate's power of ten (its schedule is Config's).
   */
  const laptopUp = new MediaQuery(media.laptopUp, false);
  const desktop = new MediaQuery(media.desktop, false);
  const tight = () => laptopUp.current && !desktop.current;

  const title = computed(() => (kind === 'map' ? `${scoresTitle(props.evalOn ?? null)} mAP` : kind === 'loss' ? 'Training loss' : kind === 'pr' ? 'Precision / Recall' : ips() ? 'Throughput' : 'Learning rate'));
  /** The header's qualifier: what x is and what the values are; the loss's last raw step beside its smoothed total. */
  const qualifier = computed(() => {
    if (kind === 'map') return 'by epoch';
    if (kind === 'pr') {
      const pc = m.prConf.get();
      if (phone.current) return pc != null ? `best-F1 conf ${conf(pc)}` : 'at best-F1 conf';
      return pc != null ? `by epoch · at best-F1 conf (latest ${conf(pc)})` : "by epoch · at each epoch's best-F1 conf";
    }
    if (ips()) return 'images/s · smoothed';
    if (kind === 'lr' && tight()) return m.lrExp.get() ? `× ${pow10(m.lrExp.get())}` : m.schedule.get().split(' · ').at(-1)!;
    if (kind === 'lr') return m.lrExp.get() ? `${m.schedule.get()} · × ${pow10(m.lrExp.get())}` : m.schedule.get();
    const raw = m.lastStep.get()?.loss.total;
    const sl = m.smoothedLast.get();
    // The scale and the smoothing are the options button's own label ("EMA 0.98 · Log"), beside this.
    if (raw === undefined) return 'by epoch';
    return sl !== undefined && !tight() ? `raw ${num(raw, 3)} · smoothed ${num(sl, 3)}` : `raw ${num(raw, 3)}`;
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
            : kind === 'lr'
              ? () => (
                  // The throughput's whole history: the tile's sparkline is its last 120 steps only.
                  <Segmented
                    size="sm"
                    label="Chart"
                    bind:value={m.lrView}
                    options={[
                      { value: 'lr', label: 'LR' },
                      { value: 'ips', label: 'Throughput' },
                    ]}
                  />
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
            bind:cursor={cursor}
            band
            future={m.future.get() && kind === 'map' ? m.future.get() : m.future.get() ? { x: m.future.get()!.x } : undefined}
            yDomain={fit.get()?.domain}
            yTickValues={fit.get() && !fit.get()!.log ? fit.get()!.ticks : undefined}
            yDecimals={fit.get() && !fit.get()!.log ? fit.get()!.decimals : undefined}
            yLog={!!fit.get()?.log}
            yFormat={ips() ? (v) => num(v, 0) : kind === 'lr' ? (v) => String(+(v / lrScale()).toFixed(2)) : undefined}
            valueFormat={ips() ? (v) => num(v, 0) : kind === 'lr' ? (v) => (v / lrScale()).toFixed(2) : (v) => metric(v)}
            xTitle={epochTitle}
            curve={unit ? 'monotone' : 'linear'}
            markers={kind === 'map' ? m.best.get() : undefined}
            legendBest={kind === 'map' ? m.bestLegend.get() : undefined}
            endDots
            xMark={m.xMark.get()}
            xEvent={kind === 'map' || kind === 'loss' ? m.failEvent.get() : undefined}
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
