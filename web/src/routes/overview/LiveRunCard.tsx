// Overview's live run as one panel: header with progress, four KPI tiles, the validation chart, the machine's meters.
import ArrowRight from '$icons/arrow-right';
import { bindable, computed, css, effect, MediaQuery } from 'fw';
import ChartSkeleton from '$lib/charts/ChartSkeleton';
import { LiveRun } from '$lib/api/live';
import type { Resource } from '$lib/api/resource';
import type { Dataset, Run, SystemInfo } from '$lib/api/types';
import LineChart from '$lib/charts/LineChart';
import { UNIT_ZOOM_NOTE, unitZoomed } from '$lib/charts/domain';
import { ETA_PENDING, etaSettled, progressText, runProgress, trainingSeconds } from '$lib/api/progress';
import { cloudGpu } from '$lib/cloud';
import { axisMetric, deviceName, duration, eta, metric, num } from '$lib/format';
import { asOf, lastKnown, Now } from '$lib/api/freshness';
import StatusBadge, { statusWord } from '$lib/ui/StatusBadge';
import { starting, STARTING } from './slot';
import { runEvalSource, scoresTitle } from '$lib/evaluated';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Skeleton from '$lib/ui/Skeleton';
import KpiStrip from '$lib/ui/KpiStrip';
import Stat from '$lib/ui/Stat';
import { media } from '$lib/design/tokens';
import Ident from '$lib/ui/Ident';
import Meta from '$lib/ui/Meta';
import { RunModel } from '../run/runModel';
import MachineFoot from './MachineFoot';

export interface LiveRunCardProps {
  run: Run;
  dataset?: Dataset | null;
  /** The dataset as people know it (lib/runDataset): undefined while loading (a placeholder), null to leave it out. */
  datasetName?: string | null;
  lost?: number | null;
  /** The page's system info, which the machine foot starts from (./MachineFoot). */
  system?: Resource<SystemInfo>;
  /** Called once when this card's live stream reports a terminal status (`completed` / `stopped` / `failed`). */
  onended?: (run: Run) => void;
}

/**
 * The live run as one panel (design/UNIFIED.md §4 Overview): a 48 px header (name, status, the meta line, Open run)
 * whose bottom edge is the 2 px progress rail; four KPI tiles (Epoch, mAP50-95, Throughput, Time left) in a 2 × 2
 * block on the left third beside the validation chart (one row of four over it below 1280 px); a 36 px foot with the
 * machine's meters where the server reports them (./MachineFoot). The learning rate and the loss are the run page's.
 *
 * `run`: the run in the training slot (./slot), training or starting; its stream keeps the card live. `lost`
 * (bindable) reports when the run's stream went quiet (`LiveRun.lost`), so the page can tell a server that stopped
 * answering from lists that alone failed to refresh. `dataset`: undefined while the datasets load, null when the run's
 * is not to be had (its split then unknown: "Evaluation").
 */
/** The chart's height beside the tiles (legend and plot): with its 40 px header and 16 px insets, a 257 px body. */
const SIDE_CHART_H = 166;

export default function LiveRunCard(props: LiveRunCardProps) {
  const lost = bindable(props, 'lost', null);
  // What the scores are on (lib/evaluated `scoresTitle`), one word in every state: until the dataset is known it is a
  // placeholder, never a guess that changes when it arrives ("Evaluation" in the skeleton, "Validation" loaded).
  const scores = computed(() => (props.dataset === undefined ? null : scoresTitle(runEvalSource(props.dataset))));

  // Latch the last non-undefined prop so teardown computations (which can run before the parent branch disposes this
  // card when `live` flips to undefined) never read `.id` off undefined.
  let lastProp = props.run;
  const propRun = computed(() => (props.run ? (lastProp = props.run) : lastProp));
  const runId = computed(() => propRun.get().id);
  // Overview's live card does not show run logs: skip the `/logs` requests and log buffering (#309).
  // Notify Overview synchronously when the stream reports a terminal state so its summary, live slot, Recent runs,
  // and sidebar stay in lockstep before the card unmounts and before the polled run list catches up.
  const live = new LiveRun(() => runId.get(), {
    logs: false,
    onEnded: (r, terminal) => props.onended?.(r ?? { ...propRun.get(), status: terminal.status, error: terminal.error, queue_position: null }),
  });
  const run = computed(() => live.run ?? propRun.get());
  // The stream went quiet: what the card shows is the last known state, as of then. The status says so, the estimate
  // is qualified and the live marks (the badge's heartbeat, the line's lime end, the lime rail) stop (DESIGN_DIRECTION
  // §2.8).
  effect(() => {
    lost.set(live.lost);
  });
  const now = new Now(() => live.lost !== null);
  const quiet = () => live.lost !== null;

  // The run page's readings (run/runModel): the series in their hues, the planned axis with the future shaded and the
  // ETA over its boundary, the best ring, the throughput's average.
  const model = new RunModel({
    get run() {
      return run.get();
    },
    steps: live.stepsLog,
    epochs: live.epochsLog,
    get version() {
      return live.version;
    },
  });

  // Phones: a shorter plot, so the recent runs below start on the first screen.
  const phoneQuery = new MediaQuery(media.phone, false);
  const sideQuery = new MediaQuery(media.desktop, false);
  // From 1280 px the chart sits beside the 2 × 2 tiles and its height is theirs: a plot that makes each tile about
  // 128 px high (Desktop's are 120), never tiles stretched half empty to a tall chart. Below, the tiles are a row over it.
  const chartH = computed(() => (phoneQuery.current ? 186 : sideQuery.current ? SIDE_CHART_H : 246));
  const fit = model.unitFit(() => model.mapSeries.get(), () => chartH.get());
  const zoomed = computed(() => unitZoomed(fit.get()));
  // A quiet stream: the line ends without the live mark.
  const series = computed(() => (quiet() ? model.mapSeries.get().map((s) => ({ ...s, end: undefined })) : model.mapSeries.get()));
  const progress = computed(() => runProgress(run.get()));
  const latest = computed(() => model.lastEpoch.get()?.epoch);

  const r = () => run.get();
  const timeLeft = computed(() => {
    const s = r().eta_s;
    if (!etaSettled(r()) || s === null) return '—';
    return quiet() ? `~${eta(s)}` : eta(s);
  });
  const timeCap = computed(() => {
    if (!etaSettled(r())) return ETA_PENDING;
    if (quiet()) return `as of ${asOf(live.lost!, now.current)}`;
    const e = trainingSeconds(r());
    return e !== null ? `${duration(e)} elapsed` : '';
  });

  return (
    <section class="card" aria-labelledby="live-title">
      <header>
        {/* A long name gives way (with "…") before it pushes the meta under Open run. */}
        <h2 id="live-title" class="truncate">
          <a href={href(`/runs/${r().id}`)}>{r().name}</a>
        </h2>
        {starting(r()) ? (
          // Taken by the worker or next with the slot free (./slot): about to train, not waiting.
          <StatusBadge state={STARTING} lastKnown={quiet() ? lastKnown(STARTING.label, live.lost!, now.current) : undefined} />
        ) : (
          <StatusBadge run={r()} lastKnown={quiet() ? lastKnown(statusWord(r()), live.lost!, now.current) : undefined} />
        )}
        <span class="meta">
          <Meta line>
            {props.datasetName === undefined ? (
              <span>
                &#8203;<Skeleton inline w="72px" h="10px" />
              </span>
            ) : props.datasetName ? (
              <span>{props.datasetName}</span>
            ) : null}
            <Ident value={r().config.arch} />
            <span class="num">{r().config.imgsz} px</span>
            <span class="num">batch {r().config.batch}</span>
            {/* The run's device as it recorded it, whatever the foot's readings (a cloud run has no foot: this server
                does not read the rented machine). */}
            {r().cloud ? <span>{cloudGpu({ device: r().device, cloud: r().cloud! })}</span> : r().device ? <span>{deviceName(r().device)}</span> : null}
          </Meta>
        </span>
        <span class="cta">
          <Button size="sm" variant="ghost" iconRight={ArrowRight} href={href(`/runs/${r().id}`)}>
            Open run
          </Button>
        </span>
        {/* The progress drawn once, as the header's bottom edge; lime is "now", a last known progress is in ink. */}
        <div class="rail" class:quiet={quiet()} role="progressbar" aria-label="Training progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.floor(progress.get() * 100 + 1e-9)}>
          <div class="fill" style:transform={`scaleX(${progress.get()})`}></div>
        </div>
      </header>
      <div class="body">
        {/* Two flush regions divided by a hairline: the tiles, then the chart with a chart panel's header band. */}
        <div class="kpis">
          <KpiStrip flush label="Training summary">
            <div class="tile">
              <Stat label="Epoch" value={num(r().epoch)} unit={`/ ${r().epochs}`} sub={`${progressText(r())} complete`} />
            </div>
            <div class="tile">
              <Stat
                label="mAP50-95 · latest"
                value={metric(model.lastEpoch.get()?.val.map50_95)}
                sub={r().best ? (r().best!.map50_95 === 0 ? 'no detections yet' : `best ${metric(r().best!.map50_95)} @ ${r().best!.epoch}`) : 'after the first epoch'}
                loading={live.loading}
              />
            </div>
            <div class="tile">
              <Stat
                label="Throughput"
                value={num(model.lastStep.get()?.imgs_per_s, 0)}
                unit="img/s"
                sub={model.ips.get().avg !== null ? `avg ${num(model.ips.get().avg, 0)}` : 'after the first step'}
                loading={live.loading}
              />
            </div>
            <div class="tile">
              <Stat label="Time left" value={timeLeft.get()} sub={timeCap.get()} />
            </div>
          </KpiStrip>
        </div>
        <div class="chart">
          {/* The chart panel's header band (design/UNIFIED.md §3 Frame): the title and its qualifier, a hairline under
              it, then the legend (the readout) and the plot in its well. */}
          <div class="chart-head">
            <h3 class="chart-title">
              {scores.get() ? scores.get() : <Skeleton inline w="72px" h="12px" />} mAP <span class="faint">by epoch</span>
            </h3>
          </div>
          <div class="chart-body">
            {live.loading ? (
              <ChartSkeleton height={chartH.get()} />
            ) : (
              <LineChart
                series={series.get()}
                height={chartH.get()}
                xDomain={model.xDomain.get()}
                xInteger
                band
                future={model.future.get()}
                yDomain={fit.get()?.domain ?? [0, 1]}
                yTickValues={fit.get()?.ticks}
                yDecimals={fit.get()?.decimals}
                yLabel={zoomed.get() ? UNIT_ZOOM_NOTE : undefined}
                valueFormat={(v) => axisMetric(v, fit.get()?.domain[1])}
                xTitle={(x) => `Epoch ${Math.round(x)}`}
                curve="monotone"
                markers={model.best.get()}
                legendBest={phoneQuery.current ? undefined : model.bestLegend(fit.get()?.domain[1])}
                endDots
                legendNote={phoneQuery.current ? undefined : latest.get() ? `Latest · ep ${latest.get()}` : 'Latest'}
                label={`${scores.get() ?? 'Evaluation'} mAP by epoch`}
                empty="Waiting for the first validation…"
              />
            )}
          </div>
        </div>
      </div>
      <MachineFoot run={r()} system={props.system} />
    </section>
  );
}

css`
  .card {
    position: relative;
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  /* 48 px: the name, its status, the meta line taking what is left, Open run; the rail on its bottom edge. */
  header {
    position: relative;
    display: flex;
    align-items: center;
    gap: var(--space-3);
    height: 48px;
    padding: 0 var(--space-2) 0 var(--space-4);
  }
  h2 {
    flex: 0 1 auto;
    min-width: 80px;
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
    letter-spacing: var(--type-lg-tracking);
  }
  h2 a:hover {
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 4px;
  }
  .meta {
    flex: 1 1 0;
    min-width: 0;
    overflow: hidden;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .cta {
    flex: none;
  }
  .rail {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 2px;
    background: var(--bg-inset);
    overflow: hidden;
  }
  .rail .fill {
    position: absolute;
    inset: 0;
    transform-origin: left;
    background: var(--accent-line);
    transition: transform var(--dur-slow) var(--ease-standard);
  }
  .rail.quiet .fill {
    background: var(--text-tertiary);
  }
  /* The tiles on the left third, the chart on the right two thirds: two flush regions, a hairline between them, no
     padding around either (design/UNIFIED.md §4 Overview). */
  .body {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
  }
  /* 2 × 2, cut by hairlines (ui/KpiStrip), filling the third top to bottom. */
  .kpis {
    --kpi-columns: repeat(2, minmax(0, 1fr));
    display: grid;
    min-width: 0;
    border-right: 1px solid var(--line-subtle);
  }
  .tile {
    min-width: 0;
  }
  .chart {
    min-width: 0;
    display: flex;
    flex-direction: column;
  }
  /* As ui/Panel's header: 40 px, the title in 13 / 500 and its qualifier in 12 px tertiary, a hairline under it. */
  .chart-head {
    display: flex;
    align-items: center;
    min-height: var(--control-xl);
    padding: var(--space-1) var(--space-3) var(--space-1) var(--space-4);
    border-bottom: 1px solid var(--line-subtle);
  }
  .chart-title {
    font-size: var(--type-base-size);
    line-height: var(--type-base-line);
    letter-spacing: var(--type-base-tracking);
    font-weight: var(--weight-medium);
    white-space: nowrap;
  }
  .chart-title .faint {
    margin-left: var(--space-1);
    font-size: var(--type-sm-size);
    letter-spacing: 0;
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
  }
  .chart-body {
    flex: 1;
    padding: var(--space-4);
  }
  /* Below 1280 px: one row of four tiles over the chart, a hairline between them. */
  @media (max-width: 1279px) {
    .body {
      grid-template-columns: minmax(0, 1fr);
    }
    .kpis {
      --kpi-columns: repeat(4, minmax(0, 1fr));
      border-right: 0;
      border-bottom: 1px solid var(--line-subtle);
    }
  }
  /* Tablets and phones: the meta line (with the device) on a row of its own under the name, never cut. */
  @media (max-width: 1023px) {
    header {
      flex-wrap: wrap;
      height: auto;
      min-height: 48px;
      padding: var(--space-2) var(--space-2) var(--space-2) var(--space-4);
      gap: var(--space-1) var(--space-3);
    }
    .meta {
      order: 3;
      flex-basis: 100%;
    }
    .cta {
      margin-left: auto;
    }
  }
  /* Phones: the tiles 2 × 2 again over the chart. */
  @media (max-width: 767px) {
    header {
      padding: var(--space-2) var(--space-1) var(--space-2) var(--space-3);
      gap: var(--space-1) var(--space-2);
    }
    .kpis {
      --kpi-columns: repeat(2, minmax(0, 1fr));
    }
    .chart-body {
      padding: var(--space-3);
    }
  }
`;
