import ArrowRight from '$icons/arrow-right';
import { bindable, computed, css, effect, MediaQuery } from 'fw';
import ChartSkeleton from '$lib/charts/ChartSkeleton';
import { LiveRun } from '$lib/api/live';
import type { Dataset, Run } from '$lib/api/types';
import LineChart from '$lib/charts/LineChart';
import { UNIT_ZOOM_NOTE, unitZoomed } from '$lib/charts/domain';
import { ETA_PENDING, etaSettled, progressText, runProgress, trainingSeconds } from '$lib/api/progress';
import { cloudGpu } from '$lib/cloud';
import { deviceName, duration, eta, metric, num } from '$lib/format';
import { asOf, lastKnown, Now } from '$lib/api/freshness';
import StatusBadge, { statusWord } from '$lib/ui/StatusBadge';
import { starting, STARTING } from './slot';
import { runEvalSource, scoresTitle } from '$lib/evaluated';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Skeleton from '$lib/ui/Skeleton';
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
export default function LiveRunCard(props: LiveRunCardProps) {
  const lost = bindable(props, 'lost', null);
  // What the scores are on (lib/evaluated `scoresTitle`), one word in every state: until the dataset is known it is a
  // placeholder, never a guess that changes when it arrives ("Evaluation" in the skeleton, "Validation" loaded).
  const scores = computed(() => (props.dataset === undefined ? null : scoresTitle(runEvalSource(props.dataset))));

  const runId = computed(() => props.run.id);
  const live = new LiveRun(() => runId.get());
  const run = computed(() => live.run ?? props.run);
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
  const chartH = computed(() => (phoneQuery.current ? 186 : 246));
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
        <div class="kpis">
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
        </div>
        <div class="chart">
          {/* The chart section as on the run page: its title and qualifier over the legend (the readout) and the plot. */}
          <p class="chart-title">
            {scores.get() ? scores.get() : <Skeleton inline w="72px" h="12px" />} mAP <span class="faint">by epoch</span>
          </p>
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
              valueFormat={(v) => metric(v)}
              xTitle={(x) => `Epoch ${Math.round(x)}`}
              curve="monotone"
              markers={model.best.get()}
              legendBest={phoneQuery.current ? undefined : model.bestLegend.get()}
              endDots
              legendNote={phoneQuery.current ? undefined : latest.get() ? `Latest · ep ${latest.get()}` : 'Latest'}
              label={`${scores.get() ?? 'Evaluation'} mAP by epoch`}
              empty="Waiting for the first validation…"
            />
          )}
        </div>
      </div>
      <MachineFoot run={r()} />
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
  /* The tiles on the left third, the chart on the right two thirds. */
  .body {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
    gap: var(--space-5);
    padding: var(--space-4);
  }
  /* 2 × 2, cut by hairlines (1 px gaps over the line colour), as the run page's strip. */
  .kpis {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    grid-auto-rows: 1fr;
    gap: 1px;
    align-self: start;
    background: var(--line-subtle);
    border: 1px solid var(--line-subtle);
    border-radius: var(--radius-md);
    overflow: hidden;
  }
  .tile {
    padding: var(--space-3);
    background: var(--bg-panel);
    min-width: 0;
  }
  .chart {
    min-width: 0;
    display: grid;
    gap: var(--space-2);
    align-content: start;
  }
  .chart-title {
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-medium);
  }
  .chart-title .faint {
    font-size: var(--type-sm-size);
    font-weight: var(--weight-regular);
  }
  /* Below 1280 px: one row of four tiles over the chart. */
  @media (max-width: 1279px) {
    .body {
      grid-template-columns: minmax(0, 1fr);
      gap: var(--space-4);
    }
    .kpis {
      grid-template-columns: repeat(4, minmax(0, 1fr));
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
    .body {
      padding: var(--space-3);
    }
    .kpis {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
`;
