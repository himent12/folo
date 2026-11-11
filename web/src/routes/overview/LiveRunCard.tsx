import ArrowRight from '$icons/arrow-right';
import { bindable, computed, css, effect, For, MediaQuery, signal } from 'fw';
import ChartSkeleton from '$lib/charts/ChartSkeleton';
import { LiveRun } from '$lib/api/live';
import type { Dataset, Run } from '$lib/api/types';
import LineChart, { type Marker } from '$lib/charts/LineChart';
import { BestPulse } from '$lib/charts/bestPulse';
import { unitDomain, unitZoomed, UNIT_ZOOM_NOTE } from '$lib/charts/domain';
import { hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import { progressText, runProgress } from '$lib/api/progress';
import { lr, metric, num } from '$lib/format';
import { etaText, lastKnown, Now } from '$lib/api/freshness';
import StatusBadge, { statusWord } from '$lib/ui/StatusBadge';
import { deviceParts } from './device';
import { starting, STARTING } from './slot';
import { runEvalSource, scoresTitle } from '$lib/evaluated';
import { href } from '$lib/router';
import Button from '$lib/ui/Button';
import Progress from '$lib/ui/Progress';
import Skeleton from '$lib/ui/Skeleton';
import Stat from '$lib/ui/Stat';
import { media } from '$lib/design/tokens';
import Ident from '$lib/ui/Ident';
import Meta from '$lib/ui/Meta';

export interface LiveRunCardProps {
  run: Run;
  dataset?: Dataset | null;
  /** The dataset as people know it (./datasetLabel): undefined while loading (a placeholder), null to leave it out. */
  datasetName?: string | null;
  lost?: number | null;
}

/**
 * `run`: the run in the training slot (./slot), training or starting; its stream keeps the card live. `lost` (bindable) reports when the run's stream went quiet (`LiveRun.lost`), so the page can tell a server that stopped
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
  // is qualified and the live marks (the badge's heartbeat, the line's lime end) stop (DESIGN_DIRECTION §2.8).
  effect(() => {
    lost.set(live.lost);
  });
  const now = new Now(() => live.lost !== null);

  // Phones: a shorter plot, so the recent runs below start on the first screen.
  const phoneQuery = new MediaQuery(media.phone, false);
  const narrow = () => phoneQuery.current;

  const charts = computed(() => {
    void live.version;
    return {
      map: [
        // One run, so ink rather than a hue (hue identifies entities): the focal metric solid with the page's one area
        // fill; the secondary a 1 px tertiary dash, so the focal series reads first.
        { id: 'm95', label: 'mAP50-95', color: 'var(--text-primary)', area: true, end: live.lost === null ? ('live' as const) : undefined, points: live.epochs.map((e) => ({ x: e.epoch, y: e.val.map50_95 })) },
        { id: 'm50', label: 'mAP50', color: 'var(--text-tertiary)', pattern: 'dash' as const, thin: true, points: live.epochs.map((e) => ({ x: e.epoch, y: e.val.map50 })) },
      ],
      lastStep: live.steps[live.steps.length - 1],
    };
  });

  // "So far": the axis ends at the last validated epoch, so the curve always reaches the right edge (no unexplained
  // empty stretch for a run that is a third done; the epoch bar beside it shows how far the whole run is).
  const lastEpoch = computed(() => {
    void live.version;
    return live.epochs[live.epochs.length - 1]?.epoch ?? 0;
  });
  // The y axis fits the run's scores (../../lib/charts/domain): 0 up to just over the best, never a mostly empty 0–1;
  // a run still near 0 gets a zoomed axis whose caption names the full scale. Epochs arrive one at a time, and the top
  // moves in whole steps, so it re-fits at most once per epoch. The plot is the chart's height less its 30 px margins.
  const chartH = computed(() => (narrow() ? 140 : 200));
  const fit = computed(() => {
    const hi = Math.max(-Infinity, ...charts.get().map.flatMap((s) => s.points.map((p) => p.y)).filter(Number.isFinite));
    return Number.isFinite(hi) ? unitDomain(hi, chartH.get() - 30) : undefined;
  });
  const zoomed = computed(() => unitZoomed(fit.get()));
  const horizon = computed<[number, number]>(() => [0, Math.max(1, lastEpoch.get())]);
  const progress = computed(() => runProgress(run.get()));

  // The focal chart's vocabulary, as on the run page (run/RunMetrics): the best epoch a hollow ink ring (the title's
  // "○ best"), pulsed once when a new best lands, and the line's end the live mark (a filled lime dot, the heartbeat).
  const pulser = new BestPulse();
  const pulse = signal(0);
  effect.pre(() => {
    const r = run.get();
    pulse.set(pulser.see(r.best?.epoch ?? null, r.status === 'running'));
  });
  const best = computed<Marker[]>(() => {
    const b = run.get().best;
    return b ? [{ x: b.epoch, y: b.map50_95, label: 'best', quiet: true, pulse: pulse.get() }] : [];
  });

  // The device this run is using, live: its own label, plus load only from the GPU it actually trains on.
  const sys = new Resource((s) => hub.api.system(s), { poll: 2000 });
  // The GPU's load is a live reading: once the server stops answering it is dropped, not shown as current.
  const deviceItems = computed(() => deviceParts(run.get(), sys.stale === undefined ? (sys.data?.gpus ?? []) : []));

  const r = () => run.get();

  // "Training now" is the status badge and the progress bar; the card's edge stays a plain hairline.
  return (
    <section class="card" aria-labelledby="live-title">
      <header>
        <div class="title-row">
          {starting(r()) ? (
            // Taken by the worker or next with the slot free (./slot): about to train, not waiting.
            <StatusBadge state={STARTING} lastKnown={live.lost !== null ? lastKnown(STARTING.label, live.lost, now.current) : undefined} />
          ) : (
            <StatusBadge run={r()} lastKnown={live.lost !== null ? lastKnown(statusWord(r()), live.lost, now.current) : undefined} />
          )}{' '}
          <h2 id="live-title">
            <a href={href(`/runs/${r().id}`)}>{r().name}</a>
          </h2>
        </div>{' '}
        <p class="meta">
          {/* Whole items wrap to a second line rather than lose the last one (DESIGN_DIRECTION §2.3). */}
          <Meta>
            {props.datasetName === undefined ? (
              <span>
                &#8203;<Skeleton inline w="72px" h="10px" />
              </span>
            ) : props.datasetName ? (
              <span>{props.datasetName}</span>
            ) : null}{' '}
            <Ident value={r().config.arch} />{' '}
            <span class="num">{r().config.imgsz} px</span>{' '}
            <span class="num">batch {r().config.batch}</span>{' '}
            <For each={deviceItems.get()} key={(part) => part}>
              {(part) => <span class="dev num">{part()}</span>}
            </For>
          </Meta>
        </p>{' '}
        <div class="cta">
          <Button size="sm" variant="ghost" iconRight={ArrowRight} href={href(`/runs/${r().id}`)}>
            Open run
          </Button>
        </div>
      </header>{' '}
      <div class="body">
        <div class="stats">
          <div class="epoch">
            <Stat label="Epoch" value={num(r().epoch)} unit={`/ ${r().epochs}`} size="lg" />{' '}
            {/* Lime is "now": a last known progress is drawn in ink. */}
            <Progress value={progress.get()} size="sm" label="Training progress" tone={live.lost !== null ? 'neutral' : 'accent'} />{' '}
            <p class="eta num">
              <span>{progressText(r())}</span>{' '}
              <span>{etaText(r().eta_s, live.lost, now.current)}</span>
            </p>
          </div>{' '}
          <div class="grid">
            {/* The best epoch's score; the chart's legend reads the latest one, and each says which it is. */}
            <Stat label="mAP50-95" value={metric(r().best?.map50_95)} sub={r().best ? `best · ep ${r().best!.epoch}` : 'best, after the first epoch'} />{' '}
            <Stat label="Throughput" value={num(charts.get().lastStep?.imgs_per_s, 0)} unit="img/s" />{' '}
            <Stat label="Learning rate" value={lr(charts.get().lastStep?.lr)} />{' '}
            <Stat label="Loss" value={num(charts.get().lastStep?.loss.total, 3)} sub="last step, raw" />
          </div>
        </div>{' '}
        <div class="chart">
          {/* "Latest · ep N" is said once: in the legend's note where the legend has room for it; on phones in the title,
              so the legend holds only the two series and their values, on one row. */}
          <p class="chart-title">
            {scores.get() ? scores.get() : <Skeleton inline w="72px" h="12px" />} mAP{' '}
            <span class="faint">
              {narrow() && lastEpoch.get() ? `Latest · ep ${lastEpoch.get()}` : `by epoch, so far${r().best ? ` · ○ best at ep ${r().best!.epoch}` : ''}`}
            </span>
          </p>{' '}
          {live.loading ? (
            <ChartSkeleton height={chartH.get()} />
          ) : (
            <LineChart
              series={charts.get().map}
              height={chartH.get() + (zoomed.get() ? 16 : 0)}
              xDomain={horizon.get()}
              xInteger
              yDomain={fit.get()?.domain ?? [0, 1]}
              yTickValues={fit.get()?.ticks}
              yDecimals={fit.get()?.decimals}
              yLabel={zoomed.get() ? UNIT_ZOOM_NOTE : undefined}
              valueFormat={(v) => metric(v)}
              xTitle={(x) => `Epoch ${Math.round(x)}`}
              curve="monotone"
              markers={best.get()}
              endDots
              legendNote={narrow() ? undefined : lastEpoch.get() ? `Latest · ep ${lastEpoch.get()}` : 'Latest'}
              label={`${scores.get() ?? 'Evaluation'} mAP by epoch`}
              empty="Waiting for the first validation…"
            />
          )}
        </div>
      </div>
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
  header {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 2px var(--space-3);
    padding: var(--space-4) var(--space-3) var(--space-3) var(--space-5);
  }
  /* The name wraps under a long status ("Last known: Running · 2m ago" on a phone) rather than be cut to "f…". */
  .title-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-1) var(--space-3);
    min-width: 0;
  }
  h2 {
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
    letter-spacing: var(--type-lg-tracking);
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  h2 a:hover {
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 4px;
  }
  .meta {
    grid-column: 1;
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .cta {
    grid-column: 2;
    grid-row: 1 / span 2;
    align-self: start;
  }
  .body {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
    gap: var(--space-5);
    padding: var(--space-2) var(--space-5) var(--space-5);
  }
  .stats {
    display: grid;
    gap: var(--space-5);
    align-content: start;
  }
  .epoch {
    display: grid;
    gap: var(--space-2);
  }
  .eta {
    display: flex;
    justify-content: space-between;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-4) var(--space-3);
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
  @media (max-width: 1023px) {
    .body {
      grid-template-columns: minmax(0, 1fr);
    }
    .stats {
      grid-template-columns: minmax(200px, 1fr) 2fr;
      align-items: start;
    }
    .grid {
      grid-template-columns: repeat(4, 1fr);
    }
  }
  @media (max-width: 767px) {
    /* Phones: the two stats that matter while watching (best mAP, throughput); the run page has the rest. */
    .grid > :global(:nth-child(n + 3)) {
      display: none;
    }
    /* Phones: progress, then the mAP chart, then the stat pair, so evidence comes before detail. */
    .stats {
      display: contents;
    }
    /* The run page's instrument: the epoch on the left, its bar and the time left beside it on the value's line. */
    .epoch {
      order: 1;
      grid-template-columns: auto minmax(0, 1fr);
      grid-template-rows: 1fr auto auto;
      column-gap: var(--space-4);
      row-gap: var(--space-1);
    }
    .epoch > :global(.stat) {
      grid-row: 1 / -1;
    }
    .epoch > :global(.track) {
      grid-row: 2;
      grid-column: 2;
    }
    .eta {
      grid-row: 3;
      grid-column: 2;
      padding-bottom: var(--space-1);
    }
    .chart {
      order: 2;
    }
    /* The supporting pair under a hairline, values on one baseline. */
    .grid {
      order: 3;
      grid-template-columns: 1fr 1fr;
      gap: var(--space-3);
      padding-top: var(--space-3);
      border-top: 1px solid var(--line-subtle);
    }
    header {
      padding: var(--space-3) var(--space-3) var(--space-2) var(--space-4);
    }
    /* The meta line runs under the CTA too: beside it, "dataset · arch · 640 px · batch 16" left "batch 16" alone on a
       second line at 390 px (round 23). Open run stays on the title's row. */
    .cta {
      grid-row: 1;
    }
    .meta {
      grid-column: 1 / -1;
    }
    .meta :global(.dev),
    .meta > span:last-child:not(:first-child) {
      display: none;
    }
    .body {
      grid-template-columns: 1fr;
      gap: var(--space-4);
      padding: var(--space-1) var(--space-4) var(--space-4);
    }
  }
`;
