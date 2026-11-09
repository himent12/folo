<script lang="ts">
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import { LiveRun } from '$lib/api/live.svelte';
  import type { Dataset, Run } from '$lib/api/types';
  import LineChart from '$lib/charts/LineChart.svelte';
  import { runHorizon } from '$lib/charts/horizon';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import { runProgress, stepsPerEpoch } from '$lib/api/progress';
  import { eta, lr, metric, num } from '$lib/format';
  import { deviceLine } from './device';
  import { href } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Stat from '$lib/ui/Stat.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';

  /** `run: null` while the runs load: the same card with real labels and placeholder values, so nothing moves. */
  let { run: initial, dataset }: { run: Run | null; dataset?: Dataset } = $props();

  const runId = $derived(initial?.id ?? null);
  const live = new LiveRun(() => runId);
  const run = $derived(live.run ?? initial);

  let cursor = $state<number | null>(null);
  // Phones: a shorter plot, so the recent runs below start on the first screen.
  let narrow = $state(typeof matchMedia !== 'undefined' && matchMedia('(max-width: 767px)').matches);
  $effect(() => {
    const mq = matchMedia('(max-width: 767px)');
    const on = () => (narrow = mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  });

  const charts = $derived.by(() => {
    void live.version;
    return {
      map: [
        // One run, so ink rather than a hue (hue identifies entities): the focal metric solid with the page's one area
        // fill; the secondary a 1 px tertiary dash, so the focal series reads first.
        { id: 'm95', label: 'mAP50-95', color: 'var(--text-primary)', area: true, points: live.epochs.map((e) => ({ x: e.epoch, y: e.val.map50_95 })) },
        { id: 'm50', label: 'mAP50', color: 'var(--text-tertiary)', pattern: 'dash' as const, thin: true, points: live.epochs.map((e) => ({ x: e.epoch, y: e.val.map50 })) },
      ],
      lastStep: live.steps[live.steps.length - 1],
    };
  });

  const horizon = $derived.by(() => {
    void live.version;
    if (!run) return [0, 1] as [number, number];
    return runHorizon(Math.max(live.epochs.length, run.step / Math.max(1, stepsPerEpoch(run, live.steps))), run.epochs, true, 'observed');
  });
  const progress = $derived(run ? runProgress(run) : 0);

  // The device this run is using, live: its own label, plus load only from the GPU it actually trains on.
  const sys = new Resource((s) => hub.api.system(s), { poll: 2000 });
  const deviceText = $derived(run ? deviceLine(run, sys.data?.gpus ?? []) : '');
</script>

<!-- The lime hairline means "training now": only once a running run has loaded, never on the placeholder. -->
<section class="card" class:live={run?.status === 'running'} aria-labelledby={run ? 'live-title' : undefined} aria-label={run ? undefined : 'Loading training'} aria-busy={!run}>
  <header>
    <div class="title-row">
      {#if run}
        <StatusBadge status={run.status} />
        <h2 id="live-title"><a href={href(`/runs/${run.id}`)}>{run.name}</a></h2>
      {:else}
        <Skeleton w="80px" h="20px" r="var(--radius-full)" />
        <h2>&#8203;<Skeleton inline w="min(160px, 30vw)" h="16px" /></h2>
      {/if}
    </div>
    <p class="meta">
      {#if run}
        <span>{dataset?.name ?? run.config.dataset_id}</span><span class="sep">·</span>
        <span class="mono">{run.config.arch}</span><span class="sep">·</span>
        <span>{run.config.imgsz} px · batch {run.config.batch}</span><span class="sep">·</span>
        <span class="truncate num">{deviceText}</span>
      {:else}
        <!-- The trailing span takes the place of the device, which phones hide. -->
        <span>&#8203;<Skeleton inline w="min(280px, 48vw)" h="10px" /></span><span></span>
      {/if}
    </p>
    <div class="cta">
      {#if run}<Button size="sm" variant="ghost" iconRight={ArrowRight} href={href(`/runs/${run.id}`)}>Open run</Button>{:else}<Skeleton w="88px" h="var(--control-sm)" />{/if}
    </div>
  </header>

  <div class="body">
    <div class="stats">
      <div class="epoch">
        <Stat label="Epoch" value={run ? num(run.epoch) : ''} unit={run ? `/ ${run.epochs}` : undefined} size="lg" loading={!run} />
        <Progress value={progress} size="sm" label="Training progress" />
        <p class="eta num">
          {#if run}
            <span>{(progress * 100).toFixed(1)}%</span>
            <span>{run.eta_s !== null ? `${eta(run.eta_s)} left` : '—'}</span>
          {:else}
            <span>&#8203;<Skeleton inline w="40px" h="10px" /></span><span>&#8203;<Skeleton inline w="64px" h="10px" /></span>
          {/if}
        </p>
      </div>
      <div class="grid">
        <Stat label="Best mAP50-95" value={metric(run?.best?.map50_95)} sub={run?.best ? `epoch ${run.best.epoch}` : 'after first epoch'} loading={!run} />
        <Stat label="Throughput" value={num(charts.lastStep?.imgs_per_s, 0)} unit="img/s" loading={!run} />
        <Stat label="Learning rate" value={lr(charts.lastStep?.lr)} loading={!run} />
        <Stat label="Loss" value={num(charts.lastStep?.loss.total, 3)} sub="last step, unsmoothed" loading={!run} />
      </div>
    </div>
    <div class="chart">
      <p class="chart-title">Validation mAP <span class="faint">by epoch</span></p>
      {#if !run || live.loading}
        <!-- Phones: the legend (note and two values) wraps to two rows. -->
        <ChartSkeleton height={narrow ? 140 : 200} legendRows={narrow ? 2 : 1} />
      {:else}
        <LineChart
          series={charts.map}
          height={narrow ? 140 : 200}
          xDomain={horizon}
          xInteger
          yDomain={[0, 1]}
          valueFormat={(v) => metric(v)}
          xTitle={(x) => `Epoch ${Math.round(x)}`}
          curve="monotone"
          legendNote="Latest"
          label="Validation mAP by epoch"
          empty="Waiting for the first validation…"
        />
      {/if}
    </div>
  </div>
</section>

<style>
  .card {
    position: relative;
    background: var(--bg-panel);
    border: 1px solid var(--line-subtle);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  :global([data-theme='light']) .card {
    border-color: var(--line-default);
  }
  .card.live::before {
    content: '';
    position: absolute;
    inset: 0 0 auto 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-line) 30%, var(--accent-line) 70%, transparent);
    opacity: 0.55;
  }
  header {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 2px var(--space-3);
    padding: var(--space-4) var(--space-3) var(--space-3) var(--space-5);
  }
  .title-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
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
  .sep {
    color: var(--text-disabled);
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
    .meta span:last-child,
    .meta .sep:nth-last-child(2) {
      display: none;
    }
    .body {
      grid-template-columns: 1fr;
      gap: var(--space-4);
      padding: var(--space-1) var(--space-4) var(--space-4);
    }
  }
</style>
