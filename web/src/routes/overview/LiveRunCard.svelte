<script lang="ts">
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import { LiveRun } from '$lib/api/live.svelte';
  import type { Dataset, Run } from '$lib/api/types';
  import { smooth } from '$lib/charts/downsample';
  import LineChart from '$lib/charts/LineChart.svelte';
  import { duration, lr, metric, num } from '$lib/format';
  import { href } from '$lib/router.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Progress from '$lib/ui/Progress.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Stat from '$lib/ui/Stat.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';

  let { run: initial, dataset }: { run: Run; dataset?: Dataset } = $props();

  const runId = $derived(initial.id);
  const live = new LiveRun(() => runId);
  const run = $derived(live.run ?? initial);

  let cursor = $state<number | null>(null);

  const charts = $derived.by(() => {
    void live.version;
    const spe = Math.max(1, run.steps_per_epoch);
    const raw = live.steps.map((s) => ({ x: s.step / spe, y: s.loss.total }));
    return {
      map: [
        { id: 'm95', label: 'mAP50-95', color: 'var(--chart-1)', area: true, points: live.epochs.map((e) => ({ x: e.epoch + 1, y: e.val.map50_95 })) },
        { id: 'm50', label: 'mAP50', color: 'var(--chart-5)', points: live.epochs.map((e) => ({ x: e.epoch + 1, y: e.val.map50 })) },
      ],
      loss: [
        { id: 'raw', label: 'raw', color: 'var(--chart-1)', ghost: true, points: raw },
        { id: 'loss', label: 'Train loss', color: 'var(--chart-1)', points: smooth(raw, 0.9) },
      ],
      lastStep: live.steps[live.steps.length - 1],
    };
  });

  const progress = $derived(run.step / Math.max(1, run.steps_per_epoch * run.epochs));
</script>

<section class="card" aria-labelledby="live-title">
  <header>
    <div class="title-row">
      <StatusBadge status={run.status} />
      <h2 id="live-title"><a href={href(`/runs/${run.id}`)}>{run.name}</a></h2>
    </div>
    <p class="meta">
      <span>{dataset?.name ?? run.config.dataset_id}</span><span class="sep">·</span>
      <span class="mono">{run.config.arch}</span><span class="sep">·</span>
      <span>{run.config.imgsz}px, batch {run.config.batch}</span><span class="sep">·</span>
      <span class="truncate">{run.device}</span>
    </p>
    <div class="cta"><Button size="sm" variant="ghost" iconRight={ArrowRight} href={href(`/runs/${run.id}`)}>Open run</Button></div>
  </header>

  <div class="body">
    <div class="stats">
      <div class="epoch">
        <Stat label="Epoch" value={num(run.epoch)} unit="/ {run.epochs}" size="lg" />
        <Progress value={progress} size="sm" label="Training progress" />
        <p class="eta num">
          <span>{(progress * 100).toFixed(1)}%</span>
          <span>{run.eta_s !== null ? `${duration(run.eta_s)} left` : '—'}</span>
        </p>
      </div>
      <div class="grid">
        <Stat label="Best mAP50-95" value={metric(run.best?.map50_95)} sub={run.best ? `epoch ${run.best.epoch + 1}` : 'after first epoch'} tone="accent" />
        <Stat label="Throughput" value={num(charts.lastStep?.imgs_per_s, 0)} unit="img/s" />
        <Stat label="Learning rate" value={lr(charts.lastStep?.lr)} />
        <Stat label="Loss" value={num(charts.lastStep?.loss.total, 3)} />
      </div>
    </div>
    <div class="chart">
      <p class="chart-title">Validation mAP</p>
      {#if live.loading}
        <Skeleton h="150px" r="var(--radius-md)" />
      {:else}
        <LineChart
          series={charts.map}
          height={150}
          xDomain={[0, run.epochs]}
          yFormat={(v) => v.toFixed(2)}
          valueFormat={(v) => v.toFixed(3)}
          xTitle={(x) => `Epoch ${Math.round(x)}`}
          curve="monotone"
          bind:cursor
          label="Validation mAP by epoch"
          empty="Waiting for the first validation…"
        />
      {/if}
    </div>
    <div class="chart">
      <p class="chart-title">Training loss</p>
      {#if live.loading}
        <Skeleton h="150px" r="var(--radius-md)" />
      {:else}
        <LineChart
          series={charts.loss}
          height={150}
          xDomain={[0, run.epochs]}
          yFormat={(v) => v.toFixed(1)}
          valueFormat={(v) => v.toFixed(3)}
          xTitle={(x) => `Epoch ${x.toFixed(1)}`}
          bind:cursor
          label="Training loss by epoch"
          legend={false}
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
  .card::before {
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
    grid-template-columns: minmax(220px, 260px) 1fr 1fr;
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
    font-size: var(--type-xs-size);
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
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
    font-weight: var(--weight-medium);
  }
  @media (max-width: 1199px) {
    .body {
      grid-template-columns: 1fr 1fr;
    }
    .stats {
      grid-column: 1 / -1;
      grid-template-columns: minmax(200px, 1fr) 2fr;
      align-items: start;
    }
    .grid {
      grid-template-columns: repeat(4, 1fr);
    }
  }
  @media (max-width: 767px) {
    header {
      padding: var(--space-4) var(--space-3) var(--space-3) var(--space-4);
    }
    .meta span:last-child,
    .meta .sep:nth-last-child(2) {
      display: none;
    }
    .body {
      grid-template-columns: 1fr;
      padding: var(--space-1) var(--space-4) var(--space-4);
    }
    .stats {
      grid-template-columns: 1fr;
    }
    .grid {
      grid-template-columns: 1fr 1fr;
    }
  }
</style>
