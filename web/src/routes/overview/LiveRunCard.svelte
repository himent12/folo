<script lang="ts">
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import { LiveRun } from '$lib/api/live.svelte';
  import type { Dataset, Run } from '$lib/api/types';
  import LineChart from '$lib/charts/LineChart.svelte';
  import { runHorizon } from '$lib/charts/horizon';
  import { hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import { bytes, duration, lr, metric, num, pct } from '$lib/format';
  import { prefs } from '$lib/prefs.svelte';
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
    return {
      map: [
        // One run, so ink rather than a hue (hue identifies entities): primary metric solid, secondary tertiary dashed.
        { id: 'm95', label: 'mAP50-95', color: 'var(--text-primary)', area: true, points: live.epochs.map((e) => ({ x: e.epoch + 1, y: e.val.map50_95 })) },
        { id: 'm50', label: 'mAP50', color: 'var(--text-tertiary)', pattern: 'dash' as const, points: live.epochs.map((e) => ({ x: e.epoch + 1, y: e.val.map50 })) },
      ],
      lastStep: live.steps[live.steps.length - 1],
    };
  });

  const horizon = $derived.by(() => {
    void live.version;
    return runHorizon(Math.max(live.epochs.length, run.step / Math.max(1, run.steps_per_epoch)), run.epochs, true, 'observed');
  });
  const progress = $derived(run.step / Math.max(1, run.steps_per_epoch * run.epochs));

  // The device this run is using, live: one meta line instead of a separate system panel.
  const sys = new Resource((s) => hub.api.system(s), { poll: 2000 });
  const gpu = $derived(sys.data?.gpus[0]);
  const deviceLine = $derived.by(() => {
    const name = (gpu?.name ?? run.device).replace('NVIDIA GeForce ', '');
    if (!gpu || !sys.data) return name;
    const util = gpu.util === null ? null : `${pct(gpu.util, 0)} busy`;
    const mem = `${bytes(sys.data.memory.used_bytes, prefs.units)} / ${bytes(sys.data.memory.total_bytes, prefs.units)} RAM`;
    return [name, util, mem].filter(Boolean).join(' · ');
  });
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
      <span>{run.config.imgsz} px · batch {run.config.batch}</span><span class="sep">·</span>
      <span class="truncate num">{deviceLine}</span>
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
        <Stat label="Best mAP50-95" value={metric(run.best?.map50_95)} sub={run.best ? `epoch ${run.best.epoch + 1}` : 'after first epoch'} />
        <Stat label="Throughput" value={num(charts.lastStep?.imgs_per_s, 0)} unit="img/s" />
        <Stat label="Learning rate" value={lr(charts.lastStep?.lr)} />
        <Stat label="Loss" value={num(charts.lastStep?.loss.total, 3)} sub="last step, raw" />
      </div>
    </div>
    <div class="chart">
      <p class="chart-title">Validation mAP <span class="faint">by epoch</span></p>
      {#if live.loading}
        <Skeleton h="200px" r="var(--radius-md)" />
      {:else}
        <LineChart
          series={charts.map}
          height={200}
          xDomain={horizon}
          valueFormat={(v) => v.toFixed(3)}
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
    /* Phones: progress, then the mAP chart, then the stat grid, so evidence comes before detail. */
    .stats {
      display: contents;
    }
    .epoch {
      order: 1;
    }
    .chart {
      order: 2;
    }
    .grid {
      order: 3;
    }
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
