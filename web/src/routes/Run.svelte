<script lang="ts">
  import Box from '@lucide/svelte/icons/box';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Clock from '@lucide/svelte/icons/clock';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import Square from '@lucide/svelte/icons/square';
  import Trash from '@lucide/svelte/icons/trash-2';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import WifiOff from '@lucide/svelte/icons/wifi-off';
  import { hub } from '$lib/api/index.svelte';
  import { LiveRun } from '$lib/api/live.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import { ago, duration, lr, metric, num } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Badge from '$lib/ui/Badge.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import ConfigPanel from './run/ConfigPanel.svelte';
  import Logs from './run/Logs.svelte';
  import PerClass from './run/PerClass.svelte';
  import RunCharts from './run/RunCharts.svelte';
  import Samples from './run/Samples.svelte';

  const id = $derived(router.route.params.id ?? '');
  const live = new LiveRun(() => id || null);
  const datasets = new Resource((s) => hub.api.datasets(s));
  const run = $derived(live.run);
  const dataset = $derived(run ? datasets.data?.find((d) => d.id === run.config.dataset_id) : undefined);

  $effect(() => {
    setCrumbs({ label: 'Runs', path: '/runs' }, { label: run?.name ?? id });
  });

  const active = $derived(run ? ['queued', 'running', 'stopping'].includes(run.status) : false);
  const lastStep = $derived.by(() => {
    void live.version;
    return live.steps[live.steps.length - 1];
  });
  const lastEpoch = $derived.by(() => {
    void live.version;
    return live.epochs[live.epochs.length - 1];
  });
  const epochsDone = $derived.by(() => {
    void live.version;
    return live.epochs.length;
  });
  const progress = $derived(run ? Math.min(1, run.step / Math.max(1, run.steps_per_epoch * run.epochs)) : 0);
  const elapsed = $derived(run?.started ? ((run.finished ? Date.parse(run.finished) : Date.now()) - Date.parse(run.started)) / 1000 : null);
  const closeFrom = $derived(run ? 1 - run.config.augment.close_mosaic / run.epochs : 1);

  let stopOpen = $state(false);
  let deleteOpen = $state(false);
  let busy = $state(false);

  async function stop() {
    if (!run) return;
    busy = true;
    try {
      const r = await hub.api.stopRun(run.id);
      live.run = r;
      stopOpen = false;
      toast.info(`Stopping ${r.name}`, `The checkpoint from epoch ${r.epoch} is kept.`);
    } catch (e) {
      toast.error(e, 'Could not stop the run');
    } finally {
      busy = false;
    }
  }

  async function remove() {
    if (!run) return;
    busy = true;
    try {
      await hub.api.deleteRun(run.id);
      deleteOpen = false;
      toast.success(`Deleted ${run.name}`, 'Checkpoints on disk were left in place.');
      router.go('/runs');
    } catch (e) {
      toast.error(e, 'Could not delete the run');
    } finally {
      busy = false;
    }
  }

  const retryHref = $derived(run ? href(`/runs/new?from=${run.id}${run.error?.includes('memory') ? `&batch=${Math.max(1, run.config.batch / 2)}` : ''}`) : '#');
</script>

{#if live.error && !run}
  <Page title="Run">
    <ErrorState error={live.error} title="Could not load this run" retry={() => router.go(router.route.path, { replace: true })} />
    <EmptyState compact title="Check the run id" description="It may have been deleted. All runs are listed on the Runs page.">
      {#snippet actions()}<Button href={href('/runs')}>All runs</Button>{/snippet}
    </EmptyState>
  </Page>
{:else if !run}
  <div class="loading" aria-busy="true" aria-label="Loading run">
    <Skeleton w="120px" h="20px" />
    <Skeleton w="280px" h="28px" />
    <Skeleton h="96px" r="var(--radius-lg)" />
    <div class="sk-grid"><Skeleton h="260px" r="var(--radius-lg)" /><Skeleton h="260px" r="var(--radius-lg)" /></div>
  </div>
{:else}
  <Page title={run.name}>
    {#snippet lead()}
      <StatusBadge status={run.status} />
      <span class="mono faint id">{run.id}</span>
      {#if live.connection === 'reconnecting'}<Badge tone="warning"><WifiOff size={12} /> Reconnecting</Badge>{/if}
    {/snippet}
    {#snippet sub()}
      <a class="link" href={href(`/datasets/${run.config.dataset_id}`)}>{dataset?.name ?? run.config.dataset_id}</a>
      <span class="mono">{run.config.arch}</span>
      <span>{run.config.imgsz}px · batch {run.config.batch}</span>
      <span class="truncate dev">{run.device}</span>
      <span class="faint">{run.started ? `started ${ago(run.started)}` : `created ${ago(run.created)}`}</span>
    {/snippet}
    {#snippet actions()}
      <Button icon={GitCompare} href={href(`/runs/compare?ids=${run.id}`)}>Compare</Button>
      {#if run.model_id}
        <Button icon={ScanSearch} href={href(`/playground?model=${run.model_id}`)}>Try model</Button>
        <Button icon={Box} href={href(`/models/${run.model_id}`)}>Model</Button>
      {/if}
      {#if run.status === 'running' || run.status === 'queued'}
        <Button variant="danger" icon={Square} onclick={() => (stopOpen = true)}>Stop</Button>
      {:else if !active}
        <Button variant="ghost" icon={Trash} aria-label="Delete run" title="Delete run" onclick={() => (deleteOpen = true)} />
      {/if}
    {/snippet}

    {#if run.status === 'failed'}
      <div class="banner danger" role="alert">
        <TriangleAlert size={16} />
        <div class="b-text">
          <p class="b-title">Failed at epoch {run.epoch}</p>
          <p class="b-msg">{run.error}</p>
        </div>
        <Button size="sm" icon={RotateCcw} href={retryHref}>{run.error?.includes('memory') ? `Retry with batch ${Math.max(1, run.config.batch / 2)}` : 'Retry'}</Button>
      </div>
    {:else if run.status === 'completed'}
      <div class="banner success">
        <CircleCheck size={16} />
        <div class="b-text">
          <p class="b-title">Completed in {duration(elapsed)}</p>
          <p class="b-msg">Best mAP50-95 {metric(run.best?.map50_95)} at epoch {(run.best?.epoch ?? 0) + 1}. The best weights are saved as a model.</p>
        </div>
        {#if run.model_id}<Button size="sm" icon={ScanSearch} href={href(`/playground?model=${run.model_id}`)}>Try in Playground</Button>{/if}
      </div>
    {:else if run.status === 'queued'}
      <div class="banner info">
        <Clock size={16} />
        <div class="b-text">
          <p class="b-title">Queued</p>
          <p class="b-msg">Waiting for the device. It starts when the run ahead of it finishes.</p>
        </div>
      </div>
    {:else if run.status === 'stopped'}
      <div class="banner neutral">
        <Square size={16} />
        <div class="b-text">
          <p class="b-title">Stopped at epoch {run.epoch}</p>
          <p class="b-msg">The last checkpoint was kept{run.model_id ? ' and saved as a model' : ''}.</p>
        </div>
      </div>
    {/if}

    <section class="cluster" aria-label="Run summary">
      <div class="cell main">
        <span class="label">Epoch</span>
        <span class="big num">{run.epoch}<span class="of">/ {run.epochs}</span></span>
        <div class="track" aria-hidden="true">
          <div class="fill" class:done={run.status === 'completed'} class:bad={run.status === 'failed'} style:transform="scaleX({run.status === 'completed' ? 1 : progress})"></div>
          {#if closeFrom < 1}<div class="close" style:left="{closeFrom * 100}%" title="Mosaic off for the last {run.config.augment.close_mosaic} epochs"></div>{/if}
        </div>
        <span class="small num">
          {#if active}{(progress * 100).toFixed(1)}% · {run.eta_s !== null ? `${duration(run.eta_s)} left` : 'estimating…'}{:else}{duration(elapsed)} total{/if}
        </span>
      </div>
      <div class="cell">
        <span class="label">Best mAP50-95</span>
        <span class="val num accent">{metric(run.best?.map50_95)}</span>
        <span class="small">{run.best ? `epoch ${run.best.epoch + 1}` : '—'}</span>
      </div>
      <div class="cell">
        <span class="label">mAP50</span>
        <span class="val num">{metric(lastEpoch?.val.map50)}</span>
        <span class="small">last epoch</span>
      </div>
      <div class="cell">
        <span class="label">Precision · Recall</span>
        <span class="val num">{num(lastEpoch?.val.precision, 2)} <span class="dim">·</span> {num(lastEpoch?.val.recall, 2)}</span>
        <span class="small">last epoch</span>
      </div>
      <div class="cell">
        <span class="label">Loss</span>
        <span class="val num">{num(lastStep?.loss.total, 3)}</span>
        <span class="small num">box {num(lastStep?.loss.box, 2)} · cls {num(lastStep?.loss.cls, 2)}</span>
      </div>
      <div class="cell">
        <span class="label">Throughput</span>
        <span class="val num">{num(lastStep?.imgs_per_s, 0)}<span class="unit">img/s</span></span>
        <span class="small num">lr {lr(lastStep?.lr)}</span>
      </div>
    </section>

    <RunCharts {run} steps={live.steps} epochs={live.epochs} version={live.version} loading={live.loading} />

    <div class="two">
      <PerClass classes={dataset?.classes ?? []} epochs={live.epochs} version={live.version} loading={live.loading} />
      <Logs logs={live.logs} version={live.version} live={active} />
    </div>

    <Samples runId={run.id} classes={dataset?.classes ?? []} {epochsDone} />

    <ConfigPanel config={run.config} datasetName={dataset?.name} />
  </Page>

  <Dialog bind:open={stopOpen} title="Stop {run.name}?" description="Training stops after the current step. The last checkpoint is kept and saved as a model." size="sm">
    {#snippet footer()}
      <Button onclick={() => (stopOpen = false)}>Keep training</Button>
      <Button variant="danger" icon={Square} loading={busy} onclick={stop}>Stop run</Button>
    {/snippet}
  </Dialog>
  <Dialog bind:open={deleteOpen} title="Delete {run.name}?" description="Removes the run and its metric history from Studio. Checkpoint files on disk are not touched." size="sm">
    {#snippet footer()}
      <Button onclick={() => (deleteOpen = false)}>Cancel</Button>
      <Button variant="danger" icon={Trash} loading={busy} onclick={remove}>Delete run</Button>
    {/snippet}
  </Dialog>
{/if}

<style>
  .loading {
    display: grid;
    gap: var(--space-4);
    max-width: var(--size-content-max);
    margin: 0 auto;
    padding: var(--space-6);
  }
  .sk-grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: var(--space-4);
  }
  .id {
    font-size: var(--type-xs-size);
  }
  .link {
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .link:hover {
    text-decoration-color: var(--text-secondary);
  }
  .dev {
    max-width: 320px;
  }
  .banner {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-3) var(--space-3) var(--space-4);
    border-radius: var(--radius-lg);
    border: 1px solid;
  }
  .banner > :global(svg) {
    flex-shrink: 0;
    align-self: flex-start;
    margin-top: 2px;
  }
  .b-text {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 2px;
  }
  .b-title {
    font-weight: var(--weight-medium);
  }
  .b-msg {
    color: var(--text-secondary);
  }
  .danger {
    background: var(--danger-muted);
    border-color: color-mix(in srgb, var(--danger-solid) 25%, transparent);
    color: var(--danger-text);
  }
  .success {
    background: var(--success-muted);
    border-color: color-mix(in srgb, var(--success-solid) 25%, transparent);
    color: var(--success-text);
  }
  .info {
    background: var(--info-muted);
    border-color: color-mix(in srgb, var(--info-solid) 25%, transparent);
    color: var(--info-text);
  }
  .neutral {
    background: var(--bg-panel);
    border-color: var(--line-default);
    color: var(--text-secondary);
  }
  .banner .b-title {
    color: var(--text-primary);
  }

  .cluster {
    display: grid;
    grid-template-columns: 1.6fr repeat(5, minmax(0, 1fr));
    background: var(--bg-panel);
    border: 1px solid var(--line-subtle);
    border-radius: var(--radius-lg);
  }
  :global([data-theme='light']) .cluster {
    border-color: var(--line-default);
  }
  .cell {
    display: grid;
    gap: 4px;
    align-content: start;
    padding: var(--space-4);
    min-width: 0;
    border-left: 1px solid var(--line-subtle);
  }
  .cell.main {
    border-left: 0;
    gap: 6px;
  }
  .label {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .big {
    font-size: var(--type-3xl-size);
    line-height: var(--type-3xl-line);
    letter-spacing: var(--type-3xl-tracking);
    font-weight: var(--weight-semibold);
  }
  .of {
    margin-left: 6px;
    font-size: var(--type-lg-size);
    font-weight: var(--weight-regular);
    color: var(--text-tertiary);
    letter-spacing: 0;
  }
  .val {
    font-size: var(--type-xl-size);
    line-height: var(--type-xl-line);
    letter-spacing: var(--type-xl-tracking);
    font-weight: var(--weight-medium);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .val.accent {
    color: var(--accent-text);
  }
  .dim {
    color: var(--text-disabled);
  }
  .unit {
    margin-left: 3px;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    font-weight: var(--weight-regular);
    letter-spacing: 0;
  }
  .small {
    font-size: var(--type-xs-size);
    color: var(--text-tertiary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .track {
    position: relative;
    height: 6px;
    border-radius: 3px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .fill {
    position: absolute;
    inset: 0;
    transform-origin: left;
    background: var(--accent-line);
    border-radius: 3px;
    transition: transform var(--dur-slow) var(--ease-standard);
  }
  .fill.done {
    background: var(--success-solid);
  }
  .fill.bad {
    background: var(--danger-solid);
  }
  .close {
    position: absolute;
    top: 0;
    bottom: 0;
    right: 0;
    background: repeating-linear-gradient(135deg, transparent 0 3px, var(--line-strong) 3px 4px);
    border-left: 1px solid var(--line-strong);
  }
  .two {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: start;
  }
  @media (max-width: 1199px) {
    .cluster {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .cell.main {
      grid-column: 1 / -1;
      border-bottom: 1px solid var(--line-subtle);
    }
    .cell:nth-child(2),
    .cell:nth-child(5) {
      border-left: 0;
    }
    .cell:nth-child(n + 5) {
      border-top: 1px solid var(--line-subtle);
    }
    .two {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 767px) {
    .cluster {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .cell {
      padding: var(--space-3);
    }
    .cell:nth-child(n) {
      border-left: 0;
      border-top: 1px solid var(--line-subtle);
    }
    .cell.main {
      border-top: 0;
      border-bottom: 0;
    }
    .cell:nth-child(odd):not(.main) {
      border-left: 1px solid var(--line-subtle);
    }
    .val {
      font-size: var(--type-lg-size);
    }
    .banner {
      flex-wrap: wrap;
    }
    .dev {
      display: none;
    }
  }
</style>
