<script lang="ts">
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import Box from '@lucide/svelte/icons/box';
  import CircleCheck from '@lucide/svelte/icons/circle-check';
  import Clock from '@lucide/svelte/icons/clock';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import Play from '@lucide/svelte/icons/play';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import Square from '@lucide/svelte/icons/square';
  import Trash from '@lucide/svelte/icons/trash-2';
  import X from '@lucide/svelte/icons/x';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import WifiOff from '@lucide/svelte/icons/wifi-off';
  import { hub } from '$lib/api/index.svelte';
  import { LiveRun } from '$lib/api/live.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import { ago, deviceName, duration, eta, metric } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Badge from '$lib/ui/Badge.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Menu, { type MenuItem } from '$lib/ui/Menu.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import ConfigPanel from './run/ConfigPanel.svelte';
  import Logs from './run/Logs.svelte';
  import PerClass from './run/PerClass.svelte';
  import RunMetrics from './run/RunMetrics.svelte';
  import Samples from './run/Samples.svelte';

  const id = $derived(router.route.params.id ?? '');
  const live = new LiveRun(() => id || null);
  const datasets = new Resource((s) => hub.api.datasets(s));
  const run = $derived(live.run);
  const dataset = $derived(run ? datasets.data?.find((d) => d.id === run.config.dataset_id) : undefined);

  $effect(() => {
    setCrumbs({ label: 'Runs', path: '/runs' }, { label: run?.name ?? id });
  });

  const others = new Resource((s) => hub.api.runs(s), { poll: 5000 });
  const queuedOnly = $derived(run?.status === 'queued' && live.steps.length === 0);
  const ahead = $derived(others.data?.find((r) => r.status === 'running' || r.status === 'stopping'));
  const active = $derived(run ? ['queued', 'running', 'stopping'].includes(run.status) : false);
  const epochsDone = $derived.by(() => {
    void live.version;
    return live.epochs.length;
  });
  const elapsed = $derived(run?.started ? ((run.finished ? Date.parse(run.finished) : Date.now()) - Date.parse(run.started)) / 1000 : null);

  let section = $state<'samples' | 'classes' | 'logs' | 'config'>('samples');
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

  const moreItems = $derived.by((): MenuItem[] => {
    if (!run) return [];
    const out: MenuItem[] = [{ label: 'Compare', icon: GitCompare, href: href(`/runs/compare?ids=${run.id}`) }];
    if (run.model_id) {
      if (run.status !== 'completed') out.push({ label: 'Try model', icon: ScanSearch, href: href(`/playground?model=${run.model_id}`) });
      out.push({ label: 'Open model', icon: Box, href: href(`/models/${run.model_id}`) });
    }
    if (!active) out.push({ label: 'Delete run', icon: Trash, danger: true, onclick: () => (deleteOpen = true) });
    return out;
  });

  // Resume: the same run continues from weights/last.folo (POST /api/runs with resume_run_id, docs/API.md).
  let resuming = $state(false);
  async function resume() {
    if (!run) return;
    resuming = true;
    try {
      const id = run.id;
      const r = await live.resume(() => hub.api.resumeRun(id));
      toast.success(`Resuming ${r.name}`, r.epoch > 0 ? `From the checkpoint of epoch ${r.epoch}.` : 'No epoch had finished, so it starts from scratch.');
    } catch (e) {
      toast.error(e, 'Could not resume the run');
    } finally {
      resuming = false;
    }
  }

  const retryHref = $derived(run ? href(`/runs/new?from=${run.id}${run.error?.includes('memory') ? `&batch=${Math.max(1, run.config.batch / 2)}` : ''}`) : '#');
</script>

{#if live.error && !run}
  <Page title="Run">
    <ErrorState error={live.error} title="Could not load this run" retry={() => live.retry()} />
    <EmptyState title="Check the run id" description="It may have been deleted; every run is listed on the Runs page.">
      {#snippet actions()}<Button href={href('/runs')}>All runs</Button>{/snippet}
    </EmptyState>
  </Page>
{:else if !run}
  <!-- Mirrors the real layout (header, summary strip, chart rows) so nothing jumps when data arrives. -->
  <div class="loading" aria-busy="true" aria-label="Loading run">
    <div class="sk-head">
      <Skeleton w="140px" h="20px" r="var(--radius-full)" />
      <Skeleton w="240px" h="28px" />
      <Skeleton w="min(420px, 100%)" h="16px" />
    </div>
    <!-- The metrics instrument (strip + one chart), then samples beside the rail. -->
    <div class="sk-metrics">
      <div class="sk-strip">
        {#each [0, 1, 2, 3, 4, 5, 6] as i (i)}<div class="sk-cell"><Skeleton w="56%" h="12px" /><Skeleton w="48%" h="28px" /><Skeleton w="68%" h="10px" /></div>{/each}
      </div>
      <div class="sk-chart"><ChartSkeleton height={320} /></div>
    </div>
    <div class="sk-row2"><Skeleton h="300px" r="var(--radius-lg)" /><Skeleton h="300px" r="var(--radius-lg)" /></div>
  </div>
{:else}
  <Page title={run.name} inlineActions>
    {#snippet lead()}
      <StatusBadge status={run.status} />
      <span class="mono faint id">{run.id}</span>
      <!-- Phones: the time joins the status, so the meta line below fits one row. -->
      <span class="faint when-phone">{run.started ? `started ${ago(run.started)}` : `created ${ago(run.created)}`}</span>
      {#if live.connection === 'reconnecting'}<Badge tone="warning"><WifiOff size={12} /> Reconnecting</Badge>{/if}
    {/snippet}
    {#snippet sub()}
      <a class="link" href={href(`/datasets/${run.config.dataset_id}`)}>{dataset?.name ?? run.config.dataset_id}</a>
      <span class="mono">{run.config.arch}</span>
      <!-- Phones: size, batch and device are in the Config section; the id is in the URL. -->
      <span class="cfg">{run.config.imgsz} px · batch {run.config.batch}</span>
      <span class="truncate dev" title={run.device}>{deviceName(run.device)}</span>
      <span class="faint when">{run.started ? `started ${ago(run.started)}` : `created ${ago(run.created)}`}</span>
    {/snippet}
    {#snippet actions()}
      <span class="wide-only">
        <Button icon={GitCompare} href={href(`/runs/compare?ids=${run.id}`)}>Compare</Button>
        {#if run.model_id}
          {#if run.status !== 'completed'}<Button icon={ScanSearch} href={href(`/playground?model=${run.model_id}`)}>Try model</Button>{/if}
          <Button icon={Box} href={href(`/models/${run.model_id}`)}>Model</Button>
        {/if}
      </span>
      {#if run.status === 'running'}
        <Button variant="danger" icon={Square} onclick={() => (stopOpen = true)}>Stop</Button>
      {:else if run.status === 'queued'}
        <Button variant="ghost" icon={X} onclick={() => (stopOpen = true)}>Remove from queue</Button>
      {:else if !active}
        <span class="wide-only"><Button variant="ghost" icon={Trash} aria-label="Delete run" title="Delete run" onclick={() => (deleteOpen = true)} /></span>
      {/if}
      <!-- Phones: the state action stays visible, everything else goes behind one overflow button. -->
      <span class="phone-only"><Menu label="More run actions" items={moreItems} /></span>
    {/snippet}

    {#if run.status === 'failed'}
      <div class="banner danger" role="alert">
        <TriangleAlert size={16} />
        <div class="b-text">
          <p class="b-title">Failed at epoch {run.epoch}</p>
          <p class="b-msg">{run.error}</p>
        </div>
        <div class="b-acts">
          <!-- Out of memory needs a smaller batch (a new run); anything else can continue from the last checkpoint. -->
          {#if !run.error?.includes('memory')}<Button size="sm" icon={Play} loading={resuming} onclick={resume}>Resume</Button>{/if}
          <Button size="sm" icon={RotateCcw} href={retryHref}>{run.error?.includes('memory') ? `Retry with batch ${Math.max(1, run.config.batch / 2)}` : 'New run like this'}</Button>
        </div>
      </div>
    {:else if run.status === 'completed'}
      <div class="banner success">
        <CircleCheck size={16} />
        <div class="b-text">
          <p class="b-title">Completed in {duration(elapsed)}</p>
          <p class="b-msg">Best mAP50-95 {metric(run.best?.map50_95)} at epoch {run.best?.epoch ?? 0}. The best weights are saved as a model.</p>
        </div>
        {#if run.model_id}<Button size="sm" icon={ScanSearch} href={href(`/playground?model=${run.model_id}`)}>Try in Playground</Button>{/if}
      </div>
    {:else if run.status === 'stopped'}
      <div class="banner neutral">
        <Square size={16} />
        <div class="b-text">
          <p class="b-title">Stopped at epoch {run.epoch}</p>
          <p class="b-msg">The last checkpoint was kept{run.model_id ? ' and saved as a model' : ''}. Resume continues from it.</p>
        </div>
        <Button size="sm" icon={Play} loading={resuming} onclick={resume}>Resume</Button>
      </div>
    {/if}

    {#if queuedOnly}
      <!-- Nothing has run yet: one clear waiting state instead of empty charts. -->
      <section class="queued" aria-label="Queued">
        <div class="q-mark" aria-hidden="true"><Clock size={20} strokeWidth={1.5} /></div>
        <h2>Queued{#if ahead}<span class="faint">&nbsp;· next in line</span>{/if}</h2>
        <p>
          {#if ahead}
            Starts when <a class="link" href={href(`/runs/${ahead.id}`)}>{ahead.name}</a> finishes{#if ahead.eta_s !== null}, in about {eta(ahead.eta_s)}{/if}.
          {:else}
            Waiting for the device. It starts as soon as it is free.
          {/if}
          Charts, samples and logs appear with the first training step.
        </p>
        <div class="q-facts num">
          <span><b>{run.epochs}</b> epochs</span>
          <span><b>{run.config.batch}</b> batch</span>
          <span><b>{run.config.imgsz}</b> px</span>
          <span class="mono">{run.config.arch}</span>
        </div>
      </section>
    {:else}
    <RunMetrics {run} steps={live.steps} epochs={live.epochs} version={live.version} loading={live.loading} />

    <!-- Evidence on the left (what the model sees, what the trainer says); reference in a sticky rail on the right. -->
    <!-- Up to 1199 px one of these shows at a time, chosen here; wider screens show them all beside a rail. -->
    <div class="phone-pick">
      <Segmented
        label="Show"
        bind:value={section}
        options={[
          { value: 'samples', label: 'Samples' },
          { value: 'classes', label: 'Classes' },
          { value: 'logs', label: 'Logs' },
          { value: 'config', label: 'Config' },
        ]}
      />
    </div>
    <div class="detail" data-show={section}>
      <div class="main-col">
        <div class="part" data-part="samples"><Samples runId={run.id} classes={dataset?.classes ?? []} {epochsDone} /></div>
        <div class="part" data-part="logs"><Logs logs={live.logs} version={live.version} live={active} /></div>
      </div>
      <aside class="rail" aria-label="Per-class results and config">
        <div class="part" data-part="classes"><PerClass classes={dataset?.classes ?? []} epochs={live.epochs} version={live.version} loading={live.loading} compact /></div>
        <div class="part" data-part="config"><ConfigPanel config={run.config} datasetName={dataset?.name} compact /></div>
      </aside>
    </div>
    {/if}
    {#if queuedOnly}<ConfigPanel config={run.config} datasetName={dataset?.name} />{/if}
  </Page>

  <Dialog
    bind:open={stopOpen}
    title={run.status === 'queued' ? `Remove ${run.name} from the queue?` : `Stop ${run.name}?`}
    description={run.status === 'queued'
      ? 'It has not started yet, so nothing is lost. You can start it again from New run.'
      : 'Training stops after the current step. The last checkpoint is kept and saved as a model.'}
    size="sm"
  >
    {#snippet footer()}
      <Button onclick={() => (stopOpen = false)}>{run.status === 'queued' ? 'Keep it queued' : 'Keep training'}</Button>
      <Button variant="destructive" icon={run.status === 'queued' ? X : Square} loading={busy} onclick={stop}>{run.status === 'queued' ? 'Remove' : 'Stop run'}</Button>
    {/snippet}
  </Dialog>
  <Dialog bind:open={deleteOpen} title="Delete {run.name}?" description="Removes the run and its metric history from Studio. Checkpoint files on disk are not touched." size="sm">
    {#snippet footer()}
      <Button onclick={() => (deleteOpen = false)}>Cancel</Button>
      <Button variant="destructive" icon={Trash} loading={busy} onclick={remove}>Delete run</Button>
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
  .sk-head {
    display: grid;
    gap: var(--space-2);
    max-width: 100%;
  }
  .sk-metrics {
    background: var(--bg-panel);
    border: 1px solid var(--line-subtle);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  .sk-strip {
    display: grid;
    grid-template-columns: 240px repeat(6, minmax(0, 1fr));
    border-bottom: 1px solid var(--line-subtle);
  }
  .sk-cell {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-4);
    border-left: 1px solid var(--line-subtle);
  }
  .sk-cell:first-child {
    border-left: 0;
  }
  .sk-chart {
    padding: var(--space-4);
  }
  @media (max-width: 1199px) {
    .sk-strip {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .sk-cell:first-child {
      display: none;
    }
  }
  .sk-row2 {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: var(--space-4);
  }
  @media (max-width: 767px) {
    .loading {
      padding: var(--space-4);
    }
    .sk-row2 {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  .id {
    font-size: var(--type-xs-size);
  }
  .when-phone {
    display: none;
    font-size: var(--type-sm-size);
  }
  .wide-only {
    display: contents;
  }
  .phone-only {
    display: none;
  }
  @media (max-width: 767px) {
    .wide-only {
      display: none;
    }
    .phone-only {
      display: contents;
    }
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
  .b-acts {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
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
  .neutral {
    background: var(--bg-panel);
    border-color: var(--line-default);
    color: var(--text-secondary);
  }
  .banner .b-title {
    color: var(--text-primary);
  }

  .queued {
    display: grid;
    justify-items: center;
    gap: var(--space-3);
    padding: var(--space-12) var(--space-6);
    text-align: center;
    background: var(--bg-panel);
    border: 1px solid var(--line-default);
    border-radius: var(--radius-lg);
  }
  .q-mark {
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--bg-active);
    color: var(--text-secondary);
  }
  .queued h2 {
    font-size: var(--type-lg-size);
    line-height: var(--type-lg-line);
  }
  .queued p {
    max-width: 460px;
    color: var(--text-secondary);
    text-wrap: balance;
  }
  .q-facts {
    display: flex;
    gap: var(--space-4);
    flex-wrap: wrap;
    justify-content: center;
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .q-facts b {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .detail {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: start;
  }
  .main-col,
  .rail {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  .rail {
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
  }
  .part {
    display: grid;
    min-width: 0;
  }
  .phone-pick {
    display: none;
  }
  @media (max-width: 1199px) {
    /* Up to 1199 px the rail no longer fits beside the samples: one section at a time, picked above. */
    /* The section picker stays reachable while the section scrolls (just under the top bar). */
    .phone-pick {
      display: block;
      position: sticky;
      top: calc(var(--size-topbar) + var(--space-2));
      z-index: var(--z-sticky);
      margin-bottom: calc(-1 * var(--space-1));
    }
    /* One part is visible at a time, so the empty columns must not add gaps. */
    .detail,
    .main-col,
    .rail {
      gap: 0;
    }
    .detail:not([data-show='samples']) [data-part='samples'],
    .detail:not([data-show='classes']) [data-part='classes'],
    .detail:not([data-show='logs']) [data-part='logs'],
    .detail:not([data-show='config']) [data-part='config'] {
      display: none;
    }
    .detail {
      grid-template-columns: minmax(0, 1fr);
    }
    .rail {
      position: static;
    }
  }
  @media (max-width: 767px) {
    .phone-pick :global(.seg) {
      display: flex;
      width: 100%;
    }
    .phone-pick :global(.seg > *) {
      flex: 1;
      justify-content: center;
    }
    .b-msg {
      font-size: var(--type-sm-size);
    }
    .banner {
      flex-direction: column;
      align-items: stretch;
      gap: var(--space-3);
    }
    /* Success is a one-line summary with its action; the details are in the tiles below. */
    .banner.success {
      flex-direction: row;
      align-items: center;
      padding: var(--space-2) var(--space-2) var(--space-2) var(--space-3);
    }
    .banner.success .b-msg {
      display: none;
    }
    .banner > :global(svg) {
      display: none;
    }
    .dev,
    .cfg,
    .id,
    .when {
      display: none;
    }
    .when-phone {
      display: inline;
    }
  }
</style>
