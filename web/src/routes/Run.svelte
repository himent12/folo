<script lang="ts">
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import Box from '@lucide/svelte/icons/box';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import Square from '@lucide/svelte/icons/square';
  import Trash from '@lucide/svelte/icons/trash-2';
  import X from '@lucide/svelte/icons/x';
  import WifiOff from '@lucide/svelte/icons/wifi-off';
  import { hub } from '$lib/api/index.svelte';
  import { LiveRun } from '$lib/api/live.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import { ago, deviceName, eta } from '$lib/format';
  import { origin } from '$lib/lineage';
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
  import Outcome from './run/Outcome.svelte';
  import PerClass from './run/PerClass.svelte';
  import RunMetrics from './run/RunMetrics.svelte';
  import Samples from './run/Samples.svelte';
  import { scheduleWarnings } from './run/warnings';

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
  // The trainer's "too short to learn" warnings, from the log (./run/warnings): one callout, never a toast.
  const warnings = $derived.by(() => {
    void live.version;
    return scheduleWarnings(live.logs);
  });

  type Section = 'samples' | 'classes' | 'logs' | 'config';
  // Below 960 px one section shows at a time. A queued run has nothing to sample yet, so it opens on its Config (the
  // dependency and start estimate are in the banner above); once picked, the choice holds.
  let picked = $state<Section | null>(null);
  const section = $derived<Section>(picked ?? (queuedOnly ? 'config' : 'samples'));
  const models = new Resource((s) => hub.api.models(s));
  /** Where the run started: "From scratch", or the model / file it was fine-tuned from (one reading, $lib/lineage). */
  const lineage = $derived.by(() => {
    if (!run) return '';
    const o = origin(run.config.weights, models.data ?? []);
    return o.kind === 'scratch' ? 'From scratch' : `Fine-tuned from ${o.label}`;
  });
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
    // Phones: the queued run's one action keeps its one name, in the menu (the status and its time keep the row).
    if (run.status === 'queued') out.push({ label: 'Remove from queue', icon: X, danger: true, onclick: () => (stopOpen = true) });
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
      toast.success(`Resuming ${r.name}`, r.epoch > 0 ? `From the checkpoint of epoch ${r.epoch}.` : 'No epoch had finished, so it starts over.');
    } catch (e) {
      toast.error(e, 'Could not resume the run');
    } finally {
      resuming = false;
    }
  }

</script>

<!-- A queued run leads with what it waits for and when it should start (the callout under the header); the frames it
     will fill stay compact until there is data, and the config is in its rail. -->
{#snippet waiting()}
  {#if ahead}
    Starts when <a class="link" href={href(`/runs/${ahead.id}`)}>{ahead.name}</a> finishes{#if ahead.eta_s !== null}, in about {eta(ahead.eta_s)}{/if}.
  {:else}
    Starts as soon as the device is free.
  {/if}
{/snippet}

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
    <!-- The metrics instrument (epoch, six KPI tabs, one chart), then the samples grid beside the per-class and config
         rail. -->
    <div class="sk-metrics">
      <div class="sk-strip">
        <div class="sk-cell sk-epoch"><Skeleton w="88px" h="32px" /><Skeleton h="4px" /><Skeleton w="160px" h="12px" /></div>
        <div class="sk-tabs">
          {#each [0, 1, 2, 3, 4, 5] as i (i)}<div class="sk-cell"><Skeleton w="56%" h="12px" /><Skeleton w="48%" h="24px" /><Skeleton w="68%" h="12px" /></div>{/each}
        </div>
      </div>
      <div class="sk-chart"><ChartSkeleton height={320} /></div>
    </div>
    <div class="sk-pick"><Skeleton w="280px" h="28px" r="var(--radius-md)" /></div>
    <div class="sk-detail">
      <div class="sk-panel">
        <div class="sk-ph"><Skeleton w="128px" h="14px" /><span class="sk-ctl"><Skeleton w="160px" h="24px" r="var(--radius-md)" /><Skeleton w="120px" h="24px" r="var(--radius-md)" /></span></div>
        <div class="sk-pb">
          <div class="sk-bar"><Skeleton w="min(300px, 60%)" h="12px" /><Skeleton w="200px" h="12px" /></div>
          <div class="sk-tiles">{#each [0, 1, 2, 3, 4, 5, 6, 7, 8] as i (i)}<Skeleton h="auto" class="sk-tile" r="var(--radius-md)" />{/each}</div>
        </div>
      </div>
      <div class="sk-rail">
        <div class="sk-panel">
          <div class="sk-ph"><Skeleton w="72px" h="14px" /></div>
          <div class="sk-rows">
            {#each [0, 1, 2, 3, 4, 5, 6, 7] as i (i)}
              <div class="sk-row"><Skeleton w="8px" h="8px" r="2px" /><Skeleton w="{56 + ((i * 23) % 40)}px" h="12px" /><Skeleton w="40px" h="12px" /><Skeleton h="4px" /></div>
            {/each}
          </div>
        </div>
        <div class="sk-panel">
          <div class="sk-ph"><Skeleton w="56px" h="14px" /></div>
          <div class="sk-rows">
            {#each [0, 1, 2, 3, 4, 5] as i (i)}
              <div class="sk-row cfg"><Skeleton w="{64 + ((i * 17) % 32)}px" h="12px" /><Skeleton w="{36 + ((i * 29) % 40)}px" h="12px" /></div>
            {/each}
          </div>
        </div>
      </div>
    </div>
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
      <span class="cfg lineage">{lineage}</span>
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
        <!-- Phones: in the ⋯ menu under the same name, so the status and its time keep one line beside it. -->
        <span class="wide-only"><Button variant="ghost" icon={X} onclick={() => (stopOpen = true)}>Remove from queue</Button></span>
      {:else if !active}
        <span class="wide-only"><Button variant="ghost" icon={Trash} aria-label="Delete run" title="Delete run" onclick={() => (deleteOpen = true)} /></span>
      {/if}
      <!-- Phones: the state action stays visible, everything else goes behind one overflow button. -->
      <span class="phone-only"><Menu label="More run actions" items={moreItems} /></span>
    {/snippet}

    <Outcome {run} {elapsed} {resuming} onresume={resume} waiting={queuedOnly ? waiting : undefined} {warnings} />

    <!-- Phones: while another section is picked, the chart folds to its one-line readout so that section starts in
         the first screen. -->
    <RunMetrics
      {run}
      steps={live.steps}
      epochs={live.epochs}
      version={live.version}
      loading={live.loading}
      queued={queuedOnly}
      folded={section !== 'samples'}
    />

    <!-- Evidence on the left (what the model sees, what the trainer says); reference in a sticky rail on the right. -->
    <!-- Below 960 px one of these shows at a time, chosen here; wider screens show them all beside a rail. -->
    <div class="phone-pick">
      <Segmented
        label="Show"
        bind:value={() => section, (v) => (picked = v)}
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
      <Button variant="destructive" icon={run.status === 'queued' ? X : Square} loading={busy} onclick={stop}>{run.status === 'queued' ? 'Remove from queue' : 'Stop run'}</Button>
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
    grid-template-columns: minmax(0, 1fr);
    border-bottom: 1px solid var(--line-subtle);
  }
  .sk-tabs {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
  }
  /* The same row geometry as the loaded cells (RunMetrics), so the chart does not move when data arrives. */
  .sk-cell {
    display: grid;
    grid-template-rows: 16px 40px 16px;
    align-items: center;
    gap: 6px;
    padding: var(--space-4);
    border-left: 1px solid var(--line-subtle);
  }
  /* The epoch row over the tabs, at every width. */
  .sk-epoch {
    grid-template-columns: 88px minmax(0, 1fr) 160px;
    grid-template-rows: 32px;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-4);
    border-left: 0;
    border-bottom: 1px solid var(--line-subtle);
  }
  .sk-tabs .sk-cell:first-child {
    border-left: 0;
  }
  .sk-chart {
    padding: var(--space-4);
  }
  .sk-pick {
    display: none;
  }
  .sk-detail {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: start;
  }
  .sk-rail {
    display: grid;
    gap: var(--space-4);
  }
  .sk-panel {
    background: var(--bg-panel);
    border: 1px solid var(--line-subtle);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  .sk-ph {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    height: 40px;
    padding: 0 var(--space-3) 0 var(--space-4);
  }
  .sk-ctl {
    display: flex;
    gap: var(--space-2);
  }
  .sk-pb {
    display: grid;
    gap: var(--space-3);
    padding: 0 var(--space-4) var(--space-4);
  }
  .sk-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    height: 24px;
  }
  /* The loaded grid's geometry: tiles at least 152 px wide, 4:3, 8 px apart. */
  .sk-tiles {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(152px, 1fr));
    gap: var(--space-2);
  }
  .sk-tiles :global(.sk-tile) {
    aspect-ratio: 4 / 3;
  }
  /* Full rows only, as loaded: eight tiles in four or two columns, nine in three (laptops). */
  .sk-tiles :global(.sk-tile:nth-child(9)) {
    display: none;
  }
  .sk-rows {
    display: grid;
    padding-bottom: var(--space-2);
  }
  /* Per-class rows (swatch, name, value, bar) and config rows (label, value), 28 px like the loaded tables. */
  .sk-row {
    display: grid;
    grid-template-columns: 8px minmax(0, 1fr) 40px 30%;
    align-items: center;
    gap: var(--space-2);
    height: 28px;
    padding: 0 var(--space-4);
  }
  .sk-row.cfg {
    grid-template-columns: auto auto;
    justify-content: space-between;
  }
  @media (max-width: 1279px) {
    .sk-tabs .sk-cell {
      grid-template-rows: 16px 32px 16px;
      gap: var(--space-1);
      padding: var(--space-3);
    }
    .sk-tiles :global(.sk-tile:nth-child(9)) {
      display: block;
    }
  }
  @media (max-width: 1023px) {
    .sk-tabs {
      display: none;
    }
    .sk-pick {
      display: block;
    }
    .sk-detail {
      grid-template-columns: minmax(0, 1fr);
    }
    .sk-rail {
      display: none;
    }
    .sk-tiles :global(.sk-tile:nth-child(9)) {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .loading {
      padding: var(--space-4);
    }
    .sk-ctl :global(:first-child) {
      display: none;
    }
    .sk-tiles {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  .id {
    font-size: var(--type-sm-size);
  }
  .when-phone {
    display: none;
    font-size: var(--type-sm-size);
    white-space: nowrap;
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
  @media (max-width: 1023px) {
    /* Tablets and phones: the rail no longer fits beside the samples, so one section at a time, picked above. */
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
