<script lang="ts">
  import { MediaQuery } from 'svelte/reactivity';
  import ChartSkeleton from '$lib/charts/ChartSkeleton.svelte';
  import Box from '@lucide/svelte/icons/box';
  import ChevronsUp from '@lucide/svelte/icons/chevrons-up';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import ScanSearch from '@lucide/svelte/icons/scan-search';
  import Square from '@lucide/svelte/icons/square';
  import Trash from '@lucide/svelte/icons/trash-2';
  import X from '@lucide/svelte/icons/x';
  import WifiOff from '@lucide/svelte/icons/wifi-off';
  import { hub } from '$lib/api/index.svelte';
  import { LiveRun } from '$lib/api/live.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import { hasFeature, ordinal, queuePlace, withQueueMove } from '$lib/api/queue';
  import type { QueueRequest } from '$lib/api/types';
  import { media } from '$lib/design/tokens';
  import { ago, deviceName, eta } from '$lib/format';
  import { origin } from '$lib/lineage';
  import { href, router } from '$lib/router.svelte';
  import { setCrumbs } from '$lib/shell/crumbs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Badge from '$lib/ui/Badge.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import { DialogStatus } from '$lib/ui/feedback.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import Menu, { type MenuItem } from '$lib/ui/Menu.svelte';
  import Tabs from '$lib/ui/Tabs.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import ConfigPanel from './run/ConfigPanel.svelte';
  import { HEAD_LABEL } from './run/resolved';
  import Logs from './run/Logs.svelte';
  import Outcome from './run/Outcome.svelte';
  import Callout from './run/Callout.svelte';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { fallbackNotice, runEvalSource, scoresTitle } from '$lib/evaluated';
  import PerClass from './run/PerClass.svelte';
  import Lineage from '$lib/ui/Lineage.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import RunMetrics from './run/RunMetrics.svelte';
  import Samples from './run/Samples.svelte';
  import { moveQueued, queueItems, REMOVE_DESCRIPTION, REMOVE_LABEL, removeTitle, startHint, startLabel } from './run/queueActions';
  import { scheduleWarnings } from './run/warnings';
  import Copy from '@lucide/svelte/icons/copy';
  import Play from '@lucide/svelte/icons/play';
  import { isOom } from '$lib/oom';
  import { copyText, pageCommands, type PageCommand } from '$lib/shell/pageCommands.svelte';

  const id = $derived(router.route.params.id ?? '');
  const live = new LiveRun(() => id || null);
  const datasets = new Resource((s) => hub.api.datasets(s));
  const run = $derived(live.run);
  const dataset = $derived(run ? datasets.data?.find((d) => d.id === run.config.dataset_id) : undefined);
  // A dataset with no validation images trains and validates on its training split (docs/API.md, Split fallback): the
  // page says so once, as the Model page does, and its titles name that split rather than "Validation".
  const evalOn = $derived(runEvalSource(dataset));
  const splitNotice = $derived(evalOn ? fallbackNotice(evalOn) : null);

  $effect(() => {
    setCrumbs({ label: 'Runs', path: '/runs' }, { label: run?.name ?? id });
  });

  const others = new Resource((s) => hub.api.runs(s), { poll: 5000 });
  const queuedOnly = $derived(run?.status === 'queued' && live.steps.length === 0);
  const ahead = $derived(others.data?.find((r) => r.status === 'running' || r.status === 'stopping'));

  // Queue control (servers that list `queue_control`, docs/API.md): the run's place, Start next, Move up / down.
  const system = new Resource((s) => hub.api.system(s));
  const queueCtl = $derived(hasFeature(system.data, 'queue_control'));
  // The run's `folo train` command on the server's machine (servers that list `run_cli`): Config shows it under "In
  // PowerShell:", and the palette copies it. Keyed by run, so another run's command never shows while it loads.
  const cli = new Resource(async (s) => {
    const rid = id;
    if (!rid || !hasFeature(system.data, 'run_cli')) return null;
    return { id: rid, cli: (await hub.api.runCli(rid, s)).cli };
  });
  const runCli = $derived(cli.data && cli.data.id === id ? cli.data.cli : null);
  const place = $derived(run?.status === 'queued' && queueCtl ? (run.queue_position ?? null) : null);
  /** The runs waiting, front first (this one among them). */
  const queue = $derived(
    (others.data ?? []).filter((r) => r.status === 'queued' && r.queue_position != null).sort((a, b) => (a.queue_position ?? 0) - (b.queue_position ?? 0)),
  );
  const waitingCount = $derived(Math.max(live.waiting ?? 0, queue.length, place ?? 0));
  /** The waiting run just in front of this one. */
  const before = $derived(place && place > 1 ? queue.find((r) => r.id !== run?.id && r.queue_position === place - 1) : undefined);
  /** Start next is a move only when something is in front (or nothing trains, and it starts at once). */
  const canStart = $derived(place !== null && (place > 1 || !ahead));

  /** Moves this run in the queue: the new order at once (here and in the queue list), the old one back on a refusal. */
  async function move(req: QueueRequest) {
    const cur = run;
    if (!cur) return;
    const id = cur.id;
    const refresh = () => {
      void others.refresh();
      hub.api.run(id).then(
        (r) => {
          if (live.run?.id === id) live.run = { ...live.run, status: r.status, started: r.started, queue_position: r.queue_position ?? null, resolved: r.resolved ?? live.run.resolved };
        },
        () => {},
      );
    };
    const r = await moveQueued(
      cur,
      req,
      () => {
        const list = others.data;
        const prev = live.run?.queue_position ?? null;
        if (list) others.set(withQueueMove(list, id, req));
        const next = others.data?.find((x) => x.id === id)?.queue_position;
        if (live.run?.id === id && next != null) live.run = { ...live.run, queue_position: next };
        return () => {
          if (list) others.set(list);
          if (live.run?.id === id) live.run = { ...live.run, queue_position: prev };
        };
      },
      refresh,
      ahead,
    );
    if (!r) return;
    if (live.run?.id === id) live.run = { ...live.run, queue_position: r.queue_position ?? null };
    void others.refresh();
  }

  // A queued run waits in a layout of its own at every width: its place and the way to the front, what it waits for,
  // then its config; the outputs it will have are one quiet list instead of empty frames. Phones show the callout and
  // the config alone: no section picker over sections with nothing behind them, no folded chart that opens empty.
  const waitingLayout = $derived(queuedOnly);
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
  /** Below 1024 px the parts are one section at a time behind tabs: then the parts' frame is their tab panel. */
  const stacked = new MediaQuery(media.upToLaptop, false);
  const phone = new MediaQuery(media.phone, false);
  const lineage = $derived(run ? origin(run.config.weights, models.data ?? []) : null);
  let stopOpen = $state(false);
  let deleteOpen = $state(false);
  let busy = $state(false);
  // A confirm that fails says so in its dialog, which stays open for Retry (the toaster sits behind the modal); one
  // that succeeds closes it and the page says what happened.
  const stopStatus = new DialogStatus();
  const deleteStatus = new DialogStatus();

  async function stop() {
    if (!run) return;
    busy = true;
    try {
      const r = await hub.api.stopRun(run.id);
      live.run = r;
      stopOpen = false;
      toast.info(`Stopping ${r.name}`, `The checkpoint from epoch ${r.epoch} is kept.`);
    } catch (e) {
      stopStatus.error(e, run.status === 'queued' ? 'Could not remove the run from the queue' : 'Could not stop the run');
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
      deleteStatus.error(e, 'Could not delete the run');
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
    // Phones: the queue actions in their one order (Start next is the banner's button), Remove from queue last under the
    // same name (the status and its time keep the row).
    const removeQueued = () => (stopOpen = true);
    if (place !== null) out.push(...queueItems(place, waitingCount, ahead, move, { start: false, remove: removeQueued }));
    else if (run.status === 'queued') out.push({ label: REMOVE_LABEL, icon: X, danger: true, onclick: removeQueued });
    if (!active) out.push({ label: 'Delete run', icon: Trash, danger: true, onclick: () => (deleteOpen = true) });
    return out;
  });

  /** Wider screens: the queued run's moves behind ⋯; Remove from queue is the button beside it, Start next the banner's. */
  const wideQueueItems = $derived(place !== null ? queueItems(place, waitingCount, ahead, move, { start: false }) : []);

  /** What the run will have once it trains, in place of the empty frames (the waiting layout). */
  const later = $derived([
    { name: 'Metrics', what: 'mAP, precision and recall, loss, LR and throughput, by epoch' },
    { name: 'Samples', what: `Predictions on 16 ${scoresTitle(evalOn).toLowerCase()} images after every epoch` },
    { name: 'Per class', what: `AP for each of ${dataset?.classes.length ?? 'its'} classes` },
    { name: 'Logs', what: "The trainer's output, live" },
  ]);

  // Resume: the same run continues from weights/last.folo (POST /api/runs with resume_run_id, docs/API.md).
  // The palette's This page: the state action the header shows (Stop, Resume), Compare and the id.
  pageCommands((): PageCommand[] => {
    const r = run;
    if (!r) return [];
    const out: PageCommand[] = [];
    if (r.status === 'running') out.push({ id: 'run-stop', label: `Stop ${r.name}`, match: ['Stop run'], icon: Square, run: () => (stopOpen = true) });
    // As the run's banner: anything but out of memory continues from the last checkpoint.
    if (r.status === 'stopped' || (r.status === 'failed' && !isOom(r.error)))
      out.push({ id: 'run-resume', label: `Resume ${r.name}`, match: ['Resume run'], icon: Play, run: () => void resume() });
    out.push({ id: 'run-compare', label: `Compare ${r.name}`, match: ['Compare run'], icon: GitCompare, run: () => router.go(`/runs/compare?ids=${r.id}`) });
    out.push({ id: 'run-copy-id', label: 'Copy run id', match: [r.id], icon: Copy, run: () => void copyText(r.id, 'run id') });
    const line = runCli;
    if (line) out.push({ id: 'run-copy-cli', label: 'Copy CLI command', match: ['folo train', 'PowerShell'], icon: Copy, run: () => void copyText(line, 'CLI command') });
    return out;
  });

  let resuming = $state(false);
  // page-level: Resume is a header and palette action, never inside a dialog.
  async function resume() {
    if (!run) return;
    resuming = true;
    try {
      const id = run.id;
      const r = await live.resume(() => hub.api.resumeRun(id));
      toast.success(`Resuming ${r.name}`, r.epoch > 0 ? `From the checkpoint of epoch ${r.epoch}.` : 'No epoch had finished, so it starts over.');
    } catch (e) {
      // page-level: as above.
      toast.error(e, 'Could not resume the run');
    } finally {
      resuming = false;
    }
  }

</script>

<!-- A queued run leads with what it waits for and when it should start (the callout under the header); the frames it
     will fill stay compact until there is data, and the config is in its rail. -->
{#snippet waiting()}
  {#if place !== null && place > 1}
    <!-- Behind other waiting runs: the one in front of it, then what trains now. -->
    Starts after {#if before}<a class="link" href={href(`/runs/${before.id}`)}>{before.name}</a>{#if place > 2}{' '}and {place - 2} more{/if}{:else}{place - 1} queued {place === 2 ? 'run' : 'runs'}{/if}.
    {#if ahead}<a class="link" href={href(`/runs/${ahead.id}`)}>{ahead.name}</a> is training{#if ahead.eta_s !== null}, about {eta(ahead.eta_s)} left{/if}.{/if}
  {:else if ahead}
    Starts when <a class="link" href={href(`/runs/${ahead.id}`)}>{ahead.name}</a> finishes{#if ahead.eta_s !== null}, in about {eta(ahead.eta_s)}{/if}.
  {:else}
    Starts as soon as the device is free.
  {/if}
{/snippet}

{#snippet startNext()}
  {#if canStart}
    <span class="start" use:tooltip={startHint(ahead)}>
      <Button size="sm" icon={ChevronsUp} onclick={() => move({ action: 'start' })}>{startLabel(ahead)}</Button>
    </span>
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
      <!-- The chart's own header row (title, range control) over its legend and plot, as RunMetrics lays them out. -->
      <div class="sk-chart">
        <div class="sk-ch"><Skeleton w="200px" h="14px" /><Skeleton w="104px" h="24px" r="var(--radius-md)" /></div>
        <ChartSkeleton height={320} />
      </div>
    </div>
    <div class="sk-pick"><Skeleton w="280px" h="28px" r="var(--radius-md)" /></div>
    <div class="sk-detail">
      <div class="sk-main">
        <div class="sk-panel">
          <div class="sk-ph"><Skeleton w="128px" h="14px" /><span class="sk-ctl"><Skeleton w="160px" h="24px" r="var(--radius-md)" /><Skeleton w="120px" h="24px" r="var(--radius-md)" /></span></div>
          <div class="sk-pb">
            <div class="sk-bar"><Skeleton w="min(300px, 60%)" h="12px" /><Skeleton w="200px" h="12px" /></div>
            <div class="sk-tiles">{#each [0, 1, 2, 3, 4, 5, 6, 7, 8] as i (i)}<Skeleton h="auto" class="sk-tile" r="var(--radius-md)" />{/each}</div>
          </div>
        </div>
        <!-- Logs under the samples (desktop and laptop), growing to the rail's foot like the loaded log. -->
        <div class="sk-panel sk-logs">
          <div class="sk-ph"><Skeleton w="56px" h="14px" /><Skeleton w="128px" h="24px" r="var(--radius-md)" /></div>
          <div class="sk-lines">
            {#each [0, 1, 2, 3, 4, 5, 6, 7, 8] as i (i)}<Skeleton w="{46 + ((i * 13) % 24)}%" h="10px" />{/each}
          </div>
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
      <Ident value={run.id} class="faint id" />
      <!-- Phones: the time joins the status, so the meta line below fits one row. -->
      <span class="faint when-phone">{run.started ? `started ${ago(run.started)}` : `created ${ago(run.created)}`}</span>
      {#if live.connection === 'reconnecting'}<Badge tone="warning"><WifiOff size={12} /> Reconnecting</Badge>{/if}
    {/snippet}
    {#snippet sub()}
      <Meta>
        <a class="link" href={href(`/datasets/${run.config.dataset_id}`)}>{dataset?.name ?? run.config.dataset_id}</a>
        <Ident value={run.config.arch} />
        <!-- Phones: size, batch and device are in the Config section; the id is in the URL. -->
        <span class="cfg num">{run.config.imgsz} px</span>
        <span class="cfg num">batch {run.config.batch}</span>
        <span class="cfg lineage">{#if !lineage || lineage.kind === 'scratch'}From scratch{:else}Fine-tuned from <Lineage origin={lineage} />{/if}</span>
        <!-- Only where they differ from the defaults (f32, the NMS head); Config always says both. -->
        {#if run.config.amp}<span class="cfg">Mixed precision</span>{/if}
        {#if run.config.head === 'o2o'}<span class="cfg">{HEAD_LABEL.o2o}</span>{/if}
        <span class="truncate dev" title={run.device}>{deviceName(run.device)}</span>
        <span class="faint when">{run.started ? `started ${ago(run.started)}` : `created ${ago(run.created)}`}</span>
      </Meta>
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
        <span class="wide-only">
          <!-- The quiet destructive control, styled as Stop is on a running run (DESIGN_DIRECTION 13). -->
          <Button variant="danger" icon={X} onclick={() => (stopOpen = true)}>{REMOVE_LABEL}</Button>
          {#if wideQueueItems.length}<Menu label="More run actions" items={wideQueueItems} />{/if}
        </span>
      {:else if !active}
        <span class="wide-only"><Button variant="ghost" icon={Trash} aria-label="Delete run" title="Delete run" onclick={() => (deleteOpen = true)} /></span>
      {/if}
      <!-- Phones: the state action stays visible, everything else goes behind one overflow button. -->
      <span class="phone-only"><Menu label="More run actions" items={moreItems} /></span>
    {/snippet}

    <Outcome
      {run}
      {elapsed}
      {resuming}
      onresume={resume}
      waiting={queuedOnly ? waiting : undefined}
      waitingTitle={place !== null ? queuePlace(place) : undefined}
      waitingActions={canStart ? startNext : undefined}
      {warnings}
      system={system.data ?? (system.error ? null : undefined)}
    />

    {#if splitNotice && !waitingLayout}
      <Callout tone="warning" icon={TriangleAlert} title={splitNotice.title} role="note" label="Evaluation split">{splitNotice.why}</Callout>
    {/if}

    {#if waitingLayout}
      <!-- Waiting: the config as asked for, beside the queue and what the run will have. Nothing here is an
           empty frame, and no panel stretches to its neighbour's height. Below 1024 px the queue leads; phones show
           the config's essentials, the rest folded (the work before the reference). -->
      <div class="waiting">
        <div class="w-main"><ConfigPanel config={run.config} resolved={run.resolved} datasetName={dataset?.name} weights={lineage} essential={phone.current} cli={runCli} /></div>
        <div class="w-side">
          {#if place !== null && queue.length}
            <Panel title="Queue" meta="{waitingCount} waiting" pad="none" id="queue">
              <ol class="queue">
                {#if ahead}
                  <li class="q-row">
                    <span class="q-n"><span class="live-dot" aria-hidden="true"></span><span class="sr-only">Training</span></span>
                    <a class="q-name" href={href(`/runs/${ahead.id}`)}>{ahead.name}</a>
                    <span class="q-meta num">{ahead.eta_s !== null ? `${eta(ahead.eta_s)} left` : 'training'}</span>
                  </li>
                {/if}
                {#each queue as q (q.id)}
                  {@const self = q.id === run.id}
                  <li class="q-row" class:self aria-current={self ? 'true' : undefined}>
                    <span class="q-n num" aria-label={ordinal(q.queue_position ?? 0)}>{q.queue_position}</span>
                    {#if self}<span class="q-name">{q.name}</span><span class="q-meta">this run</span>{:else}<a class="q-name" href={href(`/runs/${q.id}`)}>{q.name}</a><span class="q-meta"></span>{/if}
                  </li>
                {/each}
              </ol>
            </Panel>
          {/if}
          <Panel title="When training starts" pad="none" id="later">
            <ul class="later">
              {#each later as l (l.name)}<li><span class="l-name">{l.name}</span><span class="l-what">{l.what}</span></li>{/each}
            </ul>
          </Panel>
        </div>
      </div>
    {:else}
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
      {evalOn}
    />

    <!-- Evidence on the left (what the model sees, what the trainer says); reference in a rail on the right, the two
         columns ending together. -->
    <!-- Below 1024 px one of these shows at a time, chosen here; wider screens show them all beside a rail. They switch
         content, so they are underline tabs (DESIGN_DIRECTION §2.22), not a segmented control. -->
    <div class="phone-pick">
      <Tabs
        label="Run sections"
        idPrefix="run-sec"
        value={section}
        onchange={(v) => (picked = v)}
        tabs={[
          { value: 'samples', label: 'Samples' },
          { value: 'classes', label: 'Classes' },
          { value: 'logs', label: 'Logs' },
          { value: 'config', label: 'Config' },
        ]}
      />
    </div>
    <div class="detail" data-show={section} id="run-sec-panel" role={stacked.current ? 'tabpanel' : undefined} aria-labelledby={stacked.current ? `run-sec-${section}` : undefined}>
      <div class="main-col">
        <div class="part" data-part="samples"><Samples runId={run.id} classes={dataset?.classes ?? []} {epochsDone} title="{scoresTitle(evalOn)} samples" /></div>
        <div class="part" data-part="logs"><Logs logs={live.logs} version={live.version} live={active} /></div>
      </div>
      <aside class="rail" aria-label="Per-class results and config">
        <div class="part" data-part="classes"><PerClass classes={dataset?.classes ?? []} epochs={live.epochs} version={live.version} loading={live.loading} compact /></div>
        <div class="part" data-part="config"><ConfigPanel config={run.config} resolved={run.resolved} datasetName={dataset?.name} weights={lineage} compact cli={runCli} /></div>
      </aside>
    </div>
    {/if}
  </Page>

  <Dialog
    bind:open={stopOpen}
    title={run.status === 'queued' ? removeTitle(run.name) : `Stop ${run.name}?`}
    description={run.status === 'queued'
      ? REMOVE_DESCRIPTION
      : 'Training stops after the current step. The last checkpoint is kept and saved as a model.'}
    size="sm"
    status={stopStatus}
  >
    {#snippet footer()}
      <Button onclick={() => (stopOpen = false)}>{run.status === 'queued' ? 'Keep it queued' : 'Keep training'}</Button>
      <Button variant="destructive" icon={run.status === 'queued' ? X : Square} loading={busy} onclick={stop}>{run.status === 'queued' ? REMOVE_LABEL : 'Stop run'}</Button>
    {/snippet}
  </Dialog>
  <Dialog bind:open={deleteOpen} title="Delete {run.name}?" description="Removes the run and its metric history from Studio. Checkpoint files on disk are not touched." size="sm" status={deleteStatus}>
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
    max-width: calc(var(--size-content-max) + 2 * var(--space-6));
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
    border: 1px solid var(--line-panel);
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
  /* RunMetrics' chart: 0 16 16 padding, a 48 px header row (8 px above the title), 8 px to the legend. */
  .sk-chart {
    display: grid;
    gap: var(--space-2);
    padding: 0 var(--space-4) var(--space-4);
  }
  .sk-ch {
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 48px;
    padding-top: var(--space-2);
  }
  .sk-pick {
    display: none;
  }
  /* The loaded detail row: samples over logs beside per-class over config, the columns ending together. */
  .sk-detail {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: stretch;
  }
  .sk-main,
  .sk-rail {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    gap: var(--space-4);
    min-width: 0;
  }
  .sk-lines {
    display: grid;
    align-content: start;
    gap: 8px;
    min-height: 200px;
    padding: var(--space-2) var(--space-4) var(--space-4);
  }
  .sk-panel {
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
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
    .sk-rail,
    .sk-logs {
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
  :global(.ident.id) {
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
  .start {
    display: inline-flex;
  }
  /* The waiting layout: the config (8 columns) beside the queue and the outputs to come (4), each panel its own
     height; stacked under 1024 px. */
  .waiting {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: start;
  }
  .w-side {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  .w-main {
    min-width: 0;
  }
  @media (max-width: 1023px) {
    .waiting {
      grid-template-columns: minmax(0, 1fr);
    }
    /* Tablets: the queue leads, then the config, then the outputs to come. */
    .w-side {
      display: contents;
    }
    .w-side :global(#queue) {
      order: -1;
    }
  }
  /* Phones: the callout (the place and what it waits for), the queue, then the config's essentials; what the run will
     have once it trains waits for that (Run's sections show it then). */
  @media (max-width: 767px) {
    .w-side :global(#later) {
      display: none;
    }
  }
  .queue,
  .later {
    list-style: none;
    margin: 0;
    padding: 0 0 var(--space-2);
  }
  .q-row {
    display: grid;
    grid-template-columns: 24px minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2);
    height: 36px;
    padding: 0 var(--space-4);
    border-top: 1px solid var(--line-subtle);
  }
  .q-row:first-child {
    border-top: 0;
  }
  /* Each name is a link: a 44 px row on touch screens. */
  @media (pointer: coarse) {
    .q-row {
      height: 44px;
    }
  }
  .q-n {
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
  }
  .live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-line);
  }
  .q-name {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text-primary);
  }
  a.q-name:hover {
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .q-row.self {
    background: var(--bg-active);
  }
  .q-row.self .q-name {
    font-weight: var(--weight-medium);
  }
  .q-meta {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    white-space: nowrap;
  }
  .later li {
    display: grid;
    gap: 2px;
    padding: var(--space-2) var(--space-4);
    border-top: 1px solid var(--line-subtle);
  }
  .later li:first-child {
    border-top: 0;
  }
  .l-name {
    font-weight: var(--weight-medium);
  }
  .l-what {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* The two columns end together: each column's last panel takes what the other column adds. Usually the rail is the
     taller, and the log (at least 200 px) grows to its bottom; under a short rail (few classes) Config's frame runs
     down to the log's end, as a properties rail runs the page's height. No fixed heights. */
  .detail {
    --log-h: 200px;
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: stretch;
  }
  .main-col,
  .rail {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    gap: var(--space-4);
    min-width: 0;
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
    /* Flex, not block: the segmented group is inline, and a line box would add its descender under it (16 px to the
       section at the tablet tier instead of 12). */
    .phone-pick {
      display: flex;
      position: sticky;
      top: var(--size-topbar);
      z-index: var(--z-sticky);
      margin-top: calc(-1 * var(--space-2));
      background: var(--bg-app);
      /* 12 px to the section under the page's 20 px gap (16 on phones, below). */
      margin-bottom: calc(var(--space-3) - var(--space-5));
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
    /* One section at a time: the log keeps its own height. */
    .detail {
      --log-h: 300px;
    }
  }
  @media (max-width: 767px) {
    .phone-pick {
      margin-bottom: calc(var(--space-3) - var(--space-4));
    }
    .phone-pick :global([role='tab']) {
      flex: 1;
      justify-content: center;
    }
    .dev,
    .cfg,
    .when,
    :global(.ident.id) {
      display: none;
    }
    .when-phone {
      display: inline;
    }
  }
</style>
