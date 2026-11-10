<script lang="ts">
  import { untrack } from 'svelte';
  import { MediaQuery } from 'svelte/reactivity';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import Copy from '@lucide/svelte/icons/copy';
  import GitCompare from '@lucide/svelte/icons/git-compare-arrows';
  import Pencil from '@lucide/svelte/icons/pencil';
  import Plus from '@lucide/svelte/icons/plus';
  import Repeat from '@lucide/svelte/icons/repeat';
  import Sparkles from '@lucide/svelte/icons/sparkles';
  import Square from '@lucide/svelte/icons/square';
  import Trash from '@lucide/svelte/icons/trash-2';
  import WifiOff from '@lucide/svelte/icons/wifi-off';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { LiveBench } from '$lib/api/liveBench.svelte';
  import { lastKnown, Now } from '$lib/api/freshness.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { BenchEntry, BenchLoop, BenchPlan } from '$lib/api/types';
  import { clonePlan, samePlan } from '$lib/bench/plan';
  import { isEntryActive, isLoopActive, planEditable } from '$lib/bench/rules';
  import { media } from '$lib/design/tokens';
  import { ago, date, num } from '$lib/format';
  import { href, router } from '$lib/router.svelte';
  import Page from '$lib/shell/Page.svelte';
  import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs.svelte';
  import { pageCommands, type PageCommand } from '$lib/shell/pageCommands.svelte';
  import Badge from '$lib/ui/Badge.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import { DialogStatus } from '$lib/ui/feedback.svelte';
  import Gate from '$lib/ui/Gate.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import InfoButton from '$lib/ui/InfoButton.svelte';
  import Menu, { type MenuItem } from '$lib/ui/Menu.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import Note from '$lib/ui/Note.svelte';
  import { isOffline, OFFLINE_REASON } from '$lib/ui/offline';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Tabs from '$lib/ui/Tabs.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import { AgentChoice } from './label/agentChoice.svelte';
  import AddEntryDialog, { type EntryKind } from './bench/AddEntryDialog.svelte';
  import DesignDialog from './bench/DesignDialog.svelte';
  import EditBenchDialog from './bench/EditBenchDialog.svelte';
  import Leaderboard from './bench/Leaderboard.svelte';
  import { boardOrder, comparePath, compareHint, compareReason, keepSelection } from './bench/leaderboard';
  import LoopsTab from './bench/LoopsTab.svelte';
  import NewLoopDialog from './bench/NewLoopDialog.svelte';
  import PlanTab from './bench/PlanTab.svelte';
  import { cannotBench } from './bench/support';
  import Unsupported from './bench/Unsupported.svelte';
  import { classesText, entriesText, FINGERPRINT_HELP, FROZEN, imagesText, shortFingerprint, sourceKind, sourcePath } from './bench/words';

  type Tab = 'leaderboard' | 'conditions' | 'loops';
  const id = $derived(router.route.params.id ?? '');
  const tab = $derived.by<Tab>(() => {
    const t = router.route.query.get('tab');
    return t === 'conditions' || t === 'loops' ? t : 'leaderboard';
  });
  // `?tab=loops&loop=…` (the loops' first links) opens the loop's own page.
  $effect(() => {
    const l = router.route.query.get('loop');
    if (l && id) router.go(`/benchmarks/${encodeURIComponent(id)}/loops/${encodeURIComponent(l)}`, { replace: true });
  });

  const system = new Resource((s) => hub.api.system(s));
  const catalog = new Resource((s) => hub.api.benchCatalog(s));
  const agents = new Resource((s) => hub.api.labelAgents(s));
  const models = new Resource((s) => hub.api.models(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  const jobs = new Resource((s) => hub.api.labelJobs(s));
  const live = new LiveBench(() => id || null);
  const choice = new AgentChoice(() => agents.data);
  const bench = $derived(live.bench);
  const unsupported = $derived(cannotBench(system.data, live.error));

  // The header's Add entry is this page's create action: the top bar's New run steps aside.
  ownCreateAction();
  $effect(() => {
    setCrumbs({ label: 'Benchmarks', path: '/benchmarks' }, { label: bench?.name ?? id });
  });

  /** Replaces query params (null removes one) without adding history entries. */
  function setQuery(patch: Record<string, string | null>) {
    const q = new URLSearchParams(router.route.query);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null) q.delete(k);
      else q.set(k, v);
    }
    const s = q.toString();
    router.go(`/benchmarks/${encodeURIComponent(id)}${s ? `?${s}` : ''}`, { replace: true });
  }

  // ---------- the plan: edited here while the benchmark has no entries, saved with Save plan ----------
  let draft = $state.raw<BenchPlan>({ conditions: [] });
  // Fixed once an entry finished or is under way (failed and stopped ones graded nothing to keep comparable).
  const editable = $derived(!!bench && planEditable(live.entries));
  const dirty = $derived(!!bench && editable && !samePlan(draft, bench.plan));
  // A plan from the server (the first load, a save, another tab's change) replaces the editor's unless it has edits.
  const planKey = $derived(bench ? `${bench.id}:${JSON.stringify(bench.plan)}` : '');
  let shownKey = '';
  $effect(() => {
    const key = planKey;
    if (!key || key === shownKey) return;
    untrack(() => {
      if (!shownKey.startsWith(`${bench!.id}:`) || !dirty) draft = clonePlan(bench!.plan);
      shownKey = key;
    });
  });
  let saving = $state(false);
  async function savePlan(): Promise<boolean> {
    if (!bench) return false;
    saving = true;
    const benchId = bench.id;
    const plan = clonePlan(draft);
    try {
      await live.act(() => hub.api.updateBench(benchId, { plan }));
      draft = plan;
      // page-level: the plan's own Save, outside any dialog.
      toast.success('Plan saved', `${plan.conditions.length} conditions. It is fixed once the first entry runs.`);
      return true;
    } catch (e) {
      toast.error(e, 'Could not save the plan');
      return false;
    } finally {
      saving = false;
    }
  }
  const discard = () => bench && (draft = clonePlan(bench.plan));

  // Leaving with unsaved edits asks first, whichever way (a link, Back, the palette); a change of tab does not.
  let leaving = $state<{ proceed: () => void } | null>(null);
  $effect(() =>
    router.guard((to, proceed) => {
      if (!dirty || to.path === router.route.path) return true;
      leaving = { proceed };
      return false;
    }),
  );
  async function answer(keep: 'save' | 'discard') {
    const ask = leaving;
    if (keep === 'save' && !(await savePlan())) {
      leaving = null;
      return;
    }
    if (keep === 'discard') discard();
    leaving = null;
    // The editor is clean now, so the guard lets it through.
    queueMicrotask(() => ask?.proceed());
  }
  function onbeforeunload(e: BeforeUnloadEvent) {
    if (dirty) e.preventDefault();
  }

  // ---------- entries ----------
  let selected = $state<string[]>([]);
  $effect(() => {
    const kept = keepSelection(selected, live.entries);
    if (kept.length !== selected.length) untrack(() => (selected = kept));
  });
  const comparing = $derived(selected.length >= 2);
  /** The entries as the leaderboard shows them (one graded on an earlier version after the current ones): Compare's order. */
  const board = $derived(bench ? boardOrder(live.entries, bench.fingerprint) : live.entries);
  let addOpen = $state(router.route.query.get('add') !== null);
  let addKind = $state<EntryKind>(((k) => (k === 'agent' || k === 'external' ? k : 'model'))(router.route.query.get('add')));
  function started(e: BenchEntry) {
    live.putEntry(e, live.mark());
    if (tab !== 'leaderboard') setQuery({ tab: null });
    // page-level: the dialog has closed (AddEntryDialog closes before it calls this).
    toast.success(e.status === 'done' ? 'Entry scored' : 'Entry added', e.status === 'done' && e.score ? `${e.name} · score ${e.score.total.toFixed(1)}` : `${e.name} · ${e.status === 'queued' ? 'waiting for the GPU' : 'running'}`);
  }

  let stopping = $state<BenchEntry | null>(null);
  let stopBusy = $state(false);
  const stopStatus = new DialogStatus();
  async function stopEntry() {
    if (!bench || !stopping) return;
    const e = stopping;
    stopBusy = true;
    const mark = live.mark();
    try {
      live.putEntry(await hub.api.stopBenchEntry(bench.id, e.id), mark);
      stopping = null;
    } catch (err) {
      stopStatus.error(err, 'Could not stop the entry');
    } finally {
      stopBusy = false;
    }
  }
  let removing = $state<BenchEntry | null>(null);
  let removeBusy = $state(false);
  const removeStatus = new DialogStatus();
  async function removeEntry() {
    if (!bench || !removing) return;
    const e = removing;
    removeBusy = true;
    try {
      await hub.api.deleteBenchEntry(bench.id, e.id);
      live.dropEntry(e.id);
      removing = null;
      // page-level: its confirm dialog has closed on the line above.
      toast.success(`Deleted ${e.name}`, 'Its result is gone from the leaderboard. The model and its files were not touched.');
    } catch (err) {
      removeStatus.error(err, 'Could not delete the entry');
    } finally {
      removeBusy = false;
    }
  }

  // ---------- the benchmark ----------
  let editOpen = $state(false);
  let designOpen = $state(false);
  let deleteOpen = $state(false);
  let deleteBusy = $state(false);
  const deleteStatus = new DialogStatus();
  // The server's own word (`Bench.active`, exactly when DELETE answers 409 bench_active), and what the page has heard since.
  const busyWhy = $derived(bench?.active || live.entries.some(isEntryActive) || live.loops.some(isLoopActive) ? 'Stop its running entries and loops first' : null);

  // ---------- loops ----------
  let loopOpen = $state(false);
  function loopStarted(l: BenchLoop) {
    live.putLoop(l, live.mark());
    // page-level: the dialog has closed.
    toast.success(`Loop ${l.name} started`, 'Its first generation is under way.', { label: 'Open loop', run: () => router.go(`/benchmarks/${encodeURIComponent(id)}/loops/${encodeURIComponent(l.id)}`) });
  }
  async function remove() {
    if (!bench) return;
    deleteBusy = true;
    try {
      await hub.api.deleteBench(bench.id);
      deleteOpen = false;
      toast.success(`Deleted ${bench.name}`, 'Its images, entries and loops are gone. Models, runs and datasets were not touched.');
      router.go('/benchmarks');
    } catch (e) {
      deleteStatus.error(e, 'Could not delete the benchmark');
    } finally {
      deleteBusy = false;
    }
  }
  async function copyFingerprint() {
    if (!bench) return;
    // page-level: the header's ⋯ and the fingerprint's own button, outside any dialog.
    try {
      await navigator.clipboard.writeText(bench.fingerprint);
      toast.success('Fingerprint copied', bench.fingerprint);
    } catch {
      toast.error(new Error('The browser blocked clipboard access.'), 'Could not copy the fingerprint');
    }
  }

  // ---------- the header ----------
  const phone = new MediaQuery(media.phone, false);
  const now = new Now(() => live.lost !== null);
  const offline = $derived(isOffline(live.error));
  const addWhy = $derived(offline ? OFFLINE_REASON : dirty ? 'Save or discard the plan’s changes first' : null);
  const designWhy = $derived(!bench ? null : !editable ? FROZEN : null);
  /** The source, linked while the server still has it; a benchmark keeps its own copy either way. */
  const sourceKnown = $derived.by(() => {
    if (!bench) return null;
    const s = bench.source;
    if (s.kind === 'dataset') return datasets.data ? datasets.data.some((d) => d.id === s.dataset_id) : null;
    return jobs.data ? jobs.data.some((j) => j.id === s.job_id) : jobs.error ? false : null;
  });
  const menu = $derived<MenuItem[]>([
    { label: 'Edit name and description', icon: Pencil, onclick: () => (editOpen = true) },
    { label: 'Copy fingerprint', icon: Copy, onclick: () => void copyFingerprint() },
    { label: 'Delete benchmark', icon: Trash, danger: true, onclick: () => (deleteOpen = true) },
  ]);

  pageCommands((): PageCommand[] => {
    const b = bench;
    if (!b) return [];
    const out: PageCommand[] = [];
    if (!addWhy) out.push({ id: 'bench-add', label: 'Add entry', match: ['New entry', 'Benchmark a model'], icon: Plus, run: () => ((addKind = 'model'), (addOpen = true)) });
    if (comparing) out.push({ id: 'bench-compare', label: 'Compare selected entries', icon: GitCompare, run: () => router.go(comparePath(b.id, selected, board)) });
    if (editable) out.push({ id: 'bench-design', label: 'Design the plan with an agent', icon: Sparkles, run: () => (setQuery({ tab: 'conditions' }), (designOpen = true)) });
    for (const e of live.entries.filter((x) => x.status === 'running' || x.status === 'queued'))
      out.push({ id: `bench-stop-${e.id}`, label: `Stop entry ${e.name}`, icon: Square, run: () => (stopping = e) });
    if (!offline) out.push({ id: 'bench-loop-new', label: 'New self-improvement loop', match: ['New loop'], icon: Repeat, run: () => (setQuery({ tab: 'loops' }), (loopOpen = true)) });
    out.push({ id: 'bench-fingerprint', label: 'Copy fingerprint', icon: Copy, run: () => void copyFingerprint() });
    return out;
  });
</script>

<svelte:window {onbeforeunload} />

{#snippet fpWord()}Fingerprint{/snippet}

{#if unsupported}
  <Page title="Benchmarks">
    <Unsupported />
  </Page>
{:else if live.error && !bench}
  <Page title="Benchmark">
    {#snippet actions()}<Button icon={ArrowLeft} href={href('/benchmarks')}>All benchmarks</Button>{/snippet}
    {#if live.error instanceof ApiError && live.error.status === 404}
      <ErrorState error={live.error} title="Benchmark not found" />
    {:else}
      <ErrorState error={live.error} title="Could not load this benchmark" retry={() => live.retry()} />
    {/if}
  </Page>
{:else if !bench}
  <div class="sk-page" aria-busy="true" aria-label="Loading benchmark">
    <div class="sk-head"><Skeleton w="min(240px, 60%)" h="24px" /><Skeleton w="min(420px, 90%)" h="14px" /></div>
    <div class="sk-head"><Skeleton w="min(500px, 90%)" h="12px" /><Skeleton w="min(380px, 70%)" h="12px" /></div>
    <div class="sk-tabs"><Skeleton w="96px" h="14px" /><Skeleton w="88px" h="14px" /><Skeleton w="56px" h="14px" /></div>
    <!-- The leaderboard's frame: its heading, then rows of a name over its facts, a score and the axes. -->
    <div class="sk-board">
      <div class="sk-title"><Skeleton w="96px" h="14px" /></div>
      {#each [0, 1, 2, 3, 4] as i (i)}
        <div class="sk-row">
          <span class="sk-name"><Skeleton w="140px" h="12px" /><Skeleton w="100px" h="10px" /></span>
          <Skeleton w="40px" h="16px" />
          <Skeleton w="66px" h="20px" />
          <span class="sk-plot"><Skeleton h="6px" /></span>
        </div>
      {/each}
    </div>
  </div>
{:else}
  <Page title={bench.name} inlineActions>
    {#snippet lead()}
      {#if live.connection === 'reconnecting'}<Badge tone="warning"><WifiOff size={12} /> {live.lost !== null ? lastKnown('Reconnecting', live.lost, now.current) : 'Reconnecting'}</Badge>{/if}
    {/snippet}
    {#snippet sub()}
      <Meta>
        <span>
          {#if sourceKnown}<a class="link" href={href(sourcePath(bench.source))}>{bench.source_name}</a>{:else}<span title={sourceKnown === false ? 'No longer on this server; the benchmark keeps its own copy of the images.' : undefined}>{bench.source_name}</span>{#if sourceKnown === false}<Note text="deleted since" />{/if}{/if}
        </span>
        <span>{sourceKind(bench.source)}</span>
        <span class="num">{imagesText(bench.images)}</span>
        <span class="num">{num(bench.boxes)} boxes</span>
        <span class="num">{classesText(bench.classes.length)}</span>
        <span class="faint" title={date(bench.created)}>made {ago(bench.created)}</span>
      </Meta>
    {/snippet}
    {#snippet subPhone()}<span class="num">{imagesText(bench.images)} · {entriesText(bench.entries)}</span>{/snippet}
    {#snippet actions()}
      {#if phone.current}
        <Gate reason={addWhy}>
          {#snippet children(why)}<Button variant={comparing ? 'secondary' : 'primary'} icon={Plus} disabled={!!why} aria-describedby={why} onclick={() => ((addKind = 'model'), (addOpen = true))}>Add entry</Button>{/snippet}
        </Gate>
        <Menu label="More benchmark actions" items={menu} />
      {:else}
        {#if live.entries.length >= 2}
          {#if tab === 'leaderboard'}<span class="cmp-hint" role="status" data-gate-why>{compareHint(selected.length)}</span>{/if}
          <Gate reason={compareReason(selected.length)}>
            {#snippet children(why)}
              <Button variant={comparing ? 'primary' : 'secondary'} icon={GitCompare} disabled={!!why} aria-describedby={why} href={comparing ? href(comparePath(bench.id, selected, board)) : undefined}>
                Compare{#if selected.length}<span class="num count">{selected.length}</span>{/if}
              </Button>
            {/snippet}
          </Gate>
        {/if}
        {#if addWhy && addWhy !== OFFLINE_REASON}<span class="cmp-hint" data-gate-why>{addWhy}</span>{/if}
        <Gate reason={addWhy}>
          {#snippet children(why)}<Button variant={comparing ? 'secondary' : 'primary'} icon={Plus} disabled={!!why} aria-describedby={why} onclick={() => ((addKind = 'model'), (addOpen = true))}>Add entry</Button>{/snippet}
        </Gate>
        <Menu label="More benchmark actions" items={menu} />
      {/if}
    {/snippet}

    <div class="about">
      {#if bench.description.trim()}
        <p class="desc">{bench.description}</p>
      {:else}
        <p class="desc none">No description yet. Say where the camera is and what makes the job hard: an agent designing the plan reads it. <button type="button" class="inline" onclick={() => (editOpen = true)}>Add a description</button></p>
      {/if}
      <div class="fp">
        <InfoButton label="What the fingerprint is for" trigger={fpWord} text={FINGERPRINT_HELP} />
        <span title={bench.fingerprint}><Ident value={shortFingerprint(bench.fingerprint)} /></span>
        <Button variant="ghost" size="sm" icon={Copy} aria-label="Copy the fingerprint" onclick={copyFingerprint} />
      </div>
    </div>

    <div class="tabs">
      <Tabs
        label="Benchmark views"
        idPrefix="bn-tab"
        tabs={[
          { value: 'leaderboard', label: 'Leaderboard', meta: String(live.entries.length) },
          { value: 'conditions', label: 'Conditions', meta: String(bench.plan.conditions.length) },
          { value: 'loops', label: 'Loops', meta: String(live.loops.length) },
        ]}
        value={tab}
        onchange={(t) => setQuery({ tab: t === 'leaderboard' ? null : t })}
      />
    </div>

    <div id="bn-tab-panel" role="tabpanel" aria-labelledby="bn-tab-{tab}" class="panel">
      {#if tab === 'conditions'}
        {#if catalog.data}
          <PlanTab
            {bench}
            {editable}
            catalog={catalog.data}
            bind:draft
            {dirty}
            {saving}
            onsave={() => void savePlan()}
            ondiscard={discard}
            ondesign={() => (designOpen = true)}
            {designWhy}
          />
        {:else if catalog.error}
          <ErrorState error={catalog.error} title="Could not load the conditions" retry={() => catalog.refresh()} />
        {:else}
          <Skeleton h="480px" r="var(--radius-lg)" />
        {/if}
      {:else if tab === 'loops'}
        <LoopsTab {bench} loops={live.loops} entries={live.entries} datasets={datasets.data} jobs={jobs.data} onnew={() => (loopOpen = true)} {offline} />
      {:else}
        <Leaderboard
          {bench}
          entries={live.entries}
          catalog={catalog.data}
          agents={agents.data}
          bind:selected
          lost={live.lost !== null}
          onadd={() => ((addKind = 'model'), (addOpen = true))}
          onstop={(e) => (stopping = e)}
          ondelete={(e) => (removing = e)}
          oncompare={() => router.go(comparePath(bench.id, selected, board))}
        />
      {/if}
    </div>
  </Page>

  <AddEntryDialog
    bind:open={addOpen}
    bind:kind={addKind}
    {bench}
    catalog={catalog.data}
    entries={live.entries}
    models={models.data}
    modelsError={models.error}
    agents={agents.data}
    agentsError={agents.error}
    retryAgents={() => agents.refresh()}
    {choice}
    onstarted={started}
  />
  {#if catalog.data}
    <DesignDialog
      bind:open={designOpen}
      {bench}
      catalog={catalog.data}
      current={draft}
      agents={agents.data}
      agentsError={agents.error}
      retryAgents={() => agents.refresh()}
      {choice}
      onuse={(p) => (draft = clonePlan(p))}
    />
  {/if}
  <NewLoopDialog bind:open={loopOpen} {bench} models={models.data} datasets={datasets.data} jobs={jobs.data} entries={live.entries} onstarted={loopStarted} />
  <EditBenchDialog bind:open={editOpen} {bench} onsave={(change) => live.act(() => hub.api.updateBench(bench.id, change))} />

  <Dialog
    open={!!stopping}
    onclose={() => (stopping = null)}
    title="Stop {stopping?.name ?? 'the entry'}?"
    description="It ends what is in flight. A stopped entry has no score: what it ran so far is not kept."
    size="sm"
    status={stopStatus}
  >
    {#snippet footer()}
      <Button onclick={() => (stopping = null)}>Keep running</Button>
      <Button variant="destructive" icon={Square} loading={stopBusy} onclick={stopEntry}>Stop entry</Button>
    {/snippet}
  </Dialog>
  <Dialog
    open={!!removing}
    onclose={() => (removing = null)}
    title="Delete {removing?.name ?? 'the entry'}?"
    description="Removes the entry and its result from the leaderboard. The model, its files and the benchmark's images are not touched."
    size="sm"
    status={removeStatus}
  >
    {#snippet footer()}
      <Button onclick={() => (removing = null)}>Cancel</Button>
      <Button variant="destructive" icon={Trash} loading={removeBusy} onclick={removeEntry}>Delete entry</Button>
    {/snippet}
  </Dialog>
  <Dialog
    bind:open={deleteOpen}
    title="Delete {bench.name}?"
    description="Removes the benchmark, its copy of the images, its entries and its loops. Models, runs and datasets (a loop's included) are not touched."
    size="sm"
    status={deleteStatus}
  >
    {#snippet footer()}
      {#if busyWhy}<p class="why" data-gate-why>{busyWhy}</p>{/if}
      <Button onclick={() => (deleteOpen = false)}>Cancel</Button>
      <Gate reason={busyWhy}>
        {#snippet children(why)}<Button variant="destructive" icon={Trash} loading={deleteBusy} disabled={!!why} aria-describedby={why} onclick={remove}>Delete benchmark</Button>{/snippet}
      </Gate>
    {/snippet}
  </Dialog>
  <Dialog open={!!leaving} title="Save the plan?" description="{bench.name}'s plan has changes you have not saved." size="sm" onclose={() => (leaving = null)}>
    {#snippet footer()}
      <Button onclick={() => answer('discard')}>Discard changes</Button>
      <Button variant="primary" loading={saving} onclick={() => answer('save')}>Save plan</Button>
    {/snippet}
  </Dialog>
{/if}

<style>
  .count {
    font-weight: var(--weight-regular);
    margin-left: var(--space-1);
  }
  .cmp-hint {
    align-self: center;
    white-space: nowrap;
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .link {
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  /* What the benchmark stands for, then its fingerprint: one band under the header. */
  .about {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-3) var(--space-8);
    margin-top: calc(-1 * var(--space-2));
  }
  .desc {
    max-inline-size: var(--measure-dialog);
    color: var(--text-secondary);
  }
  .desc.none {
    color: var(--text-tertiary);
  }
  .inline {
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .fp {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: var(--control-md);
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  .tabs {
    margin-bottom: calc(var(--space-1) * -1);
  }
  .panel {
    display: grid;
    gap: var(--space-5);
    min-width: 0;
  }
  .why {
    margin-right: auto;
    align-self: center;
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  @media (min-width: 768px) and (max-width: 1023px) {
    .cmp-hint {
      display: none;
    }
  }
  @media (max-width: 1023px) {
    .about {
      flex-direction: column;
      gap: var(--space-2);
    }
  }
  .sk-page {
    max-width: calc(var(--size-content-max) + 2 * var(--space-6));
    margin: 0 auto;
    padding: var(--space-6);
    display: grid;
    gap: var(--space-5);
  }
  .sk-head {
    display: grid;
    gap: var(--space-2);
  }
  .sk-tabs {
    display: flex;
    gap: var(--space-5);
    padding-bottom: var(--space-3);
    border-bottom: 1px solid var(--line-subtle);
  }
  .sk-board {
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    background: var(--bg-panel);
  }
  .sk-title {
    display: flex;
    align-items: center;
    height: 40px;
    padding: 0 var(--space-4);
  }
  .sk-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 40px 66px minmax(0, 200px);
    align-items: center;
    gap: var(--space-6);
    height: 48px;
    padding: 0 var(--space-4);
    border-top: 1px solid var(--line-subtle);
  }
  .sk-name {
    display: grid;
    gap: var(--space-2);
  }
  @media (max-width: 767px) {
    .sk-row {
      grid-template-columns: minmax(0, 1fr) 40px;
      height: 56px;
    }
    .sk-row > :global(:nth-child(3)),
    .sk-plot {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .sk-page {
      padding: var(--space-4);
    }
    .why {
      flex-basis: 100%;
    }
  }
</style>
