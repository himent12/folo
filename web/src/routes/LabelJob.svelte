<script lang="ts">
  import { MediaQuery } from 'svelte/reactivity';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import Database from '@lucide/svelte/icons/database';
  import ImagePlus from '@lucide/svelte/icons/image-plus';
  import Pencil from '@lucide/svelte/icons/pencil';
  import RotateCw from '@lucide/svelte/icons/rotate-cw';
  import Square from '@lucide/svelte/icons/square';
  import Trash from '@lucide/svelte/icons/trash-2';
  import WifiOff from '@lucide/svelte/icons/wifi-off';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { isActive, LiveLabelJob } from '$lib/api/liveLabel.svelte';
  import { lastKnown, Now } from '$lib/api/freshness.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Dataset } from '$lib/api/types';
  import { media } from '$lib/design/tokens';
  import { ago, num } from '$lib/format';
  import { AGENT_NAME, INSTRUCTIONS_MAX } from '$lib/label';
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
  import Menu, { type MenuItem } from '$lib/ui/Menu.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import StatusBadge from '$lib/ui/StatusBadge.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import Logs from './run/Logs.svelte';
  import AddImagesDialog from './label/AddImagesDialog.svelte';
  import AgentPanel from './label/AgentPanel.svelte';
  import { AgentChoice } from './label/agentChoice.svelte';
  import DatasetDialog from './label/DatasetDialog.svelte';
  import Editor from './label/Editor.svelte';
  import ImagesArea, { inFilter, type Show } from './label/ImagesArea.svelte';
  import JobDialog from './label/JobDialog.svelte';
  import { cannotLabel } from './label/support';
  import Unsupported from './label/Unsupported.svelte';
  import { Uploads } from './label/uploads.svelte';
  import { images as imagesText, jobStatus, toLabel } from './label/words';

  const id = $derived(router.route.params.id ?? '');
  const show = $derived.by<Show>(() => {
    const s = router.route.query.get('show');
    return s === 'pending' || s === 'labeled' || s === 'failed' ? s : 'all';
  });
  const imageParam = $derived(router.route.query.get('image'));

  const system = new Resource((s) => hub.api.system(s));
  const agents = new Resource((s) => hub.api.labelAgents(s));
  const datasets = new Resource((s) => hub.api.datasets(s));
  const live = new LiveLabelJob(() => id || null);
  const choice = new AgentChoice(() => agents.data);
  const uploads = new Uploads(() => id, live);
  const job = $derived(live.job);
  const running = $derived(isActive(job));
  const unsupported = $derived(cannotLabel(system.data, live.error));

  // The header's Create dataset is this page's create action: the top bar's New run steps aside.
  ownCreateAction();

  $effect(() => {
    setCrumbs({ label: 'Datasets', path: '/datasets' }, { label: 'Label images', path: '/datasets/label' }, { label: job?.name ?? id });
  });

  /** Replaces query params (null removes one) without adding history entries. */
  function setQuery(patch: Record<string, string | null>) {
    const q = new URLSearchParams(router.route.query);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null) q.delete(k);
      else q.set(k, v);
    }
    const s = q.toString();
    router.go(`/datasets/label/${encodeURIComponent(id)}${s ? `?${s}` : ''}`, { replace: true });
  }
  const shown = $derived(show === 'all' ? live.images : live.images.filter((i) => inFilter(i, show)));

  // ---------- the instructions: saved when the field is left, and before a run starts ----------
  let instructions = $state('');
  /** What the field last took from the job: a change on the server shows unless the field holds an edit. */
  let taken = '';
  let instructionsError = $state<string | null>(null);
  $effect(() => {
    const server = job?.instructions ?? '';
    if (instructions === taken) instructions = server;
    taken = server;
  });
  async function saveInstructions(): Promise<boolean> {
    if (!job || running || instructions === job.instructions) return true;
    if (instructions.length > INSTRUCTIONS_MAX) {
      instructionsError = `At most ${num(INSTRUCTIONS_MAX)} characters (now ${num(instructions.length)}).`;
      return false;
    }
    const jobId = job.id;
    try {
      await live.act(() => hub.api.updateLabelJob(jobId, { instructions }));
      taken = instructions;
      instructionsError = null;
      return true;
    } catch (e) {
      instructionsError = `Not saved: ${e instanceof Error ? e.message : String(e)}`;
      return false;
    }
  }
  $effect(() => {
    if (instructions.length <= INSTRUCTIONS_MAX && instructionsError?.startsWith('At most')) instructionsError = null;
  });

  // ---------- runs ----------
  let starting = $state(false);
  let startError = $state.raw<unknown>(undefined);
  async function start(scope: 'unlabeled' | 'all' = 'unlabeled') {
    const req = choice.request();
    if (!job || !req) return;
    starting = true;
    startError = undefined;
    try {
      if (!(await saveInstructions())) return;
      const jobId = job.id;
      await live.act(() => hub.api.startLabeling(jobId, { ...req, scope }));
      againOpen = false;
    } catch (e) {
      // In its dialog when asked from one, else in the panel under the button.
      if (againOpen) againStatus.error(e, 'Could not start the agent');
      else startError = e;
    } finally {
      starting = false;
    }
  }

  let stopOpen = $state(false);
  let stopBusy = $state(false);
  const stopStatus = new DialogStatus();
  async function stop() {
    if (!job) return;
    stopBusy = true;
    const jobId = job.id;
    try {
      await live.act(() => hub.api.stopLabeling(jobId));
      stopOpen = false;
    } catch (e) {
      stopStatus.error(e, 'Could not stop the agent');
    } finally {
      stopBusy = false;
    }
  }

  let againOpen = $state(false);
  const againStatus = new DialogStatus();

  // ---------- the job ----------
  let editOpen = $state(false);
  let deleteOpen = $state(false);
  let deleteBusy = $state(false);
  const deleteStatus = new DialogStatus();
  async function remove() {
    if (!job) return;
    deleteBusy = true;
    try {
      await hub.api.deleteLabelJob(job.id);
      deleteOpen = false;
      toast.success(`Deleted ${job.name}`, 'Images from folders and datasets made from it were not touched.');
      router.go('/datasets/label');
    } catch (e) {
      deleteStatus.error(e, 'Could not delete the job');
    } finally {
      deleteBusy = false;
    }
  }

  let datasetOpen = $state(false);
  function created(ds: Dataset) {
    void datasets.refresh();
    // page-level: the dialog has closed (DatasetDialog closes before it calls this); the toast offers the dataset.
    toast.success('Dataset created', `${ds.name} · ${imagesText(ds.splits.train + ds.splits.val + ds.splits.test)}`, {
      label: 'Open dataset',
      run: () => router.go(`/datasets/${ds.id}`),
    });
  }
  const made = $derived(job?.dataset_id ? datasets.data?.find((d) => d.id === job.dataset_id) : undefined);

  // ---------- adding images ----------
  let fileInput = $state<HTMLInputElement>();
  let addOpen = $state(false);
  let addTab = $state<'upload' | 'folder'>('upload');
  function onfiles(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    void uploads.files([...(input.files ?? [])]);
    input.value = '';
  }

  // ---------- header ----------
  const phone = new MediaQuery(media.phone, false);
  /** Why a dataset cannot be made now, in words; null when it can. */
  const datasetWhy = $derived(!job ? null : running ? 'Wait for the agent to finish' : job.counts.labeled === 0 ? 'Label an image first' : null);
  /** Everything is labeled and nothing runs: making the dataset is the one thing left to do. */
  const datasetNext = $derived(!!job && !datasetWhy && job.counts.images > 0 && toLabel(job) === 0);
  const editItems: MenuItem[] = [
    { label: 'Edit job', icon: Pencil, onclick: () => (editOpen = true) },
    { label: 'Delete job', icon: Trash, danger: true, onclick: () => (deleteOpen = true) },
  ];
  const phoneItems = $derived<MenuItem[]>([
    ...(!datasetNext && !datasetWhy ? [{ label: 'Create dataset', icon: Database, onclick: () => (datasetOpen = true) }] : []),
    { label: 'Add images', icon: ImagePlus, onclick: () => ((addTab = 'upload'), (addOpen = true)) },
    ...editItems,
  ]);
  const st = $derived(job ? jobStatus(job) : null);
  const now = new Now(() => live.lost !== null);

  pageCommands((): PageCommand[] => {
    const j = job;
    if (!j) return [];
    const out: PageCommand[] = [{ id: 'label-add', label: 'Add images', icon: ImagePlus, run: () => ((addTab = 'upload'), (addOpen = true)) }];
    if (j.status === 'running') out.push({ id: 'label-stop', label: `Stop labeling ${j.name}`, match: ['Stop agent'], icon: Square, run: () => (stopOpen = true) });
    if (!datasetWhy) out.push({ id: 'label-dataset', label: 'Create dataset', icon: Database, run: () => (datasetOpen = true) });
    return out;
  });
</script>

<input bind:this={fileInput} type="file" accept="image/jpeg,image/png,image/bmp,image/webp" multiple hidden onchange={onfiles} />

{#if unsupported}
  <Page title="Label images">
    <Unsupported />
  </Page>
{:else if live.error && !job}
  <Page title="Labeling job">
    {#snippet actions()}<Button icon={ArrowLeft} href={href('/datasets/label')}>All labeling jobs</Button>{/snippet}
    {#if live.error instanceof ApiError && live.error.status === 404}
      <ErrorState error={live.error} title="Labeling job not found" />
    {:else}
      <ErrorState error={live.error} title="Could not load this labeling job" retry={() => live.retry()} />
    {/if}
  </Page>
{:else if !job || !st}
  <div class="sk-page" aria-busy="true" aria-label="Loading labeling job">
    <div class="sk-head"><Skeleton w="64px" h="18px" /><Skeleton w="220px" h="24px" /><Skeleton w="300px" h="14px" /></div>
    <div class="work">
      <div class="sk-main">
        <Skeleton h="28px" r="var(--radius-md)" />
        <div class="sk-tiles">{#each { length: 12 }, i (i)}<Skeleton h="auto" r="var(--radius-md)" class="sk-tile" image />{/each}</div>
      </div>
      <div class="sk-rail"><Skeleton h="520px" r="var(--radius-lg)" /></div>
    </div>
  </div>
{:else}
  <Page title={job.name} inlineActions>
    {#snippet lead()}
      <StatusBadge plain status={st.status} label={st.label} lastKnown={live.lost !== null ? lastKnown(st.label, live.lost, now.current) : undefined} />
      {#if live.connection === 'reconnecting'}<Badge tone="warning"><WifiOff size={12} /> Reconnecting</Badge>{/if}
    {/snippet}
    {#snippet sub()}
      <Meta>
        <span class="num">{imagesText(job.counts.images)}</span>
        <span class="num">{num(job.counts.labeled)} labeled</span>
        <span class="num">{job.classes.length} {job.classes.length === 1 ? 'class' : 'classes'}</span>
        {#if job.dataset_id}<span>made <a class="link" href={href(`/datasets/${job.dataset_id}`)}>{made?.name ?? 'a dataset'}</a></span>{/if}
        <span class="faint">created {ago(job.created)}</span>
        {#if datasetWhy && !running}<span class="faint" data-gate-why>Create dataset: {datasetWhy.toLowerCase()}</span>{/if}
      </Meta>
    {/snippet}
    {#snippet subPhone()}<span class="num">{job.counts.images ? `${num(job.counts.labeled)} of ${imagesText(job.counts.images)} labeled` : 'No images yet'}</span>{/snippet}
    {#snippet actions()}
      {#if phone.current}
        {#if datasetNext}<Button variant="primary" icon={Database} onclick={() => (datasetOpen = true)}>Create dataset</Button>{/if}
        <Menu label="More job actions" items={phoneItems} />
      {:else}
        <Gate reason={datasetWhy}>
          {#snippet children(why)}
            <Button variant={datasetNext ? 'primary' : 'secondary'} icon={Database} disabled={!!why} aria-describedby={why} onclick={() => (datasetOpen = true)}>Create dataset</Button>
          {/snippet}
        </Gate>
        <Menu label="More job actions" items={editItems} />
      {/if}
    {/snippet}

    <div class="work">
      <div class="main">
        <ImagesArea
          jobId={job.id}
          classes={job.classes}
          images={live.images}
          {show}
          {uploads}
          onshow={(s) => setQuery({ show: s === 'all' ? null : s })}
          onopen={(imageId) => setQuery({ image: imageId })}
          onbrowse={() => fileInput?.click()}
          onadd={(tab) => ((addTab = tab), (addOpen = true))}
        />
      </div>
      <aside class="rail" aria-label="Agent and log">
        <div class="agent">
          <AgentPanel
            {job}
            agents={agents.data}
            agentsError={agents.error}
            retryAgents={() => agents.refresh()}
            {choice}
            bind:instructions
            {instructionsError}
            {starting}
            {startError}
            lost={live.lost !== null}
            onsaveinstructions={() => void saveInstructions()}
            onstart={() => void start()}
            onagain={() => (againOpen = true)}
            onstop={() => (stopOpen = true)}
            onedit={() => (editOpen = true)}
          />
        </div>
        <div class="logs"><Logs logs={live.logs} version={live.version} live={running} /></div>
      </aside>
    </div>
  </Page>

  <Editor {live} {job} images={shown} imageId={imageParam} {choice} onnavigate={(to) => setQuery({ image: to })} />
  <AddImagesDialog bind:open={addOpen} bind:tab={addTab} {uploads} onbrowse={() => fileInput?.click()} />
  <DatasetDialog bind:open={datasetOpen} {job} oncreated={created} />
  <JobDialog bind:open={editOpen} {job} onsave={(change) => live.act(() => hub.api.updateLabelJob(job.id, change))} />

  <Dialog
    bind:open={stopOpen}
    title="Stop labeling {job.name}?"
    description="The agent's calls in flight end, and the images they held keep the labels they had. What it has labeled so far stays."
    size="sm"
    status={stopStatus}
  >
    {#snippet footer()}
      <Button onclick={() => (stopOpen = false)}>Keep labeling</Button>
      <Button variant="destructive" icon={Square} loading={stopBusy} onclick={stop}>Stop</Button>
    {/snippet}
  </Dialog>
  <Dialog
    bind:open={againOpen}
    title="Label all {imagesText(job.counts.images)} again?"
    description="{choice.agent ? AGENT_NAME[choice.agent.id] : 'The agent'} labels every image again. The labels they have now, hand-drawn ones too, are replaced."
    size="sm"
    status={againStatus}
  >
    {#snippet footer()}
      <Button onclick={() => (againOpen = false)}>Cancel</Button>
      <Button variant="destructive" icon={RotateCw} loading={starting} onclick={() => start('all')}>Label all again</Button>
    {/snippet}
  </Dialog>
  <Dialog
    bind:open={deleteOpen}
    title="Delete {job.name}?"
    description="Removes the job, its uploaded images and its labels. Images added from a folder stay on disk, and datasets made from this job are not touched."
    size="sm"
    status={deleteStatus}
  >
    {#snippet footer()}
      <Button onclick={() => (deleteOpen = false)}>Cancel</Button>
      <Button variant="destructive" icon={Trash} loading={deleteBusy} onclick={remove}>Delete job</Button>
    {/snippet}
  </Dialog>
{/if}

<style>
  /* Images on 8 columns, the agent and its log a sticky rail on 4 (DESIGN_DIRECTION §2.2). */
  .work {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
    gap: var(--space-4);
    align-items: start;
  }
  .main {
    min-width: 0;
  }
  .rail {
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    max-height: calc(100dvh - var(--size-topbar) - var(--space-8));
    min-width: 0;
    overflow-y: auto;
  }
  .agent {
    flex: none;
  }
  .logs {
    flex: 1 0 auto;
    display: grid;
    --log-h: 200px;
  }
  .link {
    color: var(--text-primary);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  /* Below 1024 px one column: the agent first (the work), then the images, then the log. */
  @media (max-width: 1023px) {
    .work {
      grid-template-columns: minmax(0, 1fr);
    }
    .rail {
      display: contents;
    }
    .agent {
      order: -1;
    }
    .logs {
      order: 1;
      --log-h: 300px;
    }
  }
  .sk-page {
    max-width: calc(var(--size-content-max) + 2 * var(--space-6));
    margin: 0 auto;
    padding: var(--space-6);
    display: grid;
    gap: var(--space-5);
  }
  .sk-head,
  .sk-main {
    display: grid;
    gap: var(--space-2);
  }
  .sk-main {
    gap: var(--space-3);
  }
  .sk-tiles {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(172px, 1fr));
    gap: var(--space-2);
  }
  .sk-tiles :global(.sk-tile) {
    aspect-ratio: 4 / 3;
  }
  @media (max-width: 1023px) {
    .sk-rail {
      order: -1;
    }
    .sk-rail :global(.sk) {
      height: 280px;
    }
  }
  @media (max-width: 767px) {
    .sk-page {
      padding: var(--space-4);
    }
    .sk-tiles {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
