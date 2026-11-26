// One labeling job (`#/datasets/label/<id>`): its images, the instructions and the agent runs that label them, the box
// editor, and Create dataset. The parts are in ./label/.
import ArrowLeft from '$icons/arrow-left';
import Database from '$icons/database';
import ImagePlus from '$icons/image-plus';
import Pencil from '$icons/pencil';
import RotateCw from '$icons/rotate-cw';
import Square from '$icons/square';
import Trash from '$icons/trash-2';
import WifiOff from '$icons/wifi-off';
import { computed, css, effect, field, For, MediaQuery, onCleanup, onWindow, signal, untrack } from 'fw';
import { ApiError, hub } from '$lib/api/index';
import { isActive, LiveLabelJob } from '$lib/api/liveLabel';
import { lastKnown, Now } from '$lib/api/freshness';
import { Resource } from '$lib/api/resource';
import type { Dataset } from '$lib/api/types';
import { media } from '$lib/design/tokens';
import { ago, num } from '$lib/format';
import { AGENT_NAME, INSTRUCTIONS_MAX } from '$lib/label';
import { href, router } from '$lib/router';
import Page from '$lib/shell/Page';
import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs';
import { entryState } from '$lib/shell/entryMemory';
import { pageCommands, type PageCommand } from '$lib/shell/pageCommands';
import Badge from '$lib/ui/Badge';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import ErrorState from '$lib/ui/ErrorState';
import { DialogStatus } from '$lib/ui/feedback';
import Gate from '$lib/ui/Gate';
import Menu, { type MenuItem } from '$lib/ui/Menu';
import Meta from '$lib/ui/Meta';
import Skeleton from '$lib/ui/Skeleton';
import StatusBadge from '$lib/ui/StatusBadge';
import { toast } from '$lib/ui/toast';
import Logs from './run/Logs';
import { datasetPresence, datasetsFor, type DatasetPresence } from './overview/newDatasets';
import AddImagesDialog from './label/AddImagesDialog';
import AgentPanel from './label/AgentPanel';
import { AgentChoice } from './label/agentChoice';
import DatasetDialog from './label/DatasetDialog';
import Editor from './label/Editor';
import ImagesArea, { inFilter, type Show } from './label/ImagesArea';
import JobDialog from './label/JobDialog';
import { cannotLabel } from './label/support';
import Unsupported from './label/Unsupported';
import { Uploads } from './label/uploads';
import { Instructions } from './label/instructions';
import { images as imagesText, jobStatus, toLabel } from './label/words';

/** The loading page's tiles. */
const TILES = Array.from({ length: 12 }, (_, i) => i);

export default function LabelJob() {
  const id = computed(() => router.route.params.id ?? '');
  const show = computed<Show>(() => {
    const s = router.route.query.get('show');
    return s === 'pending' || s === 'review' || s === 'reviewed' || s === 'failed' ? s : 'all';
  });
  const imageParam = computed(() => router.route.query.get('image'));

  const system = new Resource((s) => hub.api.system(s));
  const agents = new Resource((s) => hub.api.labelAgents(s));
  const live = new LiveLabelJob(() => id.get() || null);
  const choice = new AgentChoice(() => agents.data);
  const uploads = new Uploads(() => id.get(), live);
  const job = computed(() => live.job);
  /** The job, where it is known to be there (the page's own branch). */
  const j = () => job.get()!;
  const running = computed(() => isActive(job.get()));
  const unsupported = computed(() => cannotLabel(system.data, live.error));

  // The header's Create dataset is this page's create action: the top bar's New run steps aside.
  ownCreateAction();

  effect(() => {
    setCrumbs({ label: 'Datasets', path: '/datasets' }, { label: 'Label images', path: '/datasets/label' }, { label: job.get()?.name ?? id.get() });
  });

  // History: opening an image from the grid pushes an entry, so Back closes the editor instead of leaving the job
  // (matching Dataset.tsx). Stepping between images replaces the entry; closing back to the originating view goes Back
  // so Close and Back agree and leave no spare entry. An entry reached by a direct link replaces in place on close.
  const from = entryState<string>('labeljob.from');
  const view = (path: string, q: URLSearchParams | string) => `${path}?${q}`;
  /** Entry currently unwinding via async `history.back()`, so concurrent effects/clicks never double-back. */
  let backingEntry: number | null = null;
  let backTimer: ReturnType<typeof setTimeout> | undefined;
  const clearBacking = () => {
    backingEntry = null;
    clearTimeout(backTimer);
    backTimer = undefined;
  };
  onCleanup(router.onEntry(clearBacking));
  onCleanup(clearBacking);
  // If a navigation guard vetoes `history.back()`, the router bounces back to `backingEntry` without firing `onEntry`:
  // clearing on the return `popstate` (plus a bounded fallback) ensures `backingEntry` can never latch permanently.
  onWindow('popstate', () => {
    if (backingEntry !== null && router.entry === backingEntry) clearBacking();
  });

  /** Changes query params (null removes one): `push` adds a history entry; `replaceOnly` forces in-place replace. */
  function setQuery(patch: Record<string, string | null>, push = false, replaceOnly = false) {
    if (router.location.name !== 'label-job' || router.location.params.id !== id.peek()) return;
    const before = router.route.query.toString();
    const q = new URLSearchParams(router.route.query);
    for (const [k, v] of Object.entries(patch)) {
      if (v === null) q.delete(k);
      else q.set(k, v);
    }
    const s = q.toString();
    if (s === before) return;
    const at = router.route.path;
    const path = `/datasets/label/${encodeURIComponent(id.peek())}${s ? `?${s}` : ''}`;
    if (push) {
      clearBacking();
      const was = router.entry;
      router.go(path);
      if (router.entry !== was) from.set(view(at, before));
    } else if (!replaceOnly && from.get() === view(at, s)) {
      if (backingEntry === router.entry) return;
      backingEntry = router.entry;
      clearTimeout(backTimer);
      backTimer = setTimeout(clearBacking, 200);
      history.back();
    } else {
      clearBacking();
      if (replaceOnly && patch.image === null) from.set(undefined);
      router.go(path, { replace: true });
    }
  }
  const shown = computed(() => (show.get() === 'all' ? live.images : live.images.filter((i) => inFilter(i, show.get()))));
  /** Labeled images a person has checked (the job's counts do not have them: the images do). */
  const reviewed = computed(() => live.images.reduce((n, i) => n + (i.status === 'labeled' && i.reviewed ? 1 : 0), 0));
  // A linked image the job does not have (removed since, a stale link) would open the editor empty: the job holds all
  // its images once loaded, so the link is dropped in place (scrubbing any stale Forward entry) and the page says why.
  effect(() => {
    const want = imageParam.get();
    if (!want || backingEntry === router.entry || live.loading || live.error || live.images.some((i) => i.id === want)) return;
    if (router.location.name !== 'label-job' || router.location.params.id !== id.peek()) return;
    untrack(() => {
      // page-level: the editor never opened for it.
      toast.info('Image not in this job', `${want} has been removed from it.`);
      setQuery({ image: null }, false, true);
    });
  });

  // ---------- the instructions: saved when the field is left, and before a run starts (label/instructions) ----------
  const instructions = new Instructions(
    () => live.job,
    (jobId, text) => live.act(() => hub.api.updateLabelJob(jobId, { instructions: text })),
  );
  effect(() => {
    const server = job.get()?.instructions ?? '';
    untrack(() => instructions.sync(server));
  });
  effect(() => {
    if (instructions.text.length <= INSTRUCTIONS_MAX && instructions.error?.startsWith('At most')) instructions.error = null;
  });

  // ---------- runs ----------
  let mounted = true;
  onCleanup(() => (mounted = false));
  const starting = signal(false);
  const startError = signal<unknown>(undefined);
  async function start(scope: 'unlabeled' | 'all' = 'unlabeled') {
    const req = choice.request();
    const cur = job.peek();
    if (!cur || !req || starting.peek()) return;
    const jobId = cur.id;
    const epoch = hub.epoch;
    const fromAgain = againOpen.peek();
    starting.set(true);
    startError.set(undefined);
    try {
      // After the saves asked before it, with the field's latest text: the run uses what the field says.
      if (!(await instructions.save())) return;
      if (
        !mounted ||
        job.peek()?.id !== jobId ||
        hub.epoch !== epoch ||
        hub.mode === 'connecting' ||
        router.location.name !== 'label-job' ||
        router.location.params.id !== jobId
      )
        return;
      await live.act(() => hub.api.startLabeling(jobId, { ...req, scope }));
      if (mounted && job.peek()?.id === jobId && hub.epoch === epoch) againOpen.set(false);
    } catch (e) {
      const sameJob = mounted && job.peek()?.id === jobId && hub.epoch === epoch;
      // In its dialog when asked from one, else in the panel under the button.
      if (sameJob && fromAgain && againOpen.peek()) againStatus.error(e, 'Could not start the agent');
      else if (sameJob) startError.set(e);
      else if (hub.epoch === epoch) toast.error(e, 'Could not start the agent');
    } finally {
      starting.set(false);
    }
  }

  const stopOpen = signal(false);
  const stopBusy = signal(false);
  const stopStatus = new DialogStatus();
  async function stop() {
    const cur = job.peek();
    if (!cur || stopBusy.peek()) return;
    const jobId = cur.id;
    const epoch = hub.epoch;
    stopBusy.set(true);
    try {
      await live.act(() => hub.api.stopLabeling(jobId));
      if (mounted && job.peek()?.id === jobId && hub.epoch === epoch) stopOpen.set(false);
    } catch (e) {
      if (mounted && job.peek()?.id === jobId && hub.epoch === epoch && stopOpen.peek()) stopStatus.error(e, 'Could not stop the agent');
      else if (hub.epoch === epoch) toast.error(e, 'Could not stop the agent');
    } finally {
      stopBusy.set(false);
    }
  }

  const againOpen = signal(false);
  const againStatus = new DialogStatus();

  // ---------- the job ----------
  const editOpen = signal(false);
  const deleteOpen = signal(false);
  const deleteBusy = signal(false);
  const deleteStatus = new DialogStatus();
  async function remove() {
    const cur = job.peek();
    if (!cur || deleteBusy.peek()) return;
    const jobId = cur.id;
    const epoch = hub.epoch;
    const startPath = router.location.path;
    deleteBusy.set(true);
    try {
      await hub.api.deleteLabelJob(jobId);
      if (mounted && job.peek()?.id === jobId && hub.epoch === epoch) deleteOpen.set(false);
      if (hub.epoch === epoch) toast.success(`Deleted ${cur.name}`, 'Images from folders and datasets made from it were not touched.');
      if (mounted && router.location.path === startPath && hub.epoch === epoch) router.go('/datasets/label', { replace: true });
    } catch (e) {
      if (mounted && job.peek()?.id === jobId && hub.epoch === epoch && deleteOpen.peek()) deleteStatus.error(e, 'Could not delete the job');
      else if (hub.epoch === epoch) toast.error(e, 'Could not delete the job');
    } finally {
      deleteBusy.set(false);
    }
  }

  const datasetOpen = signal(false);
  function created(ds: Dataset) {
    void datasets.refresh();
    // page-level: the dialog has closed (DatasetDialog closes before it calls this); the toast offers the dataset.
    toast.success('Dataset created', `${ds.name} · ${imagesText(ds.splits.train + ds.splits.val + ds.splits.test)}`, {
      label: 'Open dataset',
      run: () => router.go(`/datasets/${ds.id}`),
    });
  }
  // The dataset made from the job, by the id the job keeps (also once it was removed from FOLO): linked only while it is
  // registered, "since removed" once it is known removed, and neither while that is not known (./overview/newDatasets).
  // The list is read (every 5 s: one removed while the page is open stops linking) while the job loads and once it made a
  // dataset, never for a job that made none nor while the page shows an error (a job not found names none).
  const madeId = computed(() => job.get()?.dataset_id ?? null);
  const datasets = datasetsFor(() => (live.error ? [] : job.get() ? [madeId.get()] : undefined));
  const presence = datasetPresence(datasets, () => [madeId.get()]);
  const madeDataset = computed(() => {
    const dsId = madeId.get();
    return dsId ? datasets.data?.find((d) => d.id === dsId) : undefined;
  });
  const made = computed<'none' | DatasetPresence>(() => (madeId.get() ? presence(madeId.get()) : 'none'));
  /** Where a removed dataset's files are, unless they were deleted with it (the server writes them under its data dir). */
  const removedWhere = computed(() =>
    system.data?.data_dir
      ? `Its files stay in ${system.data.data_dir}/datasets unless they were deleted with it.`
      : 'Its files stay on disk unless they were deleted with it.',
  );

  // ---------- adding images ----------
  let fileInput: HTMLInputElement | undefined;
  const addOpen = signal(false);
  const addTab = signal<'upload' | 'folder'>('upload');
  function onfiles(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    void uploads.files([...(input.files ?? [])]);
    input.value = '';
  }

  // ---------- header ----------
  const phone = new MediaQuery(media.phone, false);
  /** Why a dataset cannot be made now, in words; null when it can. */
  const datasetWhy = computed(() => {
    const cur = job.get();
    return !cur ? null : running.get() ? 'Wait for the agent to finish' : cur.counts.labeled === 0 ? 'Label an image first' : null;
  });
  /** Everything is labeled and nothing runs: making the dataset is the one thing left to do. */
  const datasetNext = computed(() => {
    const cur = job.get();
    return !!cur && !datasetWhy.get() && cur.counts.images > 0 && toLabel(cur) === 0;
  });
  const editItems: MenuItem[] = [
    { label: 'Edit job', icon: Pencil, onclick: () => editOpen.set(true) },
    { label: 'Delete job', icon: Trash, danger: true, onclick: () => deleteOpen.set(true) },
  ];
  const phoneItems = computed<MenuItem[]>(() => [
    ...(!datasetNext.get() && !datasetWhy.get() ? [{ label: 'Create dataset', icon: Database, onclick: () => datasetOpen.set(true) }] : []),
    { label: 'Add images', icon: ImagePlus, onclick: () => (addTab.set('upload'), addOpen.set(true)) },
    ...editItems,
  ]);
  const st = computed(() => {
    const cur = job.get();
    return cur ? jobStatus(cur) : null;
  });
  const now = new Now(() => live.lost !== null);

  pageCommands((): PageCommand[] => {
    const cur = job.get();
    if (!cur) return [];
    const out: PageCommand[] = [{ id: 'label-add', label: 'Add images', icon: ImagePlus, run: () => (addTab.set('upload'), addOpen.set(true)) }];
    if (cur.status === 'running') out.push({ id: 'label-stop', label: `Stop labeling ${cur.name}`, match: ['Stop agent'], icon: Square, run: () => stopOpen.set(true) });
    if (!datasetWhy.get()) out.push({ id: 'label-dataset', label: 'Create dataset', icon: Database, run: () => datasetOpen.set(true) });
    return out;
  });

  return (
    <>
      <input ref={(e) => (fileInput = e)} type="file" accept="image/jpeg,image/png,image/bmp,image/webp" multiple hidden onchange={onfiles} />

      {unsupported.get() ? (
        <Page title="Label images">
          <Unsupported />
        </Page>
      ) : live.error ? (
        // Also with the job heard from the stream: without its images the page would show an empty job, no Retry.
        <Page
          title="Labeling job"
          actions={() => (
            <Button icon={ArrowLeft} href={href('/datasets/label')}>
              All labeling jobs
            </Button>
          )}
        >
          {live.error instanceof ApiError && live.error.status === 404 ? (
            <ErrorState error={live.error} title="Labeling job not found" />
          ) : (
            <ErrorState error={live.error} title="Could not load this labeling job" retry={() => live.retry()} />
          )}
        </Page>
      ) : !job.get() || !st.get() ? (
        <div class="sk-page" aria-busy="true" aria-label="Loading labeling job">
          <div class="sk-head">
            <Skeleton w="64px" h="18px" />
            <Skeleton w="220px" h="24px" />
            <Skeleton w="300px" h="14px" />
          </div>
          <div class="work">
            <div class="sk-main">
              <Skeleton h="28px" r="var(--radius-md)" />
              <div class="sk-tiles">
                <For each={TILES}>{() => <Skeleton h="auto" r="var(--radius-md)" class="sk-tile" image />}</For>
              </div>
            </div>
            <div class="sk-rail">
              <Skeleton h="520px" r="var(--radius-lg)" />
            </div>
          </div>
        </div>
      ) : (
        <>
          <Page
            title={j().name}
            inlineActions
            lead={() => (
              <>
                <StatusBadge plain state={st.get()!} lastKnown={live.lost !== null ? lastKnown(st.get()!.label, live.lost, now.current) : undefined} />
                {live.connection === 'reconnecting' ? (
                  <Badge tone="warning">
                    <WifiOff size={12} /> Reconnecting
                  </Badge>
                ) : null}
              </>
            )}
            sub={() => (
              <Meta>
                <span class="num">{imagesText(j().counts.images)}</span>
                <span class="num">{num(j().counts.labeled)} labeled</span>
                {j().counts.labeled ? (
                  <span class="num">
                    {num(reviewed.get())} of {num(j().counts.labeled)} reviewed
                  </span>
                ) : null}
                <span class="num">
                  {j().classes.length} {j().classes.length === 1 ? 'class' : 'classes'}
                </span>
                {made.get() !== 'none' ? (
                  <span data-made>
                    made{' '}
                    {made.get() === 'registered' ? (
                      <a class="link" href={href(`/datasets/${madeId.get()}`)}>
                        {madeDataset.get()?.name ?? 'a dataset'}
                      </a>
                    ) : made.get() === 'removed' ? (
                      <span title={`The dataset made from this job is no longer registered in FOLO. ${removedWhere.get()}`}>a dataset, since removed from FOLO</span>
                    ) : (
                      'a dataset'
                    )}
                  </span>
                ) : null}
                <span class="faint">created {ago(j().created)}</span>
                {datasetWhy.get() && !running.get() ? (
                  <span class="faint" data-gate-why>
                    Create dataset: {datasetWhy.get()!.toLowerCase()}
                  </span>
                ) : null}
              </Meta>
            )}
            subPhone={() => (
              <span class="num">{j().counts.images ? `${num(j().counts.labeled)} of ${imagesText(j().counts.images)} labeled` : 'No images yet'}</span>
            )}
            actions={() => (
              <>
                {phone.current ? (
                  <>
                    {datasetNext.get() ? (
                      <Button variant="primary" icon={Database} onclick={() => datasetOpen.set(true)}>
                        Create dataset
                      </Button>
                    ) : null}
                    <Menu label="More job actions" items={phoneItems.get()} />
                  </>
                ) : (
                  <>
                    <Gate reason={datasetWhy.get()}>
                      {(why) => (
                        <Button
                          variant={datasetNext.get() ? 'primary' : 'secondary'}
                          icon={Database}
                          disabled={!!why()}
                          aria-describedby={why()}
                          onclick={() => datasetOpen.set(true)}
                        >
                          Create dataset
                        </Button>
                      )}
                    </Gate>
                    <Menu label="More job actions" items={editItems} />
                  </>
                )}
              </>
            )}
          >
            <div class="work" class:review-first={datasetNext.get()}>
              <div class="main">
                <ImagesArea
                  jobId={j().id}
                  classes={j().classes}
                  images={live.images}
                  show={show.get()}
                  uploads={uploads}
                  onshow={(s) => setQuery({ show: s === 'all' ? null : s })}
                  onopen={(imageId) => setQuery({ image: imageId }, !imageParam.peek())}
                  onbrowse={() => fileInput?.click()}
                  onadd={(tab) => (addTab.set(tab), addOpen.set(true))}
                />
              </div>
              <aside class="rail" aria-label="Agent and log">
                <div class="agent">
                  <AgentPanel
                    job={j()}
                    agents={agents.data}
                    agentsError={agents.error}
                    retryAgents={() => agents.refresh()}
                    choice={choice}
                    bind:instructions={field(instructions, 'text')}
                    instructionsError={instructions.error}
                    instructionsSaved={instructions.saved}
                    starting={starting.get()}
                    startError={startError.get()}
                    lost={live.lost !== null}
                    onsaveinstructions={() => void instructions.save()}
                    onstart={() => void start()}
                    onagain={() => againOpen.set(true)}
                    onstop={() => stopOpen.set(true)}
                    onedit={() => editOpen.set(true)}
                  />
                </div>
                <div class="logs">
                  <Logs log={live.logsLog} version={live.version} live={running.get()} />
                </div>
              </aside>
            </div>
          </Page>

          <Editor live={live} job={j()} images={shown.get()} imageId={imageParam.get()} choice={choice} onnavigate={(to) => setQuery({ image: to })} />
          <AddImagesDialog bind:open={addOpen} bind:tab={addTab} uploads={uploads} onbrowse={() => fileInput?.click()} />
          <DatasetDialog bind:open={datasetOpen} job={j()} oncreated={created} />
          <JobDialog
            bind:open={editOpen}
            job={j()}
            hasBoxes={live.images.some((i) => i.boxes.length > 0)}
            onsave={(change) => live.act(() => hub.api.updateLabelJob(j().id, change))}
          />

          <Dialog
            bind:open={stopOpen}
            onbeforeclose={() => !stopBusy.peek()}
            title={`Stop labeling ${j().name}?`}
            description="The agent's calls in flight end, and the images they held keep the labels they had. What it has labeled so far stays."
            size="sm"
            status={stopStatus}
            footer={() => (
              <>
                <Button disabled={stopBusy.get()} onclick={() => !stopBusy.peek() && stopOpen.set(false)}>Keep labeling</Button>
                <Button variant="destructive" icon={Square} loading={stopBusy.get()} onclick={stop}>
                  Stop
                </Button>
              </>
            )}
          />
          <Dialog
            bind:open={againOpen}
            onbeforeclose={() => !starting.peek()}
            title={`Label all ${imagesText(j().counts.images)} again?`}
            description={`${choice.agent ? AGENT_NAME[choice.agent.id] : 'The agent'} labels every image again. The labels they have now, hand-drawn ones too, are replaced.`}
            size="sm"
            status={againStatus}
            footer={() => (
              <>
                <Button disabled={starting.get()} onclick={() => !starting.peek() && againOpen.set(false)}>Cancel</Button>
                <Button variant="destructive" icon={RotateCw} loading={starting.get()} onclick={() => start('all')}>
                  Label all again
                </Button>
              </>
            )}
          />
          <Dialog
            bind:open={deleteOpen}
            onbeforeclose={() => !deleteBusy.peek()}
            title={`Delete ${j().name}?`}
            description="Removes the job, its uploaded images and its labels. Images added from a folder stay on disk, and datasets made from this job are not touched."
            size="sm"
            status={deleteStatus}
            footer={() => (
              <>
                <Button disabled={deleteBusy.get()} onclick={() => !deleteBusy.peek() && deleteOpen.set(false)}>Cancel</Button>
                <Button variant="destructive" icon={Trash} loading={deleteBusy.get()} onclick={remove}>
                  Delete job
                </Button>
              </>
            )}
          />
        </>
      )}
    </>
  );
}

css`
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
  /* Below 1024 px one column: the agent first while labeling is still needed; once everything is labeled (review-first),
     images and the review filter lead so the review queue starts above the fold, with the agent below. */
  @media (max-width: 1023px) {
    .work {
      grid-template-columns: minmax(0, 1fr);
    }
    .rail {
      display: contents;
    }
    .work:not(.review-first) .agent {
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
`;
