// A run's page (design/UNIFIED.md §4 Run): header, outcome callout, KPI strip, charts, samples, per-class, logs,
// cloud panel, and config. Live SSE updates do work proportional to deltas (append logs, incremental warnings & spot
// recovery, identity-preserving menus) so 10 Hz training ticks never re-render static panels.
import ArrowLeft from '$icons/arrow-left';
import Box from '$icons/box';
import ChevronsUp from '$icons/chevrons-up';
import GitCompare from '$icons/git-compare-arrows';
import ScanSearch from '$icons/scan-search';
import CircleStop from '$icons/circle-stop';
import Trash from '$icons/trash-2';
import X from '$icons/x';
import WifiOff from '$icons/wifi-off';
import TriangleAlert from '$icons/triangle-alert';
import Copy from '$icons/copy';
import Play from '$icons/play';
import RotateCcw from '$icons/rotate-ccw';
import CloudUpload from '$icons/cloud-upload';
import { computed, css, effect, MediaQuery, onCleanup, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { LiveRun } from '$lib/api/live';
import { modelsResource } from '$lib/api/models';
import { settledEta, trainingSeconds } from '$lib/api/progress';
import { Resource } from '$lib/api/resource';
import { runsChanged } from '$lib/api/runs';
import { hasFeature, queuePlace } from '$lib/api/queue';
import { archFromCli, isFolo2Arch } from '$lib/arch';
import { media } from '$lib/design/tokens';
import { ago, deviceName, eta } from '$lib/format';
import { timeRange } from '$lib/throughput';
import { FROM_SCRATCH_META, origin } from '$lib/lineage';
import { href, router } from '$lib/router';
import { setCrumbs } from '$lib/shell/crumbs';
import Page from '$lib/shell/Page';
import { noteRunEnded, watched } from '$lib/shell/runList';
import Badge from '$lib/ui/Badge';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import { DialogStatus } from '$lib/ui/feedback';
import Menu, { type MenuItem } from '$lib/ui/Menu';
import Tabs from '$lib/ui/Tabs';
import ErrorState from '$lib/ui/ErrorState';
import StatusBadge from '$lib/ui/StatusBadge';
import Ident from '$lib/ui/Ident';
import { SHORT } from './run/RunCharts';
import { toast } from '$lib/ui/toast';
import { tooltip } from '$lib/ui/tooltip';
import ConfigPanel from './run/ConfigPanel';
import { HEAD_LABEL } from './run/resolved';
import Logs from './run/Logs';
import Outcome, { canContinue, canResume } from './run/Outcome';
import { datasetPresence } from './overview/newDatasets';
import Callout from './run/Callout';
import { fallbackNotice, runEvalSource, scoresTitle } from '$lib/evaluated';
import PerClass from './run/PerClass';
import Lineage from '$lib/ui/Lineage';
import Meta from '$lib/ui/Meta';
import RunCharts from './run/RunCharts';
import RunChart from './run/RunChart';
import { ACTIVE, RunModel } from './run/runModel';
import Samples from './run/Samples';
import { queueItems, REMOVE_LABEL, removedToast, removeDescription, removeQueued, removeTitle, startHint, startLabel } from './run/queueActions';
import { queueState } from './run/queueState';
import { WarningsLog } from './run/warningsLog';
import { SpotRecoveryTracker } from './run/cloudRecovery';
import { beforeTraining, cloudGpu, phaseTitle, phaseWhat, providerName } from '$lib/cloud';
import { prefs } from '$lib/prefs';
import CloudPanel from './run/CloudPanel';
import { isOom } from '$lib/oom';
import { copyText, pageCommands, type PageCommand } from '$lib/shell/pageCommands';
import RunSkeleton from './run/RunSkeleton';
import WaitingPanels from './run/WaitingPanels';
import { exactTime } from './runs/when';
import DeleteRun from './run/DeleteRun';
import { goneEvalOn } from './run/goneEvalOn';
import { buildCheckpointCli, runExportMenuItems, runExportPageCommands } from './run/runExport';
import { stopDescription, stoppedToast } from './run/StopRun';

type Section = 'pr' | 'lr' | 'samples' | 'classes' | 'logs' | 'cloud' | 'config';
const NO_CLASSES: string[] = [];

/** Two menus' items are the same when they show the same entries (their actions read the page's state when used). */
const sameItems = (a: MenuItem[], b: MenuItem[]) =>
  a.length === b.length && a.every((x, i) => x.label === b[i]!.label && x.href === b[i]!.href && x.icon === b[i]!.icon && x.danger === b[i]!.danger);

export default function Run() {
  const id = computed(() => router.route.params.id ?? '');
  // Track the active run id this page is watching so terminal SSE (including during initial load before `/run` +
  // `/metrics` settle) notifies the shared run list immediately without firing on historical completed runs or waking
  // up on 10 Hz step ticks.
  let prevActiveId: string | undefined;
  const live = new LiveRun(() => id.get() || null, {
    onEnded: (r, terminal) => {
      const curId = r?.id ?? untrack(() => id.get());
      const listed = curId ? untrack(() => watched.runs?.find((x) => x.id === curId)) : undefined;
      const listedActive = !!listed && ACTIVE.includes(listed.status);
      if (r) {
        if (prevActiveId === curId || listedActive) noteRunEnded(r);
      } else if (listedActive) {
        noteRunEnded({ ...listed, status: terminal.status, error: terminal.error, queue_position: null });
      }
    },
  });
  effect(() => {
    const curId = live.runId;
    const curStatus = live.status;
    prevActiveId = curId && curStatus && ACTIVE.includes(curStatus) ? curId : undefined;
  });
  // Read again with the runs list (one tick): a dataset removed while the page is open shows as removed (stress s3-live-3).
  const datasets = new Resource((s) => hub.api.datasets(s), { poll: 5000 });
  const run = computed(() => {
    const r = live.run;
    const a = cliArch.get();
    return r && a && isFolo2Arch(a) && a !== r.config.arch ? { ...r, config: { ...r.config, arch: a } } : r;
  });
  /** The run's settings: the same object while only its progress moves (a step replaces the run, not its config). */
  const config = computed(() => run.get()?.config);
  const resolved = computed(() => run.get()?.resolved);
  const cloud = computed(() => run.get()?.cloud);
  const status = computed(() => run.get()?.status);
  const dataset = computed(() => {
    const c = config.get();
    return c ? datasets.data?.find((d) => d.id === c.dataset_id) : undefined;
  });
  // A run whose dataset was removed still names it and its classes: as recorded with the run (Run.dataset), for older
  // runs its model's class names; the per-class table says "class 0" only when none of them knows.
  const recorded = computed(() => run.get()?.dataset);
  /** The datasets are in and the run's is not among them: it was removed. */
  const datasetGone = computed(() => !!datasets.data && !dataset.get());
  const datasetName = computed(() => dataset.get()?.name ?? recorded.get()?.name);
  /** Whether the run's dataset is known removed (./overview/newDatasets): Outcome's Check dataset is then withdrawn. */
  const presence = datasetPresence(datasets, () => [config.get()?.dataset_id]);
  const modelClasses = computed(() => {
    const id = run.get()?.model_id;
    return id ? models.data?.find((m) => m.id === id)?.classes : undefined;
  });
  const classes = computed(() => dataset.get()?.classes ?? (recorded.get()?.classes.length ? recorded.get()!.classes : modelClasses.get()) ?? NO_CLASSES);
  // A dataset with no validation images trains and validates on its training split (docs/API.md, Split fallback): the
  // page says so once, as the Model page does, and its titles name that split rather than "Validation".
  // Once the dataset is removed, the trainer's log says which split the epochs were scored on (./run/goneEvalOn).
  const evalOn = computed(() => {
    if (!datasetGone.get()) return runEvalSource(dataset.get());
    void live.version;
    return goneEvalOn(datasetName.get() ?? null, live.logs);
  });
  const splitNotice = computed(() => (evalOn.get() ? fallbackNotice(evalOn.get()!) : null));

  const crumbName = computed(() => run.get()?.name ?? id.get());
  effect(() => {
    setCrumbs({ label: 'Runs', path: '/runs' }, { label: crumbName.get() });
  });

  const others = new Resource((s) => hub.api.runs(s), { poll: 5000, startFrom: () => watched.source });
  const stepCount = computed(() => {
    void live.version;
    return live.steps.length;
  });
  const queuedOnly = computed(() => status.get() === 'queued' && stepCount.get() === 0);
  /**
   * The run loaded but its history (metrics) did not: the metrics panel says so with the one Retry, the panels derived
   * from them (samples, per class) say they wait for it, and the log, a request of its own, shows what it got.
   */
  const historyError = computed(() => (run.get() ? live.error : undefined));
  const retryHistory = () => live.retry();
  // What the KPI strip and the four charts read, once for all of them: their series and the one x axis.
  const model = new RunModel({
    get run() {
      return live.run;
    },
    steps: live.stepsLog,
    epochs: live.epochsLog,
    get version() {
      return live.version;
    },
    get server() {
      return live.server;
    },
    get shortAnswers() {
      return live.shortAnswers;
    },
  });

  // A cloud run (docs/CLOUD.md): where it runs, said by its provider's name ("RunPod · RTX 4090"), and its hand-over
  // in the waiting state's place while nothing trains yet.
  const isCloud = computed(() => !!cloud.get());
  const cloudProviders = new Resource((s) => (isCloud.get() ? hub.api.cloudProviders(s) : Promise.resolve(null)));
  const provider = computed(() => {
    const c = cloud.get();
    return c ? providerName(c.provider, cloudProviders.data ?? undefined) : '';
  });
  /** Its provider launches an instance per run (a remote server does not); a provider not listed yet counts as one. */
  const provisions = computed(() => cloudProviders.data?.find((p) => p.id === cloud.get()?.provider)?.provisions ?? cloud.get()?.provider !== 'remote');
  /** Before training a cloud run is the hand-over, not a wait in the training queue: its title is the phase. */
  const handover = computed(() => (cloud.get() && status.get() === 'queued' ? cloud.get()! : null));
  /** Stop on a run waiting in the training queue takes it out of the queue (a cloud run waits in none here). */
  const removing = computed(() => status.get() === 'queued' && !cloud.get());
  // Queue control (servers that list `queue_control`, docs/API.md): the run's place, Start next, Move up / down.
  const system = new Resource((s) => hub.api.system(s));
  // The run's `folo train` command on the server's machine (servers that list `run_cli`): Config shows it under "In
  // PowerShell:", and the palette copies it. Keyed by run, so another run's command never shows while it loads.
  const cli = new Resource(async (s) => {
    const rid = id.get();
    if (!rid || !hasFeature(system.data, 'run_cli')) return null;
    const cmd = await hub.api.runCli(rid, s);
    return { id: rid, cli: cmd.cli, arch: archFromCli(cmd) };
  });
  const runCli = computed(() => (cli.data && cli.data.id === id.get() ? cli.data.cli : null));
  const cliArch = computed(() => (cli.data && cli.data.id === id.get() ? cli.data.arch : null));
  // Its place in the training queue, what trains ahead and when it should start; Start next and the moves (./run/queueState).
  const { ahead, place, queue, waitingCount, before, wait, frontTime, starting, canStart, move } = queueState({ run: () => run.get(), live, others, system });

  // A queued run waits in a layout of its own at every width: its place and the way to the front, what it waits for,
  // then its config; the outputs it will have are one quiet list instead of empty frames. Phones show the callout and
  // the config alone: no section picker over sections with nothing behind them, no folded chart that opens empty.
  const waitingLayout = queuedOnly;
  const active = computed(() => (status.get() ? ACTIVE.includes(status.get()!) : false));
  const epochsDone = computed(() => {
    void live.version;
    return live.epochs.length;
  });
  /**
   * It ended without a training step (interrupted before it started, failed before its first one, its dataset's folder
   * gone): the metrics say so in one line, and there are no samples or per-class results to wait for.
   */
  const neverTrained = computed(
    () => !active.get() && !live.loading && historyError.get() === undefined && stepCount.get() === 0 && epochsDone.get() === 0 && (run.get()?.step ?? 0) === 0,
  );
  // How long it took, as the Runs list says: from start to end, less the time it sat stopped before a resume.
  const elapsed = computed(() => {
    const r = run.get();
    return r ? trainingSeconds(r) : null;
  });
  const warningsLog = new WarningsLog();
  const warnings = computed(() => (void live.version, warningsLog.sync(live.logsLog)));
  const spotTracker = new SpotRecoveryTracker();
  const spotRecovery = computed(() => (void live.version, spotTracker.sync(live.logsLog, cloud.get(), run.get())));

  // Phones show one section at a time, Precision / Recall and Learning rate among them (design/UNIFIED.md §4 Run); a
  // tablet stacks them. A run that never trained opens on its log; once picked, the choice holds.
  const picked = signal<Section | null>(null);
  const models = modelsResource();
  /** Below 1024 px the parts stack in one column: Config then takes the panel's width. */
  const stacked = new MediaQuery(media.upToLaptop, false);
  /** Phones: the parts are one section at a time behind tabs, and the parts' frame is their tab panel. */
  const phone = new MediaQuery(media.phone, false);
  /** Phones: Precision / Recall and Learning rate are sections of the picker, not a row of charts. */
  const phoneCharts = computed(() => phone.current && !neverTrained.get() && historyError.get() === undefined);
  // A chart section picked on a phone falls back to the default once the page is wider (the charts are in their row).
  const section = computed<Section>(() => {
    const p = picked.get();
    if (p && ((p !== 'pr' && p !== 'lr') || phoneCharts.get())) return p;
    return queuedOnly.get() ? 'config' : neverTrained.get() ? 'logs' : 'samples';
  });
  /** Where the run started: "From scratch", or the model / file it was fine-tuned from (one reading, $lib/lineage). */
  const lineage = computed(() => {
    const c = config.get();
    return c ? origin(c.weights, models.data ?? []) : null;
  });
  /** A completed run's model: the header's primary (Open model), Try in Playground beside it. */
  const headerModel = computed(() => (status.get() === 'completed' ? (run.get()?.model_id ?? null) : null));
  const runModel = computed(() => { const mid = run.get()?.model_id; return mid ? models.data?.find((m) => m.id === mid) : undefined; });
  const checkpointCli = computed(() => { const r = run.get(); return r ? buildCheckpointCli(r, runModel.get(), dataset.get(), evalOn.get()) : null; });
  const hasPerClass = computed(() => (void live.version, (live.epochs[live.epochs.length - 1]?.val.per_class.length ?? 0) > 0));
  const exportItems = computed<MenuItem[]>(
    () => (run.get() && epochsDone.get() > 0 ? runExportMenuItems(() => run.peek(), () => live.epochs, () => classes.get(), () => datasetName.get(), hasPerClass.get()) : []),
    { equals: sameItems },
  );
  const stopOpen = signal(false);
  const deleteOpen = signal(false);
  const busy = signal(false);
  const resuming = signal(false);
  let mounted = true;
  let stopSeq = 0;
  let resumeSeq = 0;
  onCleanup(() => (mounted = false));
  // A confirm that fails says so in its dialog, which stays open for Retry (the toaster sits behind the modal); one
  // that succeeds closes it and the page says what happened.
  const stopStatus = new DialogStatus();
  let prevRunId = untrack(() => id.get());
  effect(() => {
    const curId = id.get();
    if (curId === prevRunId) return;
    prevRunId = curId;
    stopSeq++;
    resumeSeq++;
    untrack(() => {
      stopOpen.set(false);
      deleteOpen.set(false);
      stopStatus.clear();
      picked.set(null);
      busy.set(false);
      resuming.set(false);
    });
  });

  async function stop() {
    const cur = run.peek();
    if (!cur || busy.peek()) return;
    const seq = ++stopSeq;
    const wasRemoving = removing.peek();
    const startDone = epochsDone.peek();
    const startPath = router.location.path;
    busy.set(true);
    try {
      if (wasRemoving) {
        // A run that never started is deleted (it leaves the list); a resumed one stops with what it trained.
        const r = await live.stop(() => removeQueued(cur));
        if (mounted && seq === stopSeq) stopOpen.set(false);
        removedToast(cur);
        if (!r && mounted && seq === stopSeq && router.location.path === startPath) router.go('/runs', { replace: true });
        return;
      }
      // The run as the view now has it: the stream may have said it ended while the stop was on its way.
      const r = await live.stop(() => hub.api.stopRun(cur.id));
      runsChanged();
      if (!r) return;
      if (mounted && seq === stopSeq) stopOpen.set(false);
      // The checkpoint is the last finished epoch's: none before the first one ends.
      stoppedToast(r, untrack(() => (seq === stopSeq ? epochsDone.get() : startDone)));
    } catch (e) {
      const msg = wasRemoving ? 'Could not remove the run from the queue' : 'Could not stop the run';
      if (mounted && seq === stopSeq && stopOpen.peek()) stopStatus.error(e, msg);
      else toast.error(e, msg);
    } finally {
      if (seq === stopSeq) busy.set(false);
    }
  }

  /**
   * The header's ⋯ (DESIGN_DIRECTION §13, one action set): the header shows one secondary, Compare (in the menu on
   * phones), and a running run's state action, Stop; the rest is here, the destructive action last on every detail
   * page (Remove from queue for a waiting run, Delete run for a finished one), each behind its confirm dialog. A
   * completed run with a model has Open model as the header's primary and Try in Playground beside Compare (in place of
   * the completed banner, design/UNIFIED.md §4 Run); any other run with a model has both here. Phones keep the primary
   * and put Try in Playground in the ⋯.
   */
  const menuItems = (wide: boolean): MenuItem[] => {
    const r = run.get();
    if (!r) return [];
    const out: MenuItem[] = wide ? [] : [{ label: 'Compare', icon: GitCompare, href: href(`/runs/compare?ids=${r.id}`) }];
    if (r.model_id) {
      if (!headerModel.get()) out.push({ label: 'Open model', icon: Box, href: href(`/models/${r.model_id}`) });
      // Wherever the header does not show it (phones; a model that detected nothing, where the header leaves it out).
      if (!headerModel.get() || !wide || r.best?.map50_95 === 0) out.push({ label: 'Try in Playground', icon: ScanSearch, href: href(`/playground?model=${r.model_id}`) });
    }
    out.push({ label: 'New run like this', icon: Copy, href: href(`/runs/new?from=${r.id}`) });
    out.push(...exportItems.get());
    // The queue's one action set (`queueItems`), whole, as on the Runs list: the open menu covers the callout's Start
    // next, so the menu never relies on it. Remove from queue last.
    const removeQueued = () => stopOpen.set(true);
    const p = place.get();
    if (p !== null) out.push(...queueItems(p, waitingCount.get(), ahead.get(), move, removeQueued));
    // A cloud run waits in no queue here: the header's Stop releases its instance.
    else if (r.status === 'queued' && !r.cloud) out.push({ label: REMOVE_LABEL, icon: X, danger: true, onclick: removeQueued });
    if (!active.get()) out.push({ label: 'Delete run', icon: Trash, danger: true, onclick: () => deleteOpen.set(true) });
    return out;
  };
  // The same entries keep the same list: a step (which replaces the run) re-renders no menu.
  const moreItems = computed(() => menuItems(false), { equals: sameItems });
  const wideItems = computed(() => menuItems(true), { equals: sameItems });

  /** What the run will have once it trains, in place of the empty frames (the waiting layout). */
  const later = computed(() => [
    { name: 'Metrics', what: 'mAP, precision and recall, loss, LR and throughput, by epoch' },
    // Up to the trainer's 16 (docs/API.md `/samples`), never a count: it reopens the folder when the run starts. The
    // split is "validation" only on a dataset that has one; else (none, not reported, removed) a word true either way.
    { name: 'Samples', what: `Predictions on up to 16 ${(dataset.get()?.splits.val ?? 0) > 0 ? 'validation' : 'evaluation'} images after every epoch` },
    { name: 'Per class', what: classes.get().length === 1 ? 'AP for its one class' : `AP for each of ${classes.get().length || 'its'} classes` },
    { name: 'Logs', what: "The trainer's output, live" },
  ]);

  // Resume: the same run continues from weights/last.folo (POST /api/runs with resume_run_id, docs/API.md).
  // The palette's This page: the state action the header shows (Stop, Resume), Compare and the id.
  pageCommands((): PageCommand[] => {
    const r = run.get();
    if (!r) return [];
    const out: PageCommand[] = [];
    if (r.status === 'running' || (r.cloud && r.status === 'queued')) out.push({ id: 'run-stop', label: `Stop ${r.name}`, match: ['Stop run'], icon: CircleStop, run: () => stopOpen.set(true) });
    // As the run's banner: anything but out of memory continues from the last checkpoint.
    if ((r.status === 'stopped' || (r.status === 'failed' && !isOom(r.error))) && canResume())
      out.push(
        canContinue(r)
          ? { id: 'run-resume', label: `Resume ${r.name}`, match: ['Resume run'], icon: Play, run: () => void resume() }
          : { id: 'run-resume', label: `Start ${r.name} again`, match: ['Start again', 'Resume run'], icon: RotateCcw, run: () => void resume() },
      );
    out.push({ id: 'run-compare', label: `Compare ${r.name}`, match: ['Compare run'], icon: GitCompare, run: () => router.go(`/runs/compare?ids=${r.id}`) });
    out.push({ id: 'run-clone', label: 'New run like this', match: ['Clone run', 'Duplicate run', 'Copy settings', r.name], icon: Copy, run: () => router.go(`/runs/new?from=${r.id}`) });
    out.push({ id: 'run-copy-id', label: 'Copy run id', match: [r.id], icon: Copy, run: () => void copyText(r.id, 'run id') });
    const line = runCli.get();
    if (line) out.push({ id: 'run-copy-cli', label: 'Copy CLI command', match: ['folo train', 'PowerShell'], icon: Copy, run: () => void copyText(line, 'CLI command') });
    out.push(...runExportPageCommands(() => run.peek(), () => live.epochs, () => classes.get(), () => datasetName.get(), checkpointCli.get(), epochsDone.get() > 0, hasPerClass.get()));
    return out;
  });

  async function resume() {
    const cur = run.peek();
    if (!cur || resuming.peek()) return;
    const seq = ++resumeSeq;
    resuming.set(true);
    try {
      const id = cur.id;
      const r = await live.resume(() => hub.api.resumeRun(id));
      // page-level: Resume is a header and palette action, never inside a dialog.
      toast.success(r.epoch > 0 ? `Resuming ${r.name}` : `Starting ${r.name} again`, r.epoch > 0 ? `From the checkpoint of epoch ${r.epoch}.` : 'No epoch had finished, so it starts over.');
      runsChanged();
    } catch (e) {
      // page-level: as above.
      toast.error(e, 'Could not resume the run');
    } finally {
      if (seq === resumeSeq) resuming.set(false);
    }
  }

  const R = () => run.get()!;

  // A queued run leads with what it waits for and when it should start (the callout under the header); the frames it
  // will fill stay compact until there is data, and the config is in its rail.
  const waiting = () => {
    return (
      <>
        {handover.get() ? (
          phaseWhat(handover.get()!, provider.get(), prefs.units)
        ) : starting.get() ? (
          'Preparing data and building the model…'
        ) : place.get() !== null && place.get()! > 1 ? (
          // Behind other waiting runs: when it should start, then each run in front with its own time in brackets, in
          // queue order (what trains now, then the waiting ones), so the sum is checkable. A run whose length the server
          // cannot estimate makes the sum a floor ("more than 7m"), never an estimate of the whole (`queueWait`).
          <>
            {/* Each piece carries the space after it, so the sentence keeps its spaces however the pieces come and go
                (with a separate space between them, a training run that showed up after the page opened once read
                "and" run-on into the next name). A floor under a minute ("more than 1s") says nothing: left out. */}
            {wait.get()?.known ? <>Starts in {timeRange(wait.get()!.lo, wait.get()!.hi)}, after{' '}</> : wait.get() && wait.get()!.lo >= 60 ? <>Starts in more than {eta(wait.get()!.lo)}, after{' '}</> : 'Starts after '}
            {ahead.get() ? <><a class="link" href={href(`/runs/${ahead.get()!.id}`)}>{ahead.get()!.name}</a>{' '}({settledEta(ahead.get()!) !== null ? `${eta(settledEta(ahead.get()!)!)} left` : 'time left not known yet'}) and{' '}</> : null}
            {before.get() ? <><a class="link" href={href(`/runs/${before.get()!.id}`)}>{before.get()!.name}</a>{place.get()! > 2 ? <>{' '}and {place.get()! - 2} more</> : null}</> : <>{place.get()! - 1} queued {place.get() === 2 ? 'run' : 'runs'}</>}
            {' '}({frontTime.get()}).
          </>
        ) : ahead.get() ? (
          <>
            Starts when <a class="link" href={href(`/runs/${ahead.get()!.id}`)}>{ahead.get()!.name}</a> finishes{settledEta(ahead.get()!) !== null ? <>, in about {eta(settledEta(ahead.get()!)!)}</> : null}.
          </>
        ) : (
          'Starts as soon as the device is free.'
        )}
      </>
    );
  };

  const startNext = () => (
    <>
      {canStart.get() ? (
        <span class="start" use:tooltip={startHint(ahead.get())}>
          <Button size="sm" icon={ChevronsUp} onclick={() => move({ action: 'start' })}>
            {startLabel(ahead.get())}
          </Button>
        </span>
      ) : null}
    </>
  );

  // One word in every state ("created", as the Runs list's column), the exact times on hover.
  const when = () => (
    <time datetime={R().created} title={`Created ${exactTime(R().created)}${R().started ? ` · started ${exactTime(R().started)}` : ''}`}>
      created {ago(R().created)}
    </time>
  );

  return (
    <>
      {live.error && !run.get() ? (
        // The way back is the header's action, as on the other detail pages (Model, Dataset, Benchmark).
        <Page title="Run" actions={() => <Button icon={ArrowLeft} href={href('/runs')}>All runs</Button>}>
          <ErrorState error={live.error} title="Could not load this run" retry={() => live.retry()} />
        </Page>
      ) : !run.get() ? (
        <RunSkeleton />
      ) : (
        <>
          {/* A box of no layout of its own: what lets the styles below reach the page's title. */}
          <div class="run-page">
          <Page
            title={R().name}
            inlineActions
            badge={() => (
              // The status on the title's line (design/UNIFIED.md §2, Detail header); the run's id is in Config's header.
              // Its own row of items (the status, a phone's time, a connection warning), spaced and wrapping here.
              <span class="status-row">
                <StatusBadge run={R()} />
                {/* Phones: the time joins the status, so the meta line below fits one row. */}
                <span class="faint when-phone">{when()}</span>
                {live.connection === 'reconnecting' ? (
                  <Badge tone="warning">
                    <WifiOff size={12} /> Reconnecting
                  </Badge>
                ) : spotRecovery.get()?.kind === 'recovering' ? (
                  <Badge tone="warning">
                    <RotateCcw size={12} /> {spotRecovery.get()!.badge}
                  </Badge>
                ) : cloud.get()?.unanswered_since ? (
                  <Badge tone="warning">
                    <WifiOff size={12} /> Worker not answering
                  </Badge>
                ) : null}
              </span>
            )}
            sub={() => (
              // One line from 1024 px up, its last item ("started yesterday") never alone on a second one: below 1280
              // the size and batch step out (Config has them), below 1024 the time joins the status row and, on a line
              // with the precision or head items, the device steps out too. Phones keep the dataset, model and lineage.
              <Meta class={config.get()!.amp || config.get()!.head === 'o2o' ? 'run-meta extra' : 'run-meta'}>
                {/* The dataset's name is free text: longer than the line (a phone, a 64-character name) it is cut
                    with an ellipsis, whole in its tooltip (ui/Meta), never pushing the page sideways. */}
                {datasetGone.get() ? (
                  // Removed from FOLO: its name as recorded, no link to a page that is gone.
                  <span class="truncate" title={`${datasetName.get() ?? 'Removed dataset'} (removed): no longer registered in FOLO`}>
                    {datasetName.get() ?? 'Removed dataset'} (removed)
                  </span>
                ) : (
                  <a class="link truncate" href={href(`/datasets/${config.get()!.dataset_id}`)} title={datasetName.get() ?? config.get()!.dataset_id}>
                    {datasetName.get() ?? config.get()!.dataset_id}
                  </a>
                )}
                <Ident value={config.get()!.arch} />
                {/* Phones: size, batch and device are in the Config section; the id is in the URL. */}
                <span class="cfg num size">{config.get()!.imgsz} px</span>
                <span class="cfg num size">batch {config.get()!.batch}</span>
                <span class="cfg lineage">
                  {!lineage.get() || lineage.get()!.kind === 'scratch' ? (
                    FROM_SCRATCH_META
                  ) : (
                    <>
                      fine-tuned from <Lineage origin={lineage.get()!} />
                    </>
                  )}
                </span>
                {/* Only where they differ from the defaults (f32, the NMS head); Config always says both. */}
                {config.get()!.amp ? <span class="cfg">Mixed precision</span> : null}
                {config.get()!.head === 'o2o' ? <span class="cfg">{HEAD_LABEL.o2o}</span> : null}
                {cloud.get() ? (
                  // Where a cloud run runs: its provider and GPU (the worker's own name for it once it trains).
                  <span class="truncate dev cloud" title={`${provider.get()} · ${cloudGpu({ device: R().device, cloud: cloud.get()! })}`}>
                    {provider.get()} · {cloudGpu({ device: R().device, cloud: cloud.get()! })}
                  </span>
                ) : (
                  <span class="truncate dev" title={R().device}>
                    {deviceName(R().device)}
                  </span>
                )}
                <span class="faint when">{when()}</span>
              </Meta>
            )}
            actions={() => (
              // Compare (and a completed run's Try in Playground), a running run's state action (Stop), ⋯ with the rest,
              // the destructive action last (DESIGN_DIRECTION §13), and a completed run's Open model as the primary.
              // Phones: Compare and Try in Playground join the ⋯; the primary stays.
              <>
                <span class="wide-only">
                  <Button icon={GitCompare} href={href(`/runs/compare?ids=${R().id}`)}>
                    Compare
                  </Button>
                  {headerModel.get() && R().best?.map50_95 !== 0 ? (
                    <Button icon={ScanSearch} href={href(`/playground?model=${headerModel.get()}`)}>
                      Try in Playground
                    </Button>
                  ) : null}
                </span>
                {status.get() === 'running' || (cloud.get() && status.get() === 'queued') ? (
                  // Phones: a quiet icon button the size of ⋯ beside it, so Stop is never the largest control above the
                  // progress hero; its name says what it does, and the confirm stays. A cloud run can be stopped in its
                  // hand-over too: its instance costs money from the launch.
                  <>
                    <span class="wide-only">
                      <Button variant="danger" icon={CircleStop} onclick={() => stopOpen.set(true)}>
                        Stop
                      </Button>
                    </span>
                    <span class="phone-only">
                      <Button variant="danger" icon={CircleStop} aria-label={`Stop ${R().name}`} title="Stop run" onclick={() => stopOpen.set(true)} />
                    </span>
                  </>
                ) : null}
                {headerModel.get() ? (
                  <Button variant="primary" icon={Box} href={href(`/models/${headerModel.get()}`)}>
                    Open model
                  </Button>
                ) : null}
                {wideItems.get().length ? (
                  <span class="wide-only">
                    <Menu label="More run actions" items={wideItems.get()} />
                  </span>
                ) : null}
                <span class="phone-only">
                  <Menu label="More run actions" items={moreItems.get()} />
                </span>
              </>
            )}
          >
            <Outcome
              run={R()}
              elapsed={elapsed.get()}
              resuming={resuming.get()}
              onresume={resume}
              waiting={queuedOnly.get() ? waiting : undefined}
              waitingTitle={
                handover.get() ? phaseTitle(handover.get()!, provider.get()) : starting.get() ? 'Starting' : place.get() !== null ? queuePlace(place.get()!) : undefined
              }
              waitingIcon={handover.get() && beforeTraining(handover.get()!.phase) ? CloudUpload : undefined}
              waitingActions={canStart.get() ? startNext : undefined}
              warnings={warnings.get()}
              datasetName={datasetName.get()}
              datasetRemoved={presence(config.get()?.dataset_id) === 'removed'}
              system={system.data ?? (system.error ? null : undefined)}
            />

            {splitNotice.get() && !waitingLayout.get() ? (
              <Callout tone="warning" icon={TriangleAlert} title={splitNotice.get()!.title} role="note" label="Evaluation split">
                {splitNotice.get()!.why}
              </Callout>
            ) : null}
            {model.shortNote.get() && !waitingLayout.get() ? (
              <Callout tone="warning" icon={TriangleAlert} title={model.shortNote.get()!.title} role="note" label="Run history">
                {model.shortNote.get()!.why}
              </Callout>
            ) : null}

            {waitingLayout.get() ? (
              // Waiting: the config as asked for, beside the queue and what the run will have. Nothing here is an
              // empty frame, and no panel stretches to its neighbour's height. Below 1024 px the queue leads; phones
              // show the config's essentials, the rest folded (the work before the reference).
              <div class="waiting">
                <div class="w-main">
                  <ConfigPanel config={config.get()!} resolved={resolved.get()} datasetName={datasetName.get()} weights={lineage.get()} essential={phone.current} cli={runCli.get()} runId={R().id} />
                </div>
                <div class="w-side">
                  {cloud.get() ? <CloudPanel cloud={cloud.get()!} provider={provider.get()} run={R()} provisions={provisions.get()} recovery={spotRecovery.get()} /> : null}
                  <WaitingPanels
                    queue={place.get() !== null ? queue.get() : []}
                    ahead={ahead.get()}
                    self={R().id}
                    waiting={waitingCount.get()}
                    later={later.get()}
                  />
                </div>
              </div>
            ) : (
              <>
                <RunCharts
                  model={model}
                  loading={live.loading}
                  evalOn={evalOn.get()}
                  error={historyError.get()}
                  retry={retryHistory}
                  never={neverTrained.get()}
                  phone={phone.current}
                />

                {/* Evidence on the left (what the model sees, what the trainer says); reference on the right (Per class,
                    Config), the two columns ending together. */}
                {/* Phones show one of these at a time, chosen here; a tablet stacks them, wider screens show them two to
                    a row. They switch content, so they are underline tabs (DESIGN_DIRECTION §2.22), not a segmented
                    control. */}
                {/* A deliberate scroller (`data-scrolls`): a cloud run's seven sections do not fit a 320 px phone. */}
                <div class="phone-pick" data-scrolls>
                  <Tabs
                    label="Run sections"
                    idPrefix="run-sec"
                    value={section.get()}
                    onchange={(v) => picked.set(v)}
                    tabs={[
                      // Phones: the second row of charts is two sections here (design/UNIFIED.md §4 Run, smaller tiers).
                      ...(phoneCharts.get()
                        ? [
                            { value: 'pr' as const, label: 'P / R' },
                            { value: 'lr' as const, label: 'LR' },
                          ]
                        : []),
                      ...(neverTrained.get()
                        ? []
                        : [
                            { value: 'samples' as const, label: 'Samples' },
                            { value: 'classes' as const, label: 'Classes' },
                          ]),
                      { value: 'logs', label: 'Logs' },
                      ...(isCloud.get() ? [{ value: 'cloud' as const, label: 'Cloud' }] : []),
                      { value: 'config', label: 'Config' },
                    ]}
                  />
                </div>
                <div
                  class="detail"
                  class:cloud={isCloud.get()}
                  class:never={neverTrained.get()}
                  data-show={section.get()}
                  id="run-sec-panel"
                  role={phone.current ? 'tabpanel' : undefined}
                  aria-labelledby={phone.current ? `run-sec-${section.get()}` : undefined}
                >
                  {/* The parts in reading order, the order a phone's picker lists them and a tablet stacks them: from
                      1024 px the grid places them two to a row (Samples | Per class, Logs | Config), so Tab follows
                      what is seen at every tier. Each is a named region (its panel's title). */}
                  {phoneCharts.get() ? (
                    <>
                      <div class="part" data-part="pr">
                        <RunChart kind="pr" model={model} height={SHORT} loading={live.loading} />
                      </div>
                      <div class="part" data-part="lr">
                        <RunChart kind="lr" model={model} height={SHORT} loading={live.loading} />
                      </div>
                    </>
                  ) : null}
                  {/* A run that never trained has no samples or per-class results: the log and Config are its row. */}
                  {!neverTrained.get() ? (
                    <>
                      <div class="part" data-part="samples">
                        <Samples
                          runId={R().id}
                          classes={classes.get()}
                          datasetGone={datasetGone.get()}
                          epochsDone={epochsDone.get()}
                          ended={model.unvalidated.get()}
                          pending={model.pending.get()}
                          title={`${scoresTitle(evalOn.get())} samples`}
                          unavailable={historyError.get() !== undefined}
                        />
                      </div>
                      <div class="part" data-part="classes">
                        <PerClass
                          classes={classes.get()}
                          epochs={live.epochs}
                          version={live.version}
                          loading={live.loading || model.pending.get()}
                          compact
                          unavailable={historyError.get() !== undefined}
                          ended={model.unvalidated.get()}
                          exportItems={exportItems.get()}
                        />
                      </div>
                    </>
                  ) : null}
                  <div class="part" data-part="logs">
                    <Logs log={live.logsLog} version={live.version} live={active.get()} earlier={live.logsEarlier} />
                  </div>
                  {cloud.get() ? (
                    <div class="part" data-part="cloud">
                      <CloudPanel cloud={cloud.get()!} provider={provider.get()} run={R()} provisions={provisions.get()} recovery={spotRecovery.get()} />
                    </div>
                  ) : null}
                  {/* One column beside the log (laptops up) and on phones; on a tablet it is stacked at the panel's full
                      width, so it takes the two columns it has on a queued run's page (a panel at 768–1279 px uses its
                      width). */}
                  <div class="part" data-part="config">
                    <ConfigPanel
                      config={config.get()!}
                      resolved={resolved.get()}
                      datasetName={datasetName.get()}
                      weights={lineage.get()}
                      compact={!stacked.current || phone.current}
                      cli={runCli.get()}
                      checkpointCli={checkpointCli.get()}
                      runId={R().id}
                    />
                  </div>
                </div>
              </>
            )}
          </Page>
          </div>

          <Dialog
            bind:open={stopOpen}
            onbeforeclose={() => !busy.peek()}
            title={removing.get() ? removeTitle(R().name) : `Stop ${R().name}?`}
            description={removing.get() ? removeDescription(R()) : stopDescription(R(), epochsDone.get())}
            size="sm"
            status={stopStatus}
            footer={() => (
              <>
                <Button disabled={busy.get()} onclick={() => !busy.peek() && stopOpen.set(false)}>
                  {removing.get() ? 'Keep it queued' : handover.get() ? 'Keep going' : 'Keep training'}
                </Button>
                <Button variant="destructive" icon={removing.get() ? X : CircleStop} loading={busy.get()} onclick={stop}>
                  {removing.get() ? REMOVE_LABEL : 'Stop run'}
                </Button>
              </>
            )}
          />
          <DeleteRun run={deleteOpen.get() ? R() : null} onclose={() => deleteOpen.set(false)} ondeleted={() => router.go('/runs', { replace: true })} />
        </>
      )}
    </>
  );
}

css`
  .run-page {
    display: contents;
  }
  /* A long name (run names were not capped before 64 characters) takes two lines at most, broken anywhere, never a
     wall of four; the breadcrumb and Config have it too. */
  .run-page :global(h1[data-route-focus]) {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    overflow: hidden;
    overflow-wrap: anywhere;
  }
  .status-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-1) var(--space-2);
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
    /* Tablets: the queue (a cloud run's hand-over) leads, then the config, then the outputs to come. */
    .w-side {
      display: contents;
    }
    .w-side :global(#queue),
    .w-side :global(#cloud) {
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
  /* Rows 4 and 5 (design/UNIFIED.md §4 Run): one grid of the parts in reading order. From 1024 px it is two columns,
     filled row by row, so Validation samples and Per class start and end together (the shorter frame stretches, as
     panels side by side do, DESIGN_DIRECTION §2.8), then the log and Config share the rest, so no panel bottom sits
     100 px off its neighbour's (round 26): the log (at least 200 px) grows to Config's bottom, or Config's frame runs
     down to the log's end. No fixed heights. Below 1024 px one column, the DOM's order: Validation samples, Per class,
     Logs, a cloud run's Cloud, Config. */
  .detail {
    --log-h: 200px;
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--space-4);
    align-items: stretch;
  }
  .part {
    display: grid;
    min-width: 0;
  }
  @media (min-width: 1024px) {
    .detail {
      grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
      grid-template-rows: auto minmax(0, 1fr);
    }
    /* A cloud run has its Cloud panel between Per class and Config: the log spans it and Config, so the two columns
       still start and end together (the log is the part that grows). */
    .detail.cloud {
      grid-template-rows: auto auto minmax(0, 1fr);
    }
    .detail.cloud [data-part='logs'] {
      grid-row: span 2;
    }
    /* A run that never trained: no samples / per-class row, so the log and Config (a cloud run's Cloud panel above
       Config) are the rows. */
    .detail.never {
      grid-template-rows: minmax(0, 1fr);
    }
    .detail.cloud.never {
      grid-template-rows: auto minmax(0, 1fr);
    }
  }
  .phone-pick {
    display: none;
  }
  @media (max-width: 767px) {
    /* Phones: one section at a time, picked above. */
    /* The section picker stays reachable while the section scrolls (just under the top bar). */
    /* Flex, not block: the segmented group is inline, and a line box would add its descender under it. */
    .phone-pick {
      display: flex;
      position: sticky;
      top: var(--size-topbar);
      z-index: var(--z-sticky);
      margin-top: calc(-1 * var(--space-2));
      background: var(--bg-app);
      /* 12 px to the section under the page's 16 px gap. */
      margin-bottom: calc(var(--space-3) - var(--space-4));
    }
    /* One part is visible at a time, so the hidden ones must not add gaps. */
    .detail {
      gap: 0;
    }
    .detail:not([data-show='pr']) [data-part='pr'],
    .detail:not([data-show='lr']) [data-part='lr'],
    .detail:not([data-show='samples']) [data-part='samples'],
    .detail:not([data-show='classes']) [data-part='classes'],
    .detail:not([data-show='logs']) [data-part='logs'],
    .detail:not([data-show='cloud']) [data-part='cloud'],
    .detail:not([data-show='config']) [data-part='config'] {
      display: none;
    }
    /* One section at a time: the log keeps its own height. */
    .detail {
      --log-h: 300px;
    }
  }
  /* The header's meta line holds one row at every tier: a middle item steps out before the last one would wrap alone. */
  @media (max-width: 1279px) {
    .size {
      display: none;
    }
  }
  @media (min-width: 768px) and (max-width: 1023px) {
    .when {
      display: none;
    }
    .when-phone {
      display: inline;
    }
    :global(.run-meta.extra > .dev) {
      display: none;
    }
  }
  @media (max-width: 767px) {
    .phone-pick :global([role='tab']) {
      flex: 1;
      justify-content: center;
    }
    .dev,
    .cfg,
    .when {
      display: none;
    }
    .when-phone {
      display: inline;
    }
  }
`;
