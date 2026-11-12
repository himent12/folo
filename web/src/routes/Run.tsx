//
// A run's page. While the run trains its stream updates `live.run` and the step / epoch / log lists several times a
// second; what reads them here does work in proportion to what changed, not to the run's history: the charts read the
// append logs (run/metricSeries), the log keeps its level lists as lines arrive (run/Logs), the schedule warnings are
// looked for in new lines only (run/warningsLog, the Svelte page scanned the whole log on every step), and the parts
// that depend on the run's settings (Config, lineage, the dataset, the menus) get values that keep their identity
// while only its progress moves, so a step re-renders the progress and nothing else.
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
import { computed, css, effect, MediaQuery, signal, untrack } from 'fw';
import { hub } from '$lib/api/index';
import { LiveRun } from '$lib/api/live';
import { Resource } from '$lib/api/resource';
import { hasFeature, queuePlace, withQueueMove } from '$lib/api/queue';
import type { QueueRequest } from '$lib/api/types';
import { media } from '$lib/design/tokens';
import { ago, deviceName, eta } from '$lib/format';
import { timeRange } from '$lib/throughput';
import { FROM_SCRATCH_META, origin } from '$lib/lineage';
import { href, router } from '$lib/router';
import { setCrumbs } from '$lib/shell/crumbs';
import Page from '$lib/shell/Page';
import Badge from '$lib/ui/Badge';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import { DialogStatus } from '$lib/ui/feedback';
import Menu, { type MenuItem } from '$lib/ui/Menu';
import Tabs from '$lib/ui/Tabs';
import ErrorState from '$lib/ui/ErrorState';
import StatusBadge from '$lib/ui/StatusBadge';
import Ident from '$lib/ui/Ident';
import { toast } from '$lib/ui/toast';
import { tooltip } from '$lib/ui/tooltip';
import ConfigPanel from './run/ConfigPanel';
import { HEAD_LABEL } from './run/resolved';
import Logs from './run/Logs';
import Outcome, { canContinue } from './run/Outcome';
import Callout from './run/Callout';
import { fallbackNotice, runEvalSource, scoresTitle } from '$lib/evaluated';
import PerClass from './run/PerClass';
import Lineage from '$lib/ui/Lineage';
import Meta from '$lib/ui/Meta';
import RunMetrics from './run/RunMetrics';
import Samples from './run/Samples';
import { moveQueued, queueItems, queueWait, REMOVE_LABEL, removeDescription, removeQueued, removeTitle, startHint, startLabel, type Length } from './run/queueActions';
import { WarningsLog } from './run/warningsLog';
import { beforeTraining, phaseTitle, phaseWhat, providerName } from '$lib/cloud';
import { prefs } from '$lib/prefs';
import CloudPanel from './run/CloudPanel';
import { isOom } from '$lib/oom';
import { copyText, pageCommands, type PageCommand } from '$lib/shell/pageCommands';
import RunSkeleton from './run/RunSkeleton';
import WaitingPanels from './run/WaitingPanels';
import { wasResumed } from './run/resumes';
import { exactTime } from './runs/when';
import DeleteRun from './run/DeleteRun';

type Section = 'samples' | 'classes' | 'logs' | 'cloud' | 'config';
const NO_CLASSES: string[] = [];
const ACTIVE = ['queued', 'running', 'stopping'];

/** Two menus' items are the same when they show the same entries (their actions read the page's state when used). */
const sameItems = (a: MenuItem[], b: MenuItem[]) =>
  a.length === b.length && a.every((x, i) => x.label === b[i]!.label && x.href === b[i]!.href && x.icon === b[i]!.icon && x.danger === b[i]!.danger);

export default function Run() {
  const id = computed(() => router.route.params.id ?? '');
  const live = new LiveRun(() => id.get() || null);
  const datasets = new Resource((s) => hub.api.datasets(s));
  const run = computed(() => live.run);
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
  const modelClasses = computed(() => {
    const id = run.get()?.model_id;
    return id ? models.data?.find((m) => m.id === id)?.classes : undefined;
  });
  const classes = computed(() => dataset.get()?.classes ?? (recorded.get()?.classes.length ? recorded.get()!.classes : modelClasses.get()) ?? NO_CLASSES);
  // A dataset with no validation images trains and validates on its training split (docs/API.md, Split fallback): the
  // page says so once, as the Model page does, and its titles name that split rather than "Validation".
  const evalOn = computed(() => runEvalSource(dataset.get()));
  const splitNotice = computed(() => (evalOn.get() ? fallbackNotice(evalOn.get()!) : null));

  const crumbName = computed(() => run.get()?.name ?? id.get());
  effect(() => {
    setCrumbs({ label: 'Runs', path: '/runs' }, { label: crumbName.get() });
  });

  const others = new Resource((s) => hub.api.runs(s), { poll: 5000 });
  const stepCount = computed(() => {
    void live.version;
    return live.steps.length;
  });
  const queuedOnly = computed(() => status.get() === 'queued' && stepCount.get() === 0);
  /** The run loaded but its history (metrics) did not: its panels say so with Retry instead of "after the first epoch". */
  const historyError = computed(() => (run.get() ? live.error : undefined));
  const retryHistory = () => live.retry();

  // A cloud run (docs/CLOUD.md): where it runs, said by its provider's name ("RunPod · RTX 4090"), and its hand-over
  // in the waiting state's place while nothing trains yet.
  const isCloud = computed(() => !!cloud.get());
  const cloudProviders = new Resource((s) => (isCloud.get() ? hub.api.cloudProviders(s) : Promise.resolve(null)));
  const provider = computed(() => {
    const c = cloud.get();
    return c ? providerName(c.provider, cloudProviders.data ?? undefined) : '';
  });
  /** Before training a cloud run is the hand-over, not a wait in the training queue: its title is the phase. */
  const handover = computed(() => (cloud.get() && status.get() === 'queued' ? cloud.get()! : null));
  /** Stop on a run waiting in the training queue takes it out of the queue (a cloud run waits in none here). */
  const removing = computed(() => status.get() === 'queued' && !cloud.get());
  const ahead = computed(() => others.data?.find((r) => r.status === 'running' || r.status === 'stopping'));

  // Queue control (servers that list `queue_control`, docs/API.md): the run's place, Start next, Move up / down.
  const system = new Resource((s) => hub.api.system(s));
  const queueCtl = computed(() => hasFeature(system.data, 'queue_control'));
  // The run's `folo train` command on the server's machine (servers that list `run_cli`): Config shows it under "In
  // PowerShell:", and the palette copies it. Keyed by run, so another run's command never shows while it loads.
  const cli = new Resource(async (s) => {
    const rid = id.get();
    if (!rid || !hasFeature(system.data, 'run_cli')) return null;
    return { id: rid, cli: (await hub.api.runCli(rid, s)).cli };
  });
  const runCli = computed(() => (cli.data && cli.data.id === id.get() ? cli.data.cli : null));
  const place = computed(() => {
    const r = run.get();
    return r?.status === 'queued' && queueCtl.get() ? (r.queue_position ?? null) : null;
  });
  /** The runs waiting, front first (this one among them). */
  const queue = computed(() =>
    (others.data ?? []).filter((r) => r.status === 'queued' && r.queue_position != null).sort((a, b) => (a.queue_position ?? 0) - (b.queue_position ?? 0)),
  );
  const waitingCount = computed(() => Math.max(live.waiting ?? 0, queue.get().length, place.get() ?? 0));
  /** The waiting run just in front of this one. */
  const before = computed(() => {
    const p = place.get();
    return p && p > 1 ? queue.get().find((r) => r.id !== run.get()?.id && r.queue_position === p - 1) : undefined;
  });
  /** The waiting runs in front of this one, front first; keyed by their ids, so a poll that changes nothing asks nothing. */
  const inFront = computed(() => {
    const p = place.get();
    return p && p > 1 ? queue.get().filter((r) => r.id !== run.get()?.id && (r.queue_position ?? 0) < p) : [];
  });
  const frontKey = computed(() => inFront.get().map((r) => r.id).join(' '));
  // Their lengths, from the server's estimate of each one's settings (servers with `run_estimate`): what makes "starts
  // in about …" an estimate rather than a guess. One that cannot be estimated stays unknown, and the page says so.
  const frontLengths = new Resource(async (s) => {
    const key = frontKey.get();
    const rs = untrack(() => inFront.get());
    // No estimates on this server: every run in front stays unknown (one null each), never "nothing in front".
    if (!key || !hasFeature(system.data, 'run_estimate')) return { key, lengths: rs.map((): Length => null) };
    const lengths = await Promise.all(
      rs.map((r) =>
        hub.api.estimateRun({ ...r.config, name: r.name }, s).then(
          (e): Length => ({ lo: e.est_total_s_low, hi: e.est_total_s_high }),
          (): Length => null,
        ),
      ),
    );
    return { key, lengths };
  });
  /** Each run in front's length, in queue order (null: unknown, or not asked yet). */
  const lengths = computed<Length[]>(() => (frontLengths.data?.key === frontKey.get() ? frontLengths.data.lengths : inFront.get().map(() => null)));
  /** The wait: what the training run has left plus the lengths in front (`queueWait`); a floor when any is unknown. */
  const wait = computed(() => {
    const a = ahead.get();
    return queueWait(a ? a.eta_s : undefined, lengths.get());
  });
  /** The queued runs in front, together: one run's estimate, or their sum when every one is known. */
  const frontTime = computed(() => {
    const all = queueWait(undefined, lengths.get());
    return all?.known ? `about ${timeRange(all.lo, all.hi)}` : inFront.get().length > 1 ? 'lengths not known yet' : 'length not known yet';
  });
  /** Start next is a move only when something is in front (or nothing trains, and it starts at once). */
  const canStart = computed(() => place.get() !== null && (place.get()! > 1 || !ahead.get()) && !starting.get());
  /**
   * Nothing trains and the run is at the front of the queue (or the worker has taken it off): the worker is reading
   * its dataset and building its model, until the server reports it `running`.
   */
  const starting = computed(() => !cloud.get() && status.get() === 'queued' && !!others.data && !ahead.get() && (place.get() ?? 1) === 1);

  /** Moves this run in the queue: the new order at once (here and in the queue list), the old one back on a refusal. */
  async function move(req: QueueRequest) {
    const cur = run.peek();
    if (!cur) return;
    const id = cur.id;
    const refresh = () => {
      void others.refresh();
      hub.api.run(id).then(
        (r) => {
          // Asked for while the trainer was being built: a stream that has since said the run left the queue is newer,
          // and a run that left it never goes back (no status event would correct it until the run ends).
          if (r.status === 'queued' && live.run?.status !== 'queued') return;
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
      untrack(() => ahead.get()),
    );
    if (!r) return;
    if (live.run?.id === id) live.run = { ...live.run, queue_position: r.queue_position ?? null };
    void others.refresh();
  }

  // A queued run waits in a layout of its own at every width: its place and the way to the front, what it waits for,
  // then its config; the outputs it will have are one quiet list instead of empty frames. Phones show the callout and
  // the config alone: no section picker over sections with nothing behind them, no folded chart that opens empty.
  const waitingLayout = queuedOnly;
  const active = computed(() => (status.get() ? ACTIVE.includes(status.get()!) : false));
  const epochsDone = computed(() => {
    void live.version;
    return live.epochs.length;
  });
  // How long it took: from start to end, as the Runs list says; a run resumed after a stop counts its epochs' training
  // time instead, not the hours it sat stopped (the log says when it resumed, ./run/resumes).
  const elapsed = computed(() => {
    const r = run.get();
    if (!r?.started) return null;
    if (!active.get() && resumed.get()) {
      void live.version;
      const trained = live.epochs.reduce((a, e) => a + e.duration_s, 0);
      if (trained > 0) return trained;
    }
    return ((r.finished ? Date.parse(r.finished) : Date.now()) - Date.parse(r.started)) / 1000;
  });
  /** Its log says it was resumed (read once it has ended: a live run's duration is its wall time so far). */
  const resumed = computed(() => {
    if (active.get()) return false;
    void live.version;
    return wasResumed(live.logs);
  });
  // The trainer's "too short to learn" warnings, from the log (./run/warnings): one callout, never a toast. Read from
  // the lines added since the last update only (./run/warningsLog).
  const warningsLog = new WarningsLog();
  const warnings = computed(() => {
    void live.version;
    return warningsLog.sync(live.logsLog);
  });

  // Below 960 px one section shows at a time. A queued run has nothing to sample yet, so it opens on its Config (the
  // dependency and start estimate are in the banner above); once picked, the choice holds.
  const picked = signal<Section | null>(null);
  const section = computed<Section>(() => picked.get() ?? (queuedOnly.get() ? 'config' : 'samples'));
  const models = new Resource((s) => hub.api.models(s));
  /** Below 1024 px the parts are one section at a time behind tabs: then the parts' frame is their tab panel. */
  const stacked = new MediaQuery(media.upToLaptop, false);
  const phone = new MediaQuery(media.phone, false);
  /** Where the run started: "From scratch", or the model / file it was fine-tuned from (one reading, $lib/lineage). */
  const lineage = computed(() => {
    const c = config.get();
    return c ? origin(c.weights, models.data ?? []) : null;
  });
  const stopOpen = signal(false);
  const deleteOpen = signal(false);
  const busy = signal(false);
  // A confirm that fails says so in its dialog, which stays open for Retry (the toaster sits behind the modal); one
  // that succeeds closes it and the page says what happened.
  const stopStatus = new DialogStatus();

  async function stop() {
    const cur = run.peek();
    if (!cur) return;
    busy.set(true);
    try {
      if (removing.peek()) {
        // A run that never started is deleted (it leaves the list); a resumed one stops with what it trained.
        const r = await removeQueued(cur);
        stopOpen.set(false);
        toast.success(`Removed ${cur.name} from the queue`);
        if (r) live.run = r;
        else router.go('/runs');
        return;
      }
      const r = await hub.api.stopRun(cur.id);
      live.run = r;
      stopOpen.set(false);
      // The checkpoint is the last finished epoch's: none before the first one ends.
      const done = untrack(() => epochsDone.get());
      toast.info(
        `Stopping ${r.name}`,
        r.cloud && r.epoch === 0 ? 'Its instance is released; nothing had trained yet.' : done ? `The checkpoint from epoch ${done} is kept.` : 'No epoch had finished, so nothing is saved.',
      );
    } catch (e) {
      stopStatus.error(e, removing.peek() ? 'Could not remove the run from the queue' : 'Could not stop the run');
    } finally {
      busy.set(false);
    }
  }

  /**
   * The header's ⋯ (DESIGN_DIRECTION §13, one action set): the header shows one secondary, Compare (in the menu on
   * phones), and a running run's state action, Stop; the rest is here, the destructive action last on every detail
   * page (Remove from queue for a waiting run, Delete run for a finished one), each behind its confirm dialog. A run
   * with a model has Open model and Try in Playground here whatever its status, as its callout does.
   */
  const menuItems = (compare: boolean): MenuItem[] => {
    const r = run.get();
    if (!r) return [];
    const out: MenuItem[] = compare ? [{ label: 'Compare', icon: GitCompare, href: href(`/runs/compare?ids=${r.id}`) }] : [];
    if (r.model_id) {
      out.push({ label: 'Open model', icon: Box, href: href(`/models/${r.model_id}`) });
      out.push({ label: 'Try in Playground', icon: ScanSearch, href: href(`/playground?model=${r.model_id}`) });
    }
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
  const moreItems = computed(() => menuItems(true), { equals: sameItems });
  const wideItems = computed(() => menuItems(false), { equals: sameItems });

  /** What the run will have once it trains, in place of the empty frames (the waiting layout). */
  const later = computed(() => [
    { name: 'Metrics', what: 'mAP, precision and recall, loss, LR and throughput, by epoch' },
    { name: 'Samples', what: `Predictions on 16 ${scoresTitle(evalOn.get()).toLowerCase()} images after every epoch` },
    { name: 'Per class', what: `AP for each of ${classes.get().length || 'its'} classes` },
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
    if (r.status === 'stopped' || (r.status === 'failed' && !isOom(r.error)))
      out.push(
        canContinue(r)
          ? { id: 'run-resume', label: `Resume ${r.name}`, match: ['Resume run'], icon: Play, run: () => void resume() }
          : { id: 'run-resume', label: `Start ${r.name} again`, match: ['Start again', 'Resume run'], icon: RotateCcw, run: () => void resume() },
      );
    out.push({ id: 'run-compare', label: `Compare ${r.name}`, match: ['Compare run'], icon: GitCompare, run: () => router.go(`/runs/compare?ids=${r.id}`) });
    out.push({ id: 'run-copy-id', label: 'Copy run id', match: [r.id], icon: Copy, run: () => void copyText(r.id, 'run id') });
    const line = runCli.get();
    if (line) out.push({ id: 'run-copy-cli', label: 'Copy CLI command', match: ['folo train', 'PowerShell'], icon: Copy, run: () => void copyText(line, 'CLI command') });
    return out;
  });

  const resuming = signal(false);
  // page-level: Resume is a header and palette action, never inside a dialog.
  async function resume() {
    const cur = run.peek();
    if (!cur) return;
    resuming.set(true);
    try {
      const id = cur.id;
      const r = await live.resume(() => hub.api.resumeRun(id));
      toast.success(r.epoch > 0 ? `Resuming ${r.name}` : `Starting ${r.name} again`, r.epoch > 0 ? `From the checkpoint of epoch ${r.epoch}.` : 'No epoch had finished, so it starts over.');
    } catch (e) {
      // page-level: as above.
      toast.error(e, 'Could not resume the run');
    } finally {
      resuming.set(false);
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
            {wait.get()?.known ? <>Starts in about {timeRange(wait.get()!.lo, wait.get()!.hi)}, after</> : wait.get() ? <>Starts in more than {eta(wait.get()!.lo)}, after</> : 'Starts after'}
            {ahead.get() ? <>{' '}<a class="link" href={href(`/runs/${ahead.get()!.id}`)}>{ahead.get()!.name}</a>{' '}({ahead.get()!.eta_s !== null ? `${eta(ahead.get()!.eta_s)} left` : 'time left not known'}) and</> : null}
            {' '}
            {before.get() ? <><a class="link" href={href(`/runs/${before.get()!.id}`)}>{before.get()!.name}</a>{place.get()! > 2 ? <>{' '}and {place.get()! - 2} more</> : null}</> : <>{place.get()! - 1} queued {place.get() === 2 ? 'run' : 'runs'}</>}
            {' '}({frontTime.get()}).
          </>
        ) : ahead.get() ? (
          <>
            Starts when <a class="link" href={href(`/runs/${ahead.get()!.id}`)}>{ahead.get()!.name}</a> finishes{ahead.get()!.eta_s !== null ? <>, in about {eta(ahead.get()!.eta_s)}</> : null}.
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
        <Page title="Run">
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
            lead={() => (
              <>
                <StatusBadge run={R()} />
                <Ident value={R().id} class="faint id" />
                {/* Phones: the time joins the status, so the meta line below fits one row. */}
                <span class="faint when-phone">{when()}</span>
                {live.connection === 'reconnecting' ? (
                  <Badge tone="warning">
                    <WifiOff size={12} /> Reconnecting
                  </Badge>
                ) : null}
              </>
            )}
            sub={() => (
              // One line from 1024 px up, its last item ("started yesterday") never alone on a second one: below 1280
              // the size and batch step out (Config has them), below 1024 the time joins the status row and, on a line
              // with the precision or head items, the device steps out too. Phones keep the dataset, model and lineage.
              <Meta class={config.get()!.amp || config.get()!.head === 'o2o' ? 'run-meta extra' : 'run-meta'}>
                {datasetGone.get() ? (
                  // Removed from FOLO: its name as recorded, no link to a page that is gone.
                  <span title="This dataset is no longer registered in FOLO">{datasetName.get() ?? 'Removed dataset'} (removed)</span>
                ) : (
                  <a class="link" href={href(`/datasets/${config.get()!.dataset_id}`)}>
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
                  <span class="dev cloud">
                    {provider.get()} · {R().device ? deviceName(R().device) : cloud.get()!.offer.gpu}
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
              // One secondary (Compare), a running run's state action (Stop), and ⋯ with the rest, the destructive
              // action last (DESIGN_DIRECTION §13). Phones: Compare joins the ⋯.
              <>
                <span class="wide-only">
                  <Button icon={GitCompare} href={href(`/runs/compare?ids=${R().id}`)}>
                    Compare
                  </Button>
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
              system={system.data ?? (system.error ? null : undefined)}
            />

            {splitNotice.get() && !waitingLayout.get() ? (
              <Callout tone="warning" icon={TriangleAlert} title={splitNotice.get()!.title} role="note" label="Evaluation split">
                {splitNotice.get()!.why}
              </Callout>
            ) : null}

            {waitingLayout.get() ? (
              // Waiting: the config as asked for, beside the queue and what the run will have. Nothing here is an
              // empty frame, and no panel stretches to its neighbour's height. Below 1024 px the queue leads; phones
              // show the config's essentials, the rest folded (the work before the reference).
              <div class="waiting">
                <div class="w-main">
                  <ConfigPanel config={config.get()!} resolved={resolved.get()} datasetName={datasetName.get()} weights={lineage.get()} essential={phone.current} cli={runCli.get()} />
                </div>
                <div class="w-side">
                  {cloud.get() ? <CloudPanel cloud={cloud.get()!} provider={provider.get()} /> : null}
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
                {/* Phones: the chart is open or folded to its one-line readout by one remembered state, the same in
                    every section (RunMetrics chartFold). */}
                <RunMetrics
                  run={R()}
                  steps={live.stepsLog}
                  epochs={live.epochsLog}
                  version={live.version}
                  loading={live.loading}
                  queued={queuedOnly.get()}
                  evalOn={evalOn.get()}
                  error={historyError.get()}
                  retry={retryHistory}
                />

                {/* Evidence on the left (what the model sees, what the trainer says); reference in a rail on the right,
                    the two columns ending together. */}
                {/* Below 1024 px one of these shows at a time, chosen here; wider screens show them all beside a rail.
                    They switch content, so they are underline tabs (DESIGN_DIRECTION §2.22), not a segmented control. */}
                <div class="phone-pick">
                  <Tabs
                    label="Run sections"
                    idPrefix="run-sec"
                    value={section.get()}
                    onchange={(v) => picked.set(v)}
                    tabs={[
                      { value: 'samples', label: 'Samples' },
                      { value: 'classes', label: 'Classes' },
                      { value: 'logs', label: 'Logs' },
                      ...(isCloud.get() ? [{ value: 'cloud' as const, label: 'Cloud' }] : []),
                      { value: 'config', label: 'Config' },
                    ]}
                  />
                </div>
                <div
                  class="detail"
                  class:cloud={isCloud.get()}
                  data-show={section.get()}
                  id="run-sec-panel"
                  role={stacked.current ? 'tabpanel' : undefined}
                  aria-labelledby={stacked.current ? `run-sec-${section.get()}` : undefined}
                >
                  <div class="main-col">
                    <div class="part" data-part="samples">
                      <Samples
                        runId={R().id}
                        classes={classes.get()}
                        datasetGone={datasetGone.get()}
                        epochsDone={epochsDone.get()}
                        title={`${scoresTitle(evalOn.get())} samples`}
                        error={historyError.get()}
                        retry={retryHistory}
                      />
                    </div>
                    <div class="part" data-part="logs">
                      <Logs log={live.logsLog} version={live.version} live={active.get()} error={historyError.get()} retry={retryHistory} />
                    </div>
                  </div>
                  <aside class="rail" aria-label="Per-class results and config">
                    <div class="part" data-part="classes">
                      <PerClass classes={classes.get()} epochs={live.epochs} version={live.version} loading={live.loading} compact error={historyError.get()} retry={retryHistory} />
                    </div>
                    {cloud.get() ? (
                      <div class="part" data-part="cloud">
                        <CloudPanel cloud={cloud.get()!} provider={provider.get()} />
                      </div>
                    ) : null}
                    {/* One column in the rail (laptops up) and on phones; on a tablet the section is the panel's full
                        width, so it takes the two columns it has on a queued run's page (a panel at 768–1279 px uses
                        its width). */}
                    <div class="part" data-part="config">
                      <ConfigPanel
                        config={config.get()!}
                        resolved={resolved.get()}
                        datasetName={datasetName.get()}
                        weights={lineage.get()}
                        compact={!stacked.current || phone.current}
                        cli={runCli.get()}
                      />
                    </div>
                  </aside>
                </div>
              </>
            )}
          </Page>
          </div>

          <Dialog
            bind:open={stopOpen}
            title={removing.get() ? removeTitle(R().name) : `Stop ${R().name}?`}
            description={
              removing.get()
                ? removeDescription(R())
                : handover.get()
                  ? 'Its instance is released now; nothing has trained yet.'
                  : epochsDone.get() === 0
                    ? `Training stops after the current step. No epoch has finished yet, so nothing is saved${cloud.get() ? ' and the instance is released' : ''}.`
                    : cloud.get()
                      ? 'Training stops after the current step. The last checkpoint is copied here, saved as a model, and the instance released.'
                      : 'Training stops after the current step. The last checkpoint is kept and saved as a model.'
            }
            size="sm"
            status={stopStatus}
            footer={() => (
              <>
                <Button onclick={() => stopOpen.set(false)}>{removing.get() ? 'Keep it queued' : handover.get() ? 'Keep going' : 'Keep training'}</Button>
                <Button variant="destructive" icon={removing.get() ? X : CircleStop} loading={busy.get()} onclick={stop}>
                  {removing.get() ? REMOVE_LABEL : 'Stop run'}
                </Button>
              </>
            )}
          />
          <DeleteRun run={deleteOpen.get() ? R() : null} models={models.data} onclose={() => deleteOpen.set(false)} ondeleted={() => router.go('/runs')} />
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
  /* Beside the rail, the two columns share their rows (subgrid): Validation samples and Per class start and end
     together (the shorter one's frame stretches, as panels side by side do, DESIGN_DIRECTION §2.8), then the log and
     Config share the rest, so no panel bottom sits 100 px off its neighbour's (round 26). */
  @media (min-width: 1024px) {
    .detail {
      grid-template-rows: auto minmax(0, 1fr);
    }
    .main-col,
    .rail {
      grid-row: 1 / 3;
      grid-template-rows: subgrid;
    }
    /* A cloud run's rail has its Cloud panel between Per class and Config: the log spans it and Config, so the two
       columns still start and end together (the log is the part that grows). */
    .detail.cloud {
      grid-template-rows: auto auto minmax(0, 1fr);
    }
    .detail.cloud .main-col,
    .detail.cloud .rail {
      grid-row: 1 / 4;
    }
    .detail.cloud [data-part='logs'] {
      grid-row: span 2;
    }
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
    .detail:not([data-show='cloud']) [data-part='cloud'],
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
`;
