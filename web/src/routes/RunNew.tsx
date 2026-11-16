// New run (`#/runs/new`): the training form (dataset, model, training, augmentation), its estimate, the matching
// `folo train` command, and Start. The parts are in ./runnew/.
import Images from '$icons/images';
import Play from '$icons/play';
import Plus from '$icons/plus';
import { computed, css, deep, effect, Key, onCleanup, signal, tick, untrack } from 'fw';
import { ApiError, hub } from '$lib/api/index';
import { readDatasets } from '$lib/api/datasetReads';
import { Resource } from '$lib/api/resource';
import type { Arch, Augment, DatasetImage, Optimizer, TrainConfig } from '$lib/api/types';
import { modelChoices, scoreMeta } from '$lib/scoreSummary';
import { sentence } from '$lib/format';
import { hasCloud } from '$lib/cloud';
import { fitFix, IMGSZ_STEPS, isOom, oomCalibration, useLabel, vramForecaster, type OomFix } from '$lib/oom';
import { REF_CPU, REF_GPU } from '$lib/reference';
import { hasFeature } from '$lib/throughput';
import { href, router } from '$lib/router';
import { serverCommand, trainCommand } from '$lib/trainCli';
import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs';
import { watched } from '$lib/shell/runList';
import Page from '$lib/shell/Page';
import Button from '$lib/ui/Button';
import Dialog from '$lib/ui/Dialog';
import EmptyState from '$lib/ui/EmptyState';
import ErrorState from '$lib/ui/ErrorState';
import Field from '$lib/ui/Field.tsx';
import Input from '$lib/ui/Input';
import Panel from '$lib/ui/Panel';
import { toast } from '$lib/ui/toast';
import { tooltip } from '$lib/ui/tooltip';
import { CloudPick } from './runnew/cloudPick';
import { arches, autoCloseMosaic, closeMosaicEpochs, defaultImgsz, estimate, shortRun, warmup } from './runnew/estimate';
import { ampBlockOf } from './runnew/amp';
import { firstError, formErrors, isInt, landOn, type Jump } from './runnew/jump';
import { cloudCostOf, closeMosaicShown, reviewRows, roughBytes, shownEstimate } from './runnew/summary';
import { trainConfig } from './runnew/config';
import { deviceLabelOf, deviceMessage, missingDevice } from './runnew/device';
import { oomChangesOf } from './runnew/retry';
import { serverEstimate } from './runnew/serverEstimate';
import { clearOwnDraft, readDraft, saveDraft, type Draft } from './runnew/draft';
import { clipName, MAX_RUN_NAME, nameHint, runName, uniqueName } from './runnew/name';
import WhyLine from './runnew/WhyLine';
import { START_KEYS, startKeys } from './runnew/startKeys';
import { aheadLine, aheadOf, startedToast } from './runnew/queue';
import { queueToastKey } from './run/queueActions';
import { isPublished, otherSizeChoices, scoreContextOf, startHintOf } from './runnew/weights';
import { sampleImagesOf, trainSizesOf } from './runnew/trainImages';
import DatasetSection from './runnew/DatasetSection';
import ModelSection from './runnew/ModelSection';
import TrainingSection from './runnew/TrainingSection';
import AugmentSection from './runnew/AugmentSection';
import ReviewList from './runnew/ReviewList';
import EstimateFigures from './runnew/EstimateFigures';
import CliCommand from './runnew/CliCommand';

/** The toast of a `?from=` run that could not be loaded (Toast.key). */
const FROM_TOAST = 'runnew-from';

import OomNote from './runnew/OomNote';
import StartDock from './runnew/StartDock';

/**
 * New run follows its own URL while open (v-data-runs N1): a link that changes it (the palette's "New run on X", a
 * pasted link, Back / Forward) opens the form again under the rules of a first open, the draft included (that same
 * link's draft whole, else the link's fields over the draft's). The form never writes its own URL, so this cannot loop.
 */
export default function RunNew() {
  let shown = '';
  // Leaving (a start goes to the run), the route is another screen's: the form stays as it is until it is gone.
  const link = computed(() => (router.route.name === 'run-new' ? (shown = String(router.route.query)) : shown));
  return <Key when={link.get()}>{() => <RunNewForm />}</Key>;
}

function RunNewForm() {
  setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'New run' });
  // This page is the create action: the top bar's New run steps aside instead of showing a pressed, disabled copy.
  ownCreateAction();

  const q = untrack(() => router.route.query);
  const defaults: Augment = { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 };
  const fromId = q.get('from');
  // The tab's draft (runnew/draft): whole when it was made under this link (Back, a reload), else under the fields this
  // link sets. A copy of a run (?from=) sets them all, so it starts from the run, not the draft.
  const query = q.toString();
  const stored = readDraft(hub.server);
  const own = stored?.query === query ? stored : null;
  const draft = own ?? (fromId ? null : stored);

  const datasetId = signal<string>((own ? own.datasetId : q.get('dataset')) ?? draft?.datasetId ?? '');
  // The dock repeats the summary's Start button for narrow screens; it steps aside while that button is on screen.
  const startEl = signal<HTMLElement>();
  const formEl = signal<HTMLFormElement>();
  startKeys(() => formEl.get());
  const startVisible = signal(false);
  effect(() => {
    const el = startEl.get();
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const io = new IntersectionObserver(([e]) => startVisible.set(!!e?.isIntersecting));
    io.observe(el);
    return () => io.disconnect();
  });
  const arch = signal<Arch>(draft?.arch ?? 'folo-s');
  /** Number inputs bind `null` when cleared (the form's errors say so). */
  const epochs = signal<number>(draft ? (draft.epochs as number) : 100);
  const batch = signal(draft?.batch ?? '16');
  const IMGSZ = IMGSZ_STEPS.map(String);
  // ?imgsz= (e.g. from a dataset's "Train at 960 px" health finding) preselects a supported size. Until one is chosen
  // (there, by hand or by a retry) the size follows the dataset (see `defaultImgsz`).
  const queryImgsz = !own && IMGSZ.includes(q.get('imgsz') ?? '');
  const imgsz = signal(queryImgsz ? q.get('imgsz')! : (draft?.imgsz ?? '640'));
  const imgszChosen = signal(queryImgsz || !!draft?.imgszChosen);
  const device = signal<string>(draft?.device ?? 'auto');
  const optimizer = signal<Optimizer>(draft?.optimizer ?? 'auto');
  /** Mixed precision as asked for; what the run gets is `ampOn` (off where the server cannot train that way). */
  const amp = signal(draft?.amp ?? false);
  /** The end-to-end head (TrainConfig.head `o2o`, docs/MODEL.md §6.6): predicts without NMS. */
  const e2e = signal(draft?.e2e ?? false);
  const lr0 = signal(draft ? (draft.lr0 as number) : 0.01);
  const seed = signal(draft ? (draft.seed as number) : 0);
  const weights = signal<string>(draft?.weights ?? 'none');
  const name = signal(draft?.name ?? '');
  const nameEdited = signal(draft?.nameEdited ?? false);
  // Edited in place by the controls (a `deep` object, as Svelte's `$state`), replaced whole by a prefill or Defaults.
  const augment = signal<Augment>(deep(structuredClone({ ...defaults, ...draft?.augment })));
  // Close mosaic follows the epoch count (default 10, never more than half the run, the trainer's own cap) until it is
  // set by hand; the trainer caps a hand-set value the same way, so a short run always trains with mosaic.
  const closeEdited = signal(draft?.closeEdited ?? false);
  effect(() => {
    const e = epochs.get();
    if (closeEdited.get() || !Number.isFinite(e) || e < 1) return;
    const want = autoCloseMosaic(e);
    if (untrack(() => augment.get().close_mosaic) !== want) augment.get().close_mosaic = want;
  });
  const submitting = signal(false);
  /** The short-run warning's reasons, folded behind its "Why?". */
  const shortWhy = signal(false);
  const advancedOpen = signal(draft?.advancedOpen ?? false);
  /** The augmentation settings fold behind a one-line summary at every width until opened. */
  const augOpen = signal(draft?.augOpen ?? false);
  /** Below 960 px the dock opens the summary as a sheet, so the run can be reviewed without scrolling down. */
  const sumOpen = signal(false);

  // Each read for its stats; one whose folder is missing is marked, and Start waits (lib/api/datasetReads).
  const datasets = new Resource(readDatasets);
  const system = new Resource((s) => hub.api.system(s));
  // Run on (servers with the `cloud` feature): this machine or a cloud connection, its GPU and the time limit.
  const cloudPick = new CloudPick(() => system.data);
  if (draft) cloudPick.restore(draft.cloud);
  const cloudOn = computed(() => hasCloud(system.data));
  /** A connection is chosen: the run trains on its GPU, not this machine's. */
  const onCloud = computed(() => cloudOn.get() && cloudPick.cloud);
  const models = new Resource((s) => hub.api.models(s));
  const runs = new Resource((s) => hub.api.runs(s));
  // While another run holds this machine, a new one waits in the queue: the Summary says so and Start reads "Queue run".
  // The shell's list (polled every 2 s) is the current one; this page's own stands in until it arrives.
  const ahead = computed(() => aheadOf(watched.runs ?? runs.data, onCloud.get()));
  const startLabel = computed(() => (ahead.get() ? 'Queue run' : 'Start training'));
  /** Every run's name, so a new one is told apart (runnew/name). */
  const taken = computed(() => new Set((runs.data ?? []).map((r) => r.name)));
  // This server's measured speed (GET /api/system/throughput), where it has one: measured latency in the size table.
  const tp = new Resource((s) => (hasFeature(system.data, 'throughput') ? hub.api.throughput(s).catch(() => null) : Promise.resolve(null)));

  // Prefill from an earlier run (?from=run_x, e.g. "Retry with batch 32"). Loaded through the hub, so it waits for
  // the real server instead of asking the mock that answers before the connection settles.
  const source = new Resource((s) => (fromId ? hub.api.run(fromId, s) : Promise.resolve(null)));
  // A draft of this same link already holds the copy, and the edits since.
  const prefilled = signal(!fromId || !!own);
  effect(() => {
    const r = source.data;
    // The runs too (or their error), so the copy's name is a new one.
    if (!r || (!runs.data && !runs.error) || untrack(() => prefilled.get())) return;
    const c = r.config;
    // A retry's fix comes in the query (?batch=, ?imgsz=, ?arch=, ?device=; see $lib/oom) and wins over the run's.
    // A value the trainer would reject is ignored (the run's own stays), never offered with Start enabled.
    const whole = (k: string, lo: number, hi: number, step = 1) => {
      const v = Number(q.get(k));
      return Number.isInteger(v) && v >= lo && v <= hi && v % step === 0 ? String(v) : null;
    };
    datasetId.set(c.dataset_id);
    arch.set(arches.find((a) => a.arch === q.get('arch'))?.arch ?? c.arch);
    epochs.set(Number(whole('epochs', 1, 10000) ?? c.epochs)); // e.g. a short run's "Train N epochs"
    batch.set(whole('batch', 1, 1024) ?? String(c.batch));
    imgsz.set(whole('imgsz', 64, 4096, 32) ?? String(c.imgsz)); // the trainer's limits (docs/API.md, runs)
    imgszChosen.set(true);
    device.set(q.get('device') === 'cpu' ? 'cpu' : c.device);
    optimizer.set(c.optimizer);
    amp.set(c.amp ?? false);
    e2e.set(c.head === 'o2o');
    lr0.set(c.lr0);
    seed.set(c.seed);
    weights.set(c.weights ?? 'none');
    // A cloud run's connection, GPU and limit, where they still exist (else this machine, and Run on says why).
    cloudPick.prefill(c.compute, r.cloud?.offer.gpu ?? 'GPU');
    augment.set(deep(structuredClone(c.augment)));
    closeEdited.set(true);
    name.set(uniqueName(`${runName(c.name)}-retry`, taken.get()));
    nameEdited.set(true);
    prefilled.set(true);
  });
  effect(() => {
    if (source.error && !untrack(() => prefilled.get())) {
      // page-level: loading the run to copy, before any dialog opens. A run that does not exist says so in the user's
      // words (stress s2-runs-7: the server's hint is written for agents).
      const e = source.error;
      if (e instanceof ApiError && e.code === 'run_not_found')
        toast.show({ tone: 'error', title: 'Run to copy not found', detail: 'It was deleted or the link is wrong. The form starts from the defaults.', key: FROM_TOAST });
      else toast.error(e, 'Could not load the run to copy', { key: FROM_TOAST });
      prefilled.set(true);
    }
  });
  // The toast is about this link: it goes with the form (another link, another page).
  effect(() => () => toast.clear(FROM_TOAST));

  // The first dataset until one is picked. One a draft or a link asked for that is gone is not swapped for another in
  // silence: none is picked, and the picker says which one went (by the name its runs recorded, where one has).
  effect(() => {
    const list = datasets.data;
    if (list?.length && !untrack(() => datasetId.get())) datasetId.set(list[0]!.id);
  });
  const gone = computed(() => {
    const id = datasetId.get();
    if (!id || !datasets.data || datasets.data.some((d) => d.id === id)) return null;
    const was = runs.data?.find((r) => r.config.dataset_id === id)?.dataset?.name;
    return `${was ?? 'The dataset asked for'} was removed: pick a dataset.`;
  });

  // The draft follows every edit, once the form holds what the link asked for; a start clears it (below).
  let started = false;
  // Gone (the user went elsewhere while Start was answered, or the link changed): the start then reports by its toast
  // and leaves the user where they went.
  let mounted = true;
  onCleanup(() => (mounted = false));
  // The draft as this form last wrote it: a start clears only that one, never a newer form's (a start answered after
  // the user left and opened New run again).
  let mine: string | null = null;
  effect(() => {
    const d: Draft = {
      server: hub.server,
      query,
      datasetId: datasetId.get(),
      arch: arch.get(),
      epochs: epochs.get(),
      batch: batch.get(),
      imgsz: imgsz.get(),
      imgszChosen: imgszChosen.get(),
      device: device.get(),
      optimizer: optimizer.get(),
      amp: amp.get(),
      e2e: e2e.get(),
      lr0: lr0.get(),
      seed: seed.get(),
      weights: weights.get(),
      name: name.get(),
      nameEdited: nameEdited.get(),
      augment: { ...augment.get(), hsv: [...augment.get().hsv] as Augment['hsv'] },
      closeEdited: closeEdited.get(),
      advancedOpen: advancedOpen.get(),
      augOpen: augOpen.get(),
      cloud: cloudPick.draft,
    };
    if (prefilled.get() && !started) mine = saveDraft(d);
  });

  const dataset = computed(() => datasets.data?.find((d) => d.id === datasetId.get()));

  // A fine-tune trains the weights' own architecture (docs/API.md, TrainConfig.weights), so the weights listed first are
  // the chosen size's and switching size drops another size's choice. Trained models of the other sizes follow, a
  // group per size, and published weights (`folo weights pull`, `source: published`) in a group of their own: picking
  // either sets the size to theirs. Fused weights (an export) load for inference only, so they are never a starting point.
  const startable = computed(() => (models.data ?? []).filter((m) => m.fused !== true));
  const weightOptions = computed(() => startable.get().filter((m) => !isPublished(m) && m.arch === arch.get()));
  const publishedOptions = computed(() => startable.get().filter(isPublished));
  // Each starting point's score with what it is on (lib/scoreSummary): grouped by dataset and split, best first.
  const scoreCtx = computed(() => scoreContextOf(runs, datasets));
  const weightChoices = computed(() => modelChoices(weightOptions.get(), scoreCtx.get()));
  const otherOptions = computed(() => startable.get().filter((m) => !isPublished(m) && m.arch !== arch.get()));
  const otherChoices = computed(() => otherSizeChoices(otherOptions.get(), arch.get(), scoreCtx.get()));
  const picked = computed(() => (weights.get() === 'none' ? undefined : startable.get().find((m) => m.id === weights.get())));
  /** Start from's hint: the picked weights' score with all of its context, and what published weights are. */
  const startHint = computed(() => startHintOf(picked.get(), scoreCtx.get(), arch.get(), weightOptions.get().length, otherOptions.get(), dataset.get()));
  /** Published weights' provenance: the server's description, then the score with what it is on. */
  const pickedAbout = computed(() => {
    const p = picked.get();
    return p && isPublished(p) ? [p.about, scoreMeta(p.metrics?.map50_95, scoreCtx.get()(p))].filter(Boolean).join(' · ') : null;
  });
  /** Published weights picked: the run's size is theirs, said under the size picker. */
  const pickedPublished = computed(() => {
    const p = picked.get();
    return p && isPublished(p) ? p : null;
  });
  effect(() => {
    const a = arch.get();
    const list = startable.get();
    if (!models.data) return;
    const w = untrack(() => weights.get());
    if (w === 'none') return;
    const m = list.find((x) => x.id === w);
    if (!m || m.arch !== a) weights.set('none');
  });
  /** Start from picked: weights of another size (trained or published) bring their size (the size picker follows). */
  function pickWeights(id: string) {
    const m = startable.get().find((x) => x.id === id);
    if (m && m.arch !== arch.get()) arch.set(m.arch);
  }
  const gpu = computed(() => system.data?.gpus[0]);

  // The suggested name is unique among existing runs (dataset-size, then -2, -3, …), so runs stay tellable apart; a
  // name typed over it that is taken gets a warning under the field (runnew/name).
  effect(() => {
    // The dataset's part gives way first, so a long dataset name keeps the size (and a few numbers) in the limit.
    const ds = clipName(runName(dataset.get()?.name ?? ''), MAX_RUN_NAME - 4) || 'run';
    const auto = uniqueName(`${ds}-${arch.get().replace('folo-', '')}`, taken.get());
    if (!untrack(() => nameEdited.get())) name.set(auto);
  });

  /** The picked dataset, where its images can be read (listed, folder there): nothing is asked of one that is not. */
  const readableId = computed(() => {
    const d = dataset.get();
    return d && !d.problem ? d.id : '';
  });
  // A few training images for the augmentation preview.
  const sampleImages = new Resource<DatasetImage[]>((s) => sampleImagesOf(readableId.get(), s));

  // The training images' sizes, to train a dataset of small images at its own size. Read only for a dataset small
  // enough to list in a few pages (the stats do not carry image sizes); otherwise null, and the size stays 640.
  const trainSizes = new Resource((s) => trainSizesOf(readableId.get(), s));
  const datasetImgsz = computed(() => {
    const t = trainSizes.data;
    return t && t.id === datasetId.get() ? defaultImgsz(t.sizes) : null;
  });
  // Until the current dataset's sizes are known (or when they cannot be read), an unchosen size is the 640 default:
  // never the size fitted to the dataset picked before.
  effect(() => {
    const d = datasetImgsz.get();
    if (!untrack(() => imgszChosen.get())) imgsz.set(String(d ?? 640));
  });

  const batchN = computed(() => Math.max(1, Math.round(Number(batch.get()) || 16)));
  const imgszN = computed(() => Number(imgsz.get()) || 640);
  const onCpu = computed(() => !onCloud.get() && (device.get() === 'cpu' || (device.get() === 'auto' && !gpu.get())));

  // Mixed precision (TrainConfig.amp): shown only when the server says what `auto` can use (`capabilities`, absent from
  // older servers), and off where a run would fail with `not_supported`: one plain sentence, the command that works
  // (ui/Remedy: the shell, one unwrapped copyable line), the server's own words kept verbatim behind "Driver report".
  const caps = computed(() => system.data?.capabilities);
  const ampBlock = computed(() => ampBlockOf(caps.get(), device.get(), onCloud.get(), system.data));
  const ampOn = computed(() => amp.get() && (onCloud.get() || (!!caps.get() && !ampBlock.get())));

  /** Epochs without mosaic as the trainer runs them: every epoch when mosaic is 0, else at most half the run. */
  const closeRun = computed(() => (augment.get().mosaic <= 0 ? epochs.get() : closeMosaicEpochs(augment.get().close_mosaic, epochs.get())));
  /** This server's device is the reference machine's, so the reference's narrower range applies. */
  const refMachine = computed(() => !onCloud.get() && (onCpu.get() ? system.data?.cpu.name === REF_CPU : gpu.get()?.name === REF_GPU));
  /** The local estimate: docs/BENCHMARKS.md's reference, while the server's loads or where it has none. */
  const est = computed(() =>
    estimate(arch.get(), dataset.get()?.splits.train ?? 0, epochs.get(), batchN.get(), imgszN.get(), onCpu.get(), ampOn.get(), {
      valImages: dataset.get()?.splits.val ?? 0,
      closeEpochs: closeRun.get(),
      sameMachine: refMachine.get(),
    }),
  );
  /** The trainer's schedule: optimizer steps an epoch, fewer than batches when gradients accumulate. */
  const sched = computed(() => {
    const e = epochs.get();
    return warmup(dataset.get()?.splits.train ?? 0, Number.isInteger(e) && e > 0 ? e : 1, batchN.get());
  });
  const stepsOpen = signal(false);
  // The figures wait for the server's devices and the dataset: until then "Estimating…", never a CPU guess or 0 steps.
  // A cloud run's figures wait for its GPU too.
  const estReady = computed(() => !!dataset.get() && (system.data !== undefined || system.error !== undefined) && (!onCloud.get() || !!cloudPick.offer));
  // The server reads an adapter's memory through Vulkan; through DX12 or Metal (and from servers before that) it sends 0:
  // then there is no meter and no out-of-memory guess. A cloud run's memory is its offer's.
  const gpuMem = computed(() => {
    const g = gpu.get();
    return onCloud.get() ? cloudPick.offer?.vram_bytes || null : g && g.memory_bytes > 0 ? g.memory_bytes : null;
  });
  /** Whether GPU memory applies: a GPU of this machine, or the cloud offer's. */
  const memShown = computed(() => (onCloud.get() ? !!cloudPick.offer : !onCpu.get() && !!gpu.get()));
  // The cheapest offer with the memory these settings need (the reference model's forecast), until one is picked.
  effect(() => {
    if (onCloud.get()) cloudPick.preselect(dataset.get() ? est.get().vramBytes : null);
  });

  // A retry of a run that ran out of GPU memory: what changed since, and an estimate that agrees with the failure.
  const oomRun = computed(() => {
    const r = source.data;
    return r && r.status === 'failed' && isOom(r.error) ? r : null;
  });
  const oomChanges = computed(() => oomChangesOf(oomRun.get(), batchN.get(), imgszN.get(), arch.get(), device.get()));
  const was = (key: string) => oomChanges.get().find((x) => x.key === key)?.was;
  // The estimate is rough; when it thought the failed settings fit, it is scaled up until they no longer do.
  // The run page forecasts its fixes with the same correction (lib/oom), so the retry it offers is shown here as fitting.
  const scale = computed(() => {
    const r = oomRun.get();
    return r ? oomCalibration(r.config, gpuMem.get()) : 1;
  });

  // Start answered `501 not_supported` for mixed precision: the server cannot train that way after all (restarted
  // without tensor cores since it said it could). Its message stays at the Mixed precision field until the switch or
  // the device changes; the capabilities are read again, so the field shows what works now (off, with the remedy).
  // While the switch is still on, Start waits (a field in error gates the form).
  const ampRefused = signal<{ device: string; message: string } | null>(null);
  const ampErr = computed(() => {
    const r = ampRefused.get();
    return r && r.device === device.get() ? r : null;
  });

  // Start answered `404 device_not_found` (`501 not_implemented` from servers before it), or the device list lacks the
  // GPU the form holds: said at the Device field in
  // one line built from the devices this machine lists now, until another device is picked. The server's own words
  // (they name its backends) fold into the field's technical details.
  const deviceError = signal<{ device: string; gone: boolean; detail: string } | null>(null);
  const deviceErr = computed(() => {
    const e = deviceError.get();
    if (e && e.device === device.get()) return e;
    // A GPU kept from the draft or a copied run that this server does not list is caught before Start, not by it.
    return onCloud.get() ? null : missingDevice(device.get(), system.data);
  });
  const deviceMsg = computed(() => deviceMessage(deviceErr.get(), system));

  const errors = computed(() =>
    formErrors(
      name.get(),
      epochs.get(),
      lr0.get(),
      seed.get(),
      augment.get(),
      dataset.get(),
      gone.get(),
      weights.get(),
      picked.get(),
      arch.get(),
      deviceErr.get(),
      onCloud.get(),
      device.get(),
      ampErr.get(),
      ampOn.get(),
      cloudOn.get(),
      cloudPick,
    ),
  );
  const valid = computed(() => prefilled.get() && Object.values(errors.get()).every((e) => e === null));
  /**
   * Settings no estimate can stand for: no dataset, or one with nothing to train on, or an epoch count (or Close mosaic)
   * out of range. The Summary shows dashes (never "0 optimizer steps", "under 1 min" or "24–55 h" for 99,999 epochs),
   * no memory verdict, and the server is not asked.
   */
  const noFigures = computed(() => !!errors.get().dataset || !isInt(epochs.get(), 1, 10000) || !!errors.get().close);
  /** The first problem, named by its field (shown beside a disabled Start, as a way to that field). */
  const why = computed(() => firstError(errors.get()));
  const showWhy = computed(() => !valid.get() && !!why.get() && prefilled.get());
  // The trainer's own short-run warnings (too few steps from scratch, mostly warmup): not errors, a warning with the fix.
  const short = computed(() => (errors.get().epochs ? null : shortRun(dataset.get()?.splits.train ?? 0, epochs.get(), batchN.get(), weights.get() === 'none')));
  const selArch = computed(() => arches.find((a) => a.arch === arch.get()));

  const config = computed<TrainConfig>(() =>
    trainConfig(
      name.get(),
      datasetId.get(),
      arch.get(),
      epochs.get(),
      batchN.get(),
      imgszN.get(),
      onCloud.get(),
      device.get(),
      optimizer.get(),
      lr0.get(),
      weights.get(),
      augment.get(),
      seed.get(),
      caps.get(),
      ampOn.get(),
      e2e.get(),
      cloudPick,
    ),
  );

  const canEstimate = computed(() => hasFeature(system.data, 'run_estimate'));
  const estKey = computed(() => JSON.stringify(config.get()));
  const { served, estimating } = serverEstimate(config, estKey, () => canEstimate.get() && !noFigures.get() && (!onCloud.get() || !!cloudPick.compute));
  const closeEff = computed(() => closeMosaicShown(errors.get(), served.get(), estKey.get(), augment.get(), epochs.get()));
  /**
   * Whether the figures shown answer earlier settings: the last answer stays up while the request for these is out, so
   * the form never empties as it is edited, but it is not these settings' figure (DESIGN_DIRECTION §2.8, "a figure is
   * either current or visibly marked as not"): Est. time and Est. GPU memory dim to tertiary, in the Summary and the
   * dock, and the basis line reads "Updating…" until the answer for these settings is in.
   */
  const estStale = computed(() => {
    const s = served.get();
    return !!s && s.key !== estKey.get();
  });
  const shown = computed(() => shownEstimate(served.get(), est.get(), onCloud.get(), refMachine.get(), onCpu.get(), canEstimate.get(), estimating.get()));
  // Measured peaks need no scaling: only the reference model is corrected by a failed run.
  const vram = computed(() => shown.get().vram * (shown.get().vramMeasured ? 1 : scale.get()));
  const vramPct = computed(() => {
    const m = gpuMem.get();
    return m ? vram.get() / m : 0;
  });
  /**
   * One condition, one severity, on the meter and the line under it (the figure stays in ink): over the GPU's memory is
   * `danger` (likely won't fit), within 10% of it `warning` (tight; lib/oom's headroom), else neutral.
   */
  const vramLevel = computed(() =>
    !memShown.get() || !gpuMem.get() || noFigures.get() ? null : vramPct.get() > 1 ? ('danger' as const) : vramPct.get() > 0.9 ? ('warning' as const) : null,
  );
  const noFit = computed(() => vramLevel.get() === 'danger');
  /**
   * Whether these settings fit the GPU: `fits`, `no` (forecast over its memory) or `unknown` (the adapter does not report
   * its memory, so nothing can be checked); null on the CPU, where GPU memory does not apply.
   */
  const vramFit = computed(() => (!memShown.get() ? null : !gpuMem.get() ? ('unknown' as const) : noFit.get() ? ('no' as const) : ('fits' as const)));
  // The time limit follows the estimate's high end until it is set by hand.
  effect(() => cloudPick.follow(estReady.get() ? shown.get().hi : null));
  const cloudCost = computed(() => cloudCostOf(onCloud.get(), cloudPick, served.get(), shown.get()));

  // Settings forecast not to fit: the one primary becomes the change that fits ("Use batch 8"), chosen by lib/oom's
  // `fitFix` from the same forecast as the figure above (the server's estimate, else the reference model) with 10% of
  // the memory to spare; Start steps back to a secondary with the risk said beside it. Asked once the estimate for
  // these settings is in, so the two never disagree.
  const fitted = signal<{ key: string; fix: OomFix | null } | null>(null);
  effect(() => {
    const key = estKey.get();
    if (!noFit.get() || estimating.get() || untrack(() => fitted.get()?.key) === key) return;
    const ctrl = new AbortController();
    const c = untrack(() => config.get());
    const forecaster = vramForecaster(
      hub.api,
      untrack(() => system.data),
      ctrl.signal,
    );
    fitFix(
      c,
      gpuMem.get(),
      forecaster,
      untrack(() => scale.get()),
    ).then(
      (fix) => !ctrl.signal.aborted && fitted.set({ key, fix }),
      () => {},
    );
    return () => ctrl.abort();
  });
  const fix = computed(() => {
    const f = fitted.get();
    return noFit.get() && f?.key === estKey.get() ? f.fix : null;
  });
  const fixLoading = computed(() => noFit.get() && fitted.get()?.key !== estKey.get());
  /** Applies the change, as a retry's query would have (lib/oom `OomFix.query`). */
  function applyFix(f: OomFix) {
    for (const [k, v] of Object.entries(f.query)) {
      if (k === 'batch') batch.set(v);
      else if (k === 'imgsz') (imgsz.set(v), imgszChosen.set(true));
      else if (k === 'arch') arch.set(v as Arch);
      else if (k === 'device') device.set(v);
    }
  }

  // The same run from a terminal: the server's command where it builds one (only it knows its files), else Studio's,
  // from the CLI's own mapping ($lib/trainCli). A fine-tune names the weights file by path; where the server does
  // not report it, a placeholder and a note say what to fill in.
  const ownCli = computed(() =>
    trainCommand(
      config.get(),
      dataset.get()?.path ?? datasetId.get(),
      models.data?.find((m) => m.id === weights.get()),
    ),
  );
  const servedCli = computed(() => {
    const s = served.get();
    return s?.key === estKey.get() ? serverCommand(s.est) : null;
  });
  const cliParts = computed(() => {
    const s = servedCli.get();
    return s ? s.split(/ (?=--)/) : ownCli.get().parts;
  });
  const cli = computed(() => servedCli.get() ?? ownCli.get().line);
  const cliPlaceholder = computed(() => (servedCli.get() ? null : ownCli.get().placeholder));

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    if (!valid.get() || submitting.get()) return;
    // The Summary sheet closes first: the start's result is the page's (a toast, or the Device field's message).
    sumOpen.set(false);
    submitting.set(true);
    const before = ahead.get();
    // Where the user asked to be when Start was pressed: the URL's route, not the one on display (App keeps that on
    // the form while the next screen's chunk loads, so the form stays mounted and `route` still says run-new).
    const here = router.location;
    const stayed = () => {
      const now = router.location;
      return mounted && now.path === here.path && now.query.toString() === here.query.toString();
    };
    try {
      const run = await hub.api.createRun(config.get());
      started = true;
      if (mine !== null) clearOwnDraft(mine);
      const where = run.cloud ? `, at ${cloudPick.provider} on ${run.cloud.offer.gpu}` : '';
      // page-level: the sheet is closed (above).
      const t = startedToast(run, before);
      // Keyed by the run's place in the queue: a later "Removed … from the queue" takes its place (run/queueActions).
      toast.success(t.title, `${t.place ? `${t.place} · ` : ''}${run.epochs} epochs of ${run.config.arch} on ${dataset.get()?.name}${where}.`, undefined, { key: queueToastKey(run.id) });
      if (stayed()) router.go(`/runs/${run.id}`);
    } catch (err) {
      if (err instanceof ApiError && (err.code === 'device_not_found' || (err.code === 'not_implemented' && device.get().startsWith('gpu:')))) {
        deviceError.set({
          device: device.get(),
          gone: err.code === 'device_not_found',
          detail: [err.message, err.hint]
            .filter((x): x is string => !!x)
            .map((x) => sentence(x, true))
            .join(' '),
        });
        // The adapters may have changed (an eGPU unplugged): list them again, and show the field with its message.
        void system.refresh();
        void jump('device');
        // page-level: the sheet closed when the form was sent.
      } else if (err instanceof ApiError && err.code === 'not_supported' && config.get().amp) {
        // At the field, not a toast: the message, then what the server can do now (the capabilities read again).
        ampRefused.set({ device: device.get(), message: err.message });
        void system.refresh();
        void jump('amp');
      } else {
        // A cloud run's refusal at the control it is about (the GPU, Run on, Stop after), else a toast.
        const field = cloudPick.refuse(err);
        // page-level: the sheet closed when the form was sent.
        if (field) void jump(field);
        else toast.error(err, 'Could not start the run');
      }
    } finally {
      submitting.set(false);
    }
  }

  // ---------- review ----------
  const deviceLabel = computed(() => deviceLabelOf(device.get(), gpu.get(), system));
  const review = computed(() =>
    reviewRows(
      dataset.get(),
      errors.get(),
      arch.get(),
      epochs.get(),
      batchN.get(),
      imgszN.get(),
      onCloud.get(),
      cloudPick,
      deviceLabel.get(),
      deviceMsg.get(),
      caps.get(),
      ampOn.get(),
      weights.get(),
      picked.get(),
      optimizer.get(),
      e2e.get(),
    ),
  );

  async function jump(key: Jump) {
    sumOpen.set(false);
    if (key === 'opt' || key === 'lr0' || key === 'seed' || key === 'head') advancedOpen.set(true);
    if (key === 'close') augOpen.set(true);
    await tick();
    // A closing dialog hands focus back to its opener; move on after that.
    requestAnimationFrame(() => landOn(key));
  }

  // The Svelte original's snippets, as local components.
  const Review = (p: { withName: boolean }) => (
    <ReviewList withName={p.withName} name={runName(name.get())} errors={errors.get()} review={review.get()} jump={jump} />
  );

  const Figures = (p: { where: string }) => (
    <EstimateFigures
      where={p.where}
      blocked={noFigures.get()}
      estReady={estReady.get()}
      sched={sched.get()}
      batchN={batchN.get()}
      est={est.get()}
      bind:stepsOpen={stepsOpen}
      shown={shown.get()}
      estStale={estStale.get()}
      estimating={estimating.get()}
      memShown={memShown.get()}
      vram={vram.get()}
      onCloud={onCloud.get()}
      cloudPick={cloudPick}
      cloudCost={cloudCost.get()}
      gpuMem={gpuMem.get()}
      scale={scale.get()}
      onCpu={onCpu.get()}
      vramLevel={vramLevel.get()}
      vramPct={vramPct.get()}
      noFit={noFit.get()}
      fix={fix.get()}
      fixLoading={fixLoading.get()}
      short={short.get()}
      bind:shortWhy={shortWhy}
      bind:epochs={epochs}
    />
  );

  /** Why Start is off, as a way to that field (runnew/WhyLine). */
  const Why = (p: { id: string; cls?: string }) => <WhyLine id={p.id} cls={p.cls} why={showWhy.get() ? why.get() : null} jump={jump} />;

  return (
    <Page title="New training run">
      {datasets.error && !datasets.data ? (
        <ErrorState error={datasets.error} retry={() => datasets.refresh()} title="Could not load datasets" />
      ) : datasets.data && datasets.data.length === 0 ? (
        <Panel>
          <EmptyState
            icon={Images}
            title="Add a dataset first"
            description="Runs train on a registered dataset: add a folder or generate a synthetic one."
            actions={() => (
              // Straight to the Add dataset dialog, as Overview's first step (Datasets reads `?add=`).
              <>
                <Button variant="primary" icon={Plus} href={href('/datasets?add=folder')}>
                  Add dataset
                </Button>
                <Button href={href('/datasets?add=synthetic')}>Generate sample set</Button>
              </>
            )}
          />
        </Panel>
      ) : (
        <form class="layout" ref={formEl} onsubmit={submit} novalidate>
          {oomRun.get() ? (
            <OomNote
              oomRun={oomRun.get()!}
              oomChanges={oomChanges.get()}
              vramFit={vramFit.get()}
              vram={vram.get()}
              gpuMem={gpuMem.get()}
              vramLevel={vramLevel.get()}
            />
          ) : null}
          <div class="form">
            <DatasetSection datasets={datasets} bind:datasetId={datasetId} dataset={dataset.get()} errors={errors.get()} />

            <ModelSection
              bind:arch={arch}
              bind:weights={weights}
              tp={tp}
              oomRun={oomRun.get()}
              was={was}
              pickedPublished={pickedPublished.get()}
              selArch={selArch.get()}
              startHint={startHint.get()}
              pickedAbout={pickedAbout.get()}
              errors={errors.get()}
              weightChoices={weightChoices.get()}
              otherChoices={otherChoices.get()}
              publishedOptions={publishedOptions.get()}
              scoreCtx={scoreCtx.get()}
              pickWeights={pickWeights}
            />

            <TrainingSection
              bind:epochs={epochs}
              bind:batch={batch}
              bind:imgsz={imgsz}
              bind:imgszChosen={imgszChosen}
              bind:device={device}
              bind:optimizer={optimizer}
              bind:amp={amp}
              bind:ampRefused={ampRefused}
              bind:e2e={e2e}
              bind:lr0={lr0}
              bind:seed={seed}
              bind:advancedOpen={advancedOpen}
              arch={arch.get()}
              errors={errors.get()}
              was={was}
              oomRun={oomRun.get()}
              dataset={dataset.get()}
              datasetImgsz={datasetImgsz.get()}
              IMGSZ={IMGSZ}
              cloudOn={cloudOn.get()}
              onCloud={onCloud.get()}
              cloudPick={cloudPick}
              system={system}
              gpu={gpu.get()}
              deviceMsg={deviceMsg.get()}
              deviceErr={deviceErr.get()}
              caps={caps.get()}
              ampBlock={ampBlock.get()}
              ampErr={ampErr.get()}
              ampOn={ampOn.get()}
            />

            <AugmentSection
              bind:augment={augment}
              bind:augOpen={augOpen}
              bind:closeEdited={closeEdited}
              defaults={defaults}
              errors={errors.get()}
              closeEff={closeEff.get()}
              epochs={epochs.get()}
              dataset={dataset.get()}
              sampleImages={sampleImages}
            />
          </div>

          <aside class="summary">
            <Panel title="Summary">
              {/* Two groups: what the run is (name, settings) and what it costs (estimates, Start). A rail stacks them; a
                  full-width summary (below 960 px) sets them side by side. */}
              <div class="sum">
                <div class="sum-col">
                  <Field label="Run name" for="name" error={errors.get().name} hint={nameHint(name.get(), taken.get(), nameEdited.get()) ?? undefined}>
                    <Input
                      id="name"
                      bind:value={name}
                      oninput={() => nameEdited.set(true)}
                      onblur={() => {
                        const n = runName(name.get());
                        if (n && n !== name.get()) name.set(n);
                      }}
                      invalid={!!errors.get().name}
                      spellcheck="false"
                      autocomplete="off"
                    />
                  </Field>
                  <Review withName={false} />
                </div>
                <div class="sum-col">
                  <Figures where="rail" />
                  {ahead.get() ? <p class="ahead">{aheadLine(ahead.get()!)}</p> : null}
                  <div class="start" ref={startEl}>
                    {noFit.get() && (fix.get() || fixLoading.get()) ? (
                      <Button variant="primary" size="lg" block loading={fixLoading.get()} onclick={() => fix.get() && applyFix(fix.get()!)}>
                        {fix.get() ? `${useLabel(fix.get()!)}${fix.get()!.vram !== undefined ? ` · ${roughBytes(fix.get()!.vram!)}` : ''}` : 'Finding settings that fit'}
                      </Button>
                    ) : null}
                    <span class="start-tip" use:tooltip={{ text: startLabel.get(), keys: START_KEYS }}>
                      <Button
                        type="submit"
                        variant={noFit.get() ? 'secondary' : 'primary'}
                        size="lg"
                        icon={Play}
                        block
                        loading={submitting.get()}
                        disabled={!valid.get()}
                        aria-describedby={[showWhy.get() ? 'start-why' : '', noFit.get() ? 'rail-nofit' : ''].filter(Boolean).join(' ') || undefined}
                      >
                        {startLabel.get()}
                      </Button>
                    </span>
                    <Why id="start-why" />
                  </div>
                  <CliCommand cliParts={cliParts.get()} cli={cli.get()} cliPlaceholder={cliPlaceholder.get()} />
                </div>
              </div>
            </Panel>
          </aside>

          <StartDock
            startVisible={startVisible.get()}
            showWhy={showWhy.get()}
            why={why.get()}
            whyLine={(id, cls) => <Why id={id} cls={cls} />}
            bind:sumOpen={sumOpen}
            shown={shown.get()}
            estReady={estReady.get()}
            estStale={estStale.get()}
            cloudCost={cloudCost.get()}
            memShown={memShown.get()}
            noFit={noFit.get()}
            gpuMem={gpuMem.get()}
            vram={vram.get()}
            short={short.get()}
            dataset={dataset.get()}
            arch={arch.get()}
            fix={fix.get()}
            fixLoading={fixLoading.get()}
            applyFix={applyFix}
            submitting={submitting.get()}
            valid={valid.get()}
            startLabel={startLabel.get()}
          />

          {/* Inside the form, so its Start button submits it. */}
          <Dialog
            bind:open={sumOpen}
            title="Summary"
            size="sm"
            footer={() => (
              <>
                <Button variant="ghost" onclick={() => sumOpen.set(false)}>
                  Keep editing
                </Button>
                {noFit.get() && (fix.get() || fixLoading.get()) ? (
                  <>
                    <Button
                      type="submit"
                      icon={Play}
                      loading={submitting.get()}
                      disabled={!valid.get()}
                      aria-describedby={[showWhy.get() ? 'sheet-why' : '', 'sheet-nofit'].filter(Boolean).join(' ')}
                    >
                      {startLabel.get()}
                    </Button>
                    <Button variant="primary" loading={fixLoading.get()} onclick={() => fix.get() && applyFix(fix.get()!)}>
                      {fix.get() ? useLabel(fix.get()!) : 'Finding a fit'}
                    </Button>
                  </>
                ) : (
                  <Button
                    type="submit"
                    variant="primary"
                    icon={Play}
                    loading={submitting.get()}
                    disabled={!valid.get()}
                    aria-describedby={showWhy.get() ? 'sheet-why' : undefined}
                  >
                    {startLabel.get()}
                  </Button>
                )}
              </>
            )}
          >
            <div class="sum">
              <Review withName />
              <Figures where="sheet" />
              {ahead.get() ? <p class="ahead">{aheadLine(ahead.get()!)}</p> : null}
              <Why id="sheet-why" />
            </div>
          </Dialog>
        </form>
      )}
    </Page>
  );
}

css`
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 340px;
    gap: var(--space-4);
    align-items: start;
  }
  .form {
    display: grid;
    gap: var(--space-4);
    min-width: 0;
  }
  .summary {
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
  }
  .sum,
  .sum-col {
    display: grid;
    align-content: start;
    gap: var(--space-4);
    min-width: 0;
  }
  .start-tip {
    display: block;
  }
  /* What the run waits for: a plain line in the Summary's secondary ink, not a warning (waiting is normal). */
  .ahead {
    max-inline-size: var(--measure);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .layout :global([data-jumped]) {
    outline: 2px solid var(--focus);
    outline-offset: 4px;
    border-radius: var(--radius-md);
  }
  @media (max-width: 1279px) {
    /* Laptops keep the sticky rail beside the form, 40 px slimmer. */
    .layout {
      grid-template-columns: minmax(0, 1fr) 300px;
    }
  }
  @media (max-width: 1023px) {
    .layout {
      padding-bottom: 64px;
    }
    .layout {
      grid-template-columns: minmax(0, 1fr);
    }
    .summary {
      position: static;
    }
    /* Full width under the form: the settings on the left, the estimates and Start on the right. */
    .summary .sum {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      column-gap: var(--space-6);
    }
  }
  @media (max-width: 767px) {
    .summary .sum {
      grid-template-columns: minmax(0, 1fr);
    }
  }
`;
