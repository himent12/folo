// Reactive state, validation, server/local estimation, OOM fit-fix, draft persistence, and submission for New run (`../RunNew.tsx`).
// Call `runNewState()` inside the component body so its resources, effects, and cleanup follow the component's life.
import { computed, deep, effect, onCleanup, signal, tick, untrack } from 'fw';
import { followedDatasets, listedName } from '$lib/api/datasetReads';
import { ApiError, hub } from '$lib/api/index';
import { Resource } from '$lib/api/resource';
import { runsChanged } from '$lib/api/runs';
import type { Arch, Augment, DatasetImage, Model, Optimizer, TrainConfig } from '$lib/api/types';
import { archFromCli, archNameSuffix, archScale, archWithFamily, isFolo2Arch, supportsFolo2 } from '$lib/arch';
import { hasCloud } from '$lib/cloud';
import { sentence } from '$lib/format';
import { fitFix, isOom, oomCalibration, vramForecaster, type OomFix } from '$lib/oom';
import { REF_CPU, REF_GPU } from '$lib/reference';
import { router } from '$lib/router';
import { modelChoices, scoreMeta } from '$lib/scoreSummary';
import { watched } from '$lib/shell/runList';
import { hasFeature } from '$lib/throughput';
import { serverCommand } from '$lib/trainCli';
import { toast } from '$lib/ui/toast';
import { queueToastKey } from '../run/queueActions';
import { ampBlockOf } from './amp';
import { cliOf } from './CliCommand';
import { CloudPick } from './cloudPick';
import { trainConfig } from './config';
import { canonicalDevice, deviceLabelOf, deviceMessage, missingDevice } from './device';
import { arches, autoCloseMosaic, closeMosaicEpochs, defaultImgsz, estimate, shortRun, warmup } from './estimate';
import { copiedSeed, errorCount, firstError, formErrors, isInt, landOn, type Jump } from './jump';
import { clipName, MAX_RUN_NAME, runName, uniqueName } from './name';
import { initialDraftState, useDraftPersistence } from './newDraft';
import { DEFAULT_AUGMENT, parseInitialQuery, useRunPrefill } from './newPrefill';
import { aheadOf, startedToast } from './queue';
import { oomChangesOf } from './retry';
import { estimateKey, serverEstimate, withServerCliName } from './serverEstimate';
import { startKeys } from './startKeys';
import { cloudCostOf, closeMosaicShown, reviewRows, shownEstimate } from './summary';
import { sampleImagesOf, trainSizesOf } from './trainImages';
import { isPublished, otherSizeChoices, scoreContextOf, startHintOf } from './weights';

export function runNewState() {
  const q = untrack(() => router.route.query);
  const query = q.toString();
  const fromId = q.get('from');
  const { own, draft } = initialDraftState(query, hub.server, fromId);
  const init = parseInitialQuery(q, own, draft);

  const datasetId = signal<string>(init.datasetId);
  // The dock repeats the summary's Start button for narrow screens; it steps aside while that button is on screen.
  const startEl = signal<HTMLElement>();
  const formEl = signal<HTMLFormElement>();
  startKeys(() => formEl.get(), { onCancel: () => router.go('/runs') });
  const startVisible = signal(false);
  effect(() => {
    const el = startEl.get();
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const io = new IntersectionObserver(([e]) => startVisible.set(!!e?.isIntersecting));
    io.observe(el);
    return () => io.disconnect();
  });

  const arch = signal<Arch>(init.arch);
  /** Number inputs bind `null` when cleared (the form's errors say so). */
  const epochs = signal<number>(draft ? (draft.epochs as number) : 100);
  const batch = signal(draft?.batch ?? '16');
  const imgsz = signal(init.imgsz);
  const imgszChosen = signal(init.imgszChosen);
  const device = signal<string>(canonicalDevice(draft?.device ?? 'auto'));
  const optimizer = signal<Optimizer>(draft?.optimizer ?? 'auto');
  /** Mixed precision as asked for; what the run gets is `ampOn` (off where the server cannot train that way). */
  const amp = signal(draft?.amp ?? false);
  /** The end-to-end head (TrainConfig.head `o2o`, docs/MODEL.md §6.6): predicts without NMS. */
  const e2e = signal(draft?.e2e ?? false);
  const lr0 = signal(draft ? (draft.lr0 as number) : 0.01);
  const seed = signal(draft?.seed ?? '0'); // as typed (lib/seed)
  const seedCopy = copiedSeed(seed, draft?.seedOver ?? false);
  const weights = signal<string>(init.weights);
  const name = signal(draft?.name ?? '');
  const nameEdited = signal(draft?.nameEdited ?? false);
  // Edited in place by the controls (a `deep` object, as Svelte's `$state`), replaced whole by a prefill or Defaults.
  const augment = signal<Augment>(deep(structuredClone({ ...DEFAULT_AUGMENT, ...draft?.augment })));
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

  // Each read for its stats, kept current; one whose folder is missing is marked, and Start waits (lib/api/datasetReads).
  const datasets = followedDatasets();
  const system = new Resource((s) => hub.api.system(s));
  const folo2Supported = computed(() => supportsFolo2(system.data));
  // Reactively step back any `folo2-*` arch (initial link, draft, or async ?from=/?weights= prefill) on servers without `folo2_arch`.
  effect(() => {
    const cur = arch.get();
    if (system.data && !folo2Supported.get() && isFolo2Arch(cur)) {
      arch.set(archWithFamily(cur, 'folo'));
    }
  });
  // Run on (servers with the `cloud` feature): this machine or a cloud connection, its GPU and the time limit.
  const cloudPick = new CloudPick(() => system.data);
  if (draft) cloudPick.restore(draft.cloud);
  const cloudOn = computed(() => hasCloud(system.data));
  /** A connection is chosen: the run trains on its GPU, not this machine's. */
  const onCloud = computed(() => cloudOn.get() && cloudPick.cloud);
  const models = new Resource((s) => hub.api.models(s));
  const runs = new Resource((s) => hub.api.runs(s), { startFrom: () => watched.source });
  // While another run holds this machine, a new one waits in the queue: the Summary says so and Start reads "Queue run".
  // The shell's list (polled every 2 s) is the current one; this page's own stands in until it arrives.
  const ahead = computed(() => aheadOf(watched.runs ?? runs.data, onCloud.get()));
  const startLabel = computed(() => (ahead.get() ? 'Queue run' : 'Start training'));
  /** Every run's name, so a new one is told apart (runnew/name). */
  const taken = computed(() => new Set((runs.data ?? []).map((r) => r.name)));
  // This server's measured speed (GET /api/system/throughput), where it has one: measured latency in the size table.
  const tp = new Resource((s) => (hasFeature(system.data, 'throughput') ? hub.api.throughput(s).catch(() => null) : Promise.resolve(null)));

  // Fused weights (an export) load for inference only, so they are never a starting point.
  // Real servers list models with wire `folo-*` arch; `resolvedModelArchs` overlays canonical `folo2-*` discovered via `model(id)` / `runCli`.
  const resolvedModelArchs = signal<Record<string, Arch>>({});
  const resolvingModelId = signal<string | null>(null);
  const checkedModelsTick = signal(0);
  const resolvingWeights = computed(() => {
    const id = resolvingModelId.get();
    return id !== null && weights.get() === id;
  });
  const startable = computed(() => {
    const overrides = resolvedModelArchs.get();
    return (models.data ?? [])
      .filter((m) => m.fused !== true)
      .map((m) => (overrides[m.id] && overrides[m.id] !== m.arch ? { ...m, arch: overrides[m.id]! } : m));
  });

  const { source, prefilled, pendingWeights } = useRunPrefill({
    q,
    fromId,
    own,
    queryWeights: init.queryWeights,
    runs,
    models,
    startable,
    taken,
    datasetId,
    arch,
    epochs,
    batch,
    imgsz,
    imgszChosen,
    device,
    optimizer,
    amp,
    e2e,
    lr0,
    seedCopy,
    weights,
    cloudPick,
    augment,
    closeEdited,
    name,
    nameEdited,
  });

  // The first dataset until one is picked. One a draft or a link asked for that is gone is not swapped for another in
  // silence: none is picked, and the picker says which one went (by the name its runs recorded, where one has).
  effect(() => {
    const list = datasets.data;
    if (list?.length && !untrack(() => datasetId.get())) datasetId.set(list[0]!.id);
  });
  const gone = computed(() => {
    const id = datasetId.get();
    if (!id || !datasets.data || datasets.data.some((d) => d.id === id)) return null;
    const was = listedName(id) ?? runs.data?.find((r) => r.config.dataset_id === id)?.dataset?.name;
    return `${was ?? 'The dataset asked for'} was removed: pick a dataset.`;
  });

  // Gone (the user went elsewhere while Start was answered, or the link changed): the start then reports by its toast
  // and leaves the user where they went.
  let mounted = true;
  onCleanup(() => (mounted = false));
  const draftStore = useDraftPersistence({
    ready: () => prefilled.get() && !pendingWeights.get(),
    build: () => ({
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
      seedOver: seedCopy.over(),
      weights: weights.get(),
      name: name.get(),
      nameEdited: nameEdited.get(),
      augment: { ...augment.get(), hsv: [...augment.get().hsv] as Augment['hsv'] },
      closeEdited: closeEdited.get(),
      advancedOpen: advancedOpen.get(),
      augOpen: augOpen.get(),
      cloud: cloudPick.draft,
    }),
  });

  const dataset = computed(() => datasets.data?.find((d) => d.id === datasetId.get()));

  // A fine-tune trains the weights' own architecture (docs/API.md, TrainConfig.weights), so the weights listed first are
  // the chosen size's and switching size drops another size's choice. Trained models of the other sizes follow, a
  // group per size, and published weights (`folo weights pull`, `source: published`) in a group of their own: picking
  // either sets the size to theirs.
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
  const checkedModels = new Set<string>();
  const inFlightModels = new Set<string>();
  /** Tracks a pending `pickWeights(id)` awaiting canonical arch resolution, plus the arch active when picked. */
  const pendingPick = signal<{ id: string; startArch: Arch } | null>(null);
  let resolveSeq = 0;
  async function resolveModelArch(m: Model, applyResolvedArch = false) {
    if (hub.api.mock || isFolo2Arch(m.arch) || checkedModels.has(m.id)) return;
    if (applyResolvedArch) {
      pendingPick.set({ id: m.id, startArch: arch.peek() });
      resolvingModelId.set(m.id);
    }
    if (inFlightModels.has(m.id)) return;
    inFlightModels.add(m.id);
    const seq = ++resolveSeq;
    resolvingModelId.set(m.id);
    try {
      const detail = await hub.api.model(m.id).catch(() => null);
      let resolved: Arch = detail && isFolo2Arch(detail.arch) ? detail.arch : m.arch;
      if (!isFolo2Arch(resolved) && m.run_id) {
        const cli = await hub.api.runCli(m.run_id).catch(() => null);
        const cliArch = archFromCli(cli);
        if (cliArch && isFolo2Arch(cliArch)) resolved = cliArch;
      }
      checkedModels.add(m.id);
      if (mounted) {
        if (resolved !== m.arch) {
          resolvedModelArchs.set({ ...resolvedModelArchs.peek(), [m.id]: resolved });
        }
        const pp = pendingPick.peek();
        const stillSamePick = pp?.id === m.id && arch.peek() === pp.startArch;
        if (pp?.id === m.id) pendingPick.set(null);
        const draftMatch = resolved !== m.arch && archScale(arch.peek()) === archScale(resolved);
        if ((stillSamePick || draftMatch) && weights.peek() === m.id && arch.peek() !== resolved) {
          arch.set(resolved);
        }
        checkedModelsTick.set((t) => t + 1);
      }
    } finally {
      inFlightModels.delete(m.id);
      if (mounted && (seq === resolveSeq || resolvingModelId.peek() === m.id)) resolvingModelId.set(null);
    }
  }
  effect(() => {
    const a = arch.get();
    const list = startable.get();
    void checkedModelsTick.get();
    if (!models.data || pendingWeights.get() !== null) return;
    const w = untrack(() => weights.get());
    if (w === 'none') return;
    const m = list.find((x) => x.id === w);
    if (!m) {
      weights.set('none');
      return;
    }
    if (m.arch !== a) {
      const pp = untrack(() => pendingPick.get());
      // If the user explicitly changed `arch` while a checkpoint lookup was pending, honor the newer arch choice.
      if (pp?.id === m.id && a !== pp.startArch) {
        pendingPick.set(null);
        weights.set('none');
        return;
      }
      // On a real server, a picked or draft-restored checkpoint listed as wire `folo-*` may actually be `folo2-*`:
      // defer clearing `weights` until its canonical arch lookup finishes once.
      if (
        !hub.api.mock &&
        !isFolo2Arch(m.arch) &&
        !checkedModels.has(m.id) &&
        ((pp?.id === m.id && inFlightModels.has(m.id)) || (isFolo2Arch(a) && archScale(a) === archScale(m.arch)))
      ) {
        untrack(() => void resolveModelArch(m));
        return;
      }
      weights.set('none');
    }
  });
  effect(() => {
    const p = picked.get();
    if (p && !hub.api.mock && !isFolo2Arch(p.arch) && !checkedModels.has(p.id)) {
      void resolveModelArch(p);
    }
  });
  /** Start from picked: weights of another size (trained or published) bring their size (the size picker follows). */
  function pickWeights(id: string) {
    const m = startable.get().find((x) => x.id === id);
    if (!m) {
      pendingPick.set(null);
      return;
    }
    // Wait for canonical arch resolution on unchecked real-server wire `folo-*` checkpoints before switching family/name.
    if (!hub.api.mock && !isFolo2Arch(m.arch) && !checkedModels.has(m.id)) {
      void resolveModelArch(m, true);
      return;
    }
    pendingPick.set(null);
    if (m.arch !== arch.get()) arch.set(m.arch);
  }
  const gpu = computed(() => system.data?.gpus[0]);

  // The suggested name is unique among existing runs (dataset-size, then -2, -3, …), so runs stay tellable apart; a
  // name typed over it that is taken gets a warning under the field (runnew/name).
  effect(() => {
    // The dataset's part gives way first, so a long dataset name keeps the size (and a few numbers) in the limit.
    const suffix = archNameSuffix(arch.get());
    const ds = clipName(runName(dataset.get()?.name ?? ''), Math.max(8, MAX_RUN_NAME - (suffix.length + 3))) || 'run';
    const auto = uniqueName(`${ds}-${suffix}`, taken.get());
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
  // The estimate's same refusal, before Start: only while the switch is on, for this machine, and until one is answered.
  const ampEstimated = signal<{ device: string; message: string } | null>(null);
  const ampErr = computed(() => {
    const r = ampRefused.get();
    if (r && r.device === device.get()) return r;
    const e = ampEstimated.get();
    return e && e.device === device.get() && ampOn.get() && !onCloud.get() ? e : null;
  });

  // Start answered `404 device_not_found` (`501 not_implemented` from servers before it), or the device list lacks the
  // GPU the form holds (from a draft or a copied run): said at the Device field in one line built from the devices this
  // machine lists now, until another device is picked. The server's words fold into the field's technical details.
  const deviceError = signal<{ device: string; gone: boolean; detail: string } | null>(null);
  const deviceErr = computed(() => {
    const e = deviceError.get();
    if (e && e.device === device.get()) return e;
    return onCloud.get() ? null : missingDevice(device.get(), system.data);
  });
  const deviceMsg = computed(() => deviceMessage(deviceErr.get(), system));

  const rawErrors = computed(() =>
    seedCopy.errors(
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
    ),
  );
  // Suppress transient size-mismatch error while an unchecked real-server checkpoint's canonical arch is resolving.
  const errors = computed(() => {
    const e = rawErrors.get();
    return resolvingWeights.get() && e.weights ? { ...e, weights: null } : e;
  });
  const valid = computed(() => prefilled.get() && !pendingWeights.get() && !resolvingWeights.get() && errorCount(errors.get()) === 0);
  /**
   * Settings no estimate can stand for: no dataset, or one with nothing to train on, or an epoch count (or Close mosaic)
   * out of range. The Summary shows dashes (never "0 optimizer steps", "under 1 min" or "24–55 h" for 99,999 epochs),
   * no memory verdict, and the server is not asked.
   */
  const noFigures = computed(() => !!errors.get().dataset || !isInt(epochs.get(), 1, 10000) || !!errors.get().close);
  /** The first problem, named by its field (shown beside a disabled Start, as a way to that field). */
  const why = computed(() => firstError(errors.get()));
  const showWhy = computed(() => !valid.get() && !!why.get() && prefilled.get() && !pendingWeights.get() && !resolvingWeights.get());
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
  const estKey = computed(() => estimateKey(config.get()));
  const { served, estimating } = serverEstimate(
    config,
    estKey,
    () =>
      canEstimate.get() &&
      prefilled.get() &&
      !pendingWeights.get() &&
      !noFigures.get() &&
      !(onCloud.get() && isFolo2Arch(arch.get())) &&
      (!onCloud.get() || !!cloudPick.compute),
    ampEstimated,
  );
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
  const estHigh = computed(() => (estReady.get() ? shown.get().hi : null));
  // The time limit follows the estimate's high end until it is set by hand.
  effect(() => cloudPick.follow(estHigh.get()));
  const cloudCost = computed(() => cloudCostOf(onCloud.get(), cloudPick, served.get(), shown.get()));

  // Settings forecast not to fit: the one primary becomes the change that fits ("Use batch 8"), chosen by lib/oom's
  // `fitFix` from the same forecast as the figure above (the server's estimate, else the reference model) with 10% of
  // the memory to spare; Start steps back to a secondary with the risk said beside it. Asked once the estimate for
  // these settings is in, so the two never disagree.
  const fitted = signal<{ key: string; fix: OomFix | null } | null>(null);
  effect(() => {
    const key = estKey.get();
    if (!prefilled.get() || !!pendingWeights.get() || !noFit.get() || estimating.get() || untrack(() => fitted.get()?.key) === key) return;
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

  // The same run from a terminal (runnew/CliCommand): the server's command for these settings, where it built one.
  const cli = cliOf(
    () => config.get(),
    () => dataset.get()?.path ?? datasetId.get(),
    () => models.data?.find((m) => m.id === weights.get()),
    () => {
      const s = served.get();
      if (!s || s.key !== estKey.get()) return null;
      const cmd = serverCommand(s.est);
      return cmd ? withServerCliName(cmd, s.name, config.get().name) : null;
    },
  );

  async function jump(key: Jump) {
    sumOpen.set(false);
    if (key === 'opt' || key === 'lr0' || key === 'seed' || key === 'head') advancedOpen.set(true);
    if (key === 'close') augOpen.set(true);
    await tick();
    // A closing dialog hands focus back to its opener; move on after that.
    requestAnimationFrame(() => landOn(key));
  }

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
      runsChanged();
      draftStore.markStarted();
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
        if (err instanceof ApiError && err.code === 'dataset_not_found') void datasets.refresh(); // `gone` says so
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

  return {
    datasetId,
    startEl,
    formEl,
    startVisible,
    arch,
    epochs,
    batch,
    imgsz,
    imgszChosen,
    device,
    optimizer,
    amp,
    e2e,
    lr0,
    seed,
    weights,
    name,
    nameEdited,
    augment,
    closeEdited,
    submitting,
    shortWhy,
    advancedOpen,
    augOpen,
    sumOpen,
    datasets,
    system,
    folo2Supported,
    cloudPick,
    cloudOn,
    onCloud,
    ahead,
    startLabel,
    taken,
    tp,
    prefilled,
    pendingWeights,
    dataset,
    weightChoices,
    otherChoices,
    publishedOptions,
    scoreCtx,
    startHint,
    pickedAbout,
    pickedPublished,
    pickWeights,
    gpu,
    sampleImages,
    datasetImgsz,
    batchN,
    onCpu,
    caps,
    ampBlock,
    ampRefused,
    ampErr,
    ampOn,
    est,
    sched,
    stepsOpen,
    estReady,
    gpuMem,
    memShown,
    oomRun,
    oomChanges,
    was,
    scale,
    deviceErr,
    deviceMsg,
    errors,
    valid,
    noFigures,
    why,
    showWhy,
    short,
    selArch,
    closeEff,
    estStale,
    estimating,
    shown,
    vram,
    vramPct,
    vramLevel,
    noFit,
    vramFit,
    cloudCost,
    fix,
    fixLoading,
    applyFix,
    cli,
    review,
    jump,
    submit,
  };
}

export type RunNewState = ReturnType<typeof runNewState>;
