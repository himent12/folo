<script lang="ts">
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import Images from '@lucide/svelte/icons/images';
  import Play from '@lucide/svelte/icons/play';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import { tick, untrack } from 'svelte';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Arch, Augment, Dataset, DatasetImage, Device, Model, Optimizer, Run, RunEstimate, TrainConfig } from '$lib/api/types';
  import { modelChoices, scoreContext, scoreMeta } from '$lib/scoreSummary';
  import { FROM_SCRATCH } from '$lib/lineage';
  import { publishedSummary } from '$lib/published';
  import { compact, deviceName, num, sentence } from '$lib/format';
  import { hasCloud } from '$lib/cloud';
  import { fitFix, IMGSZ_STEPS, isOom, oomCalibration, useLabel, vramForecaster, type OomFix } from '$lib/oom';
  import { REF_CPU, REF_GPU, referenceSource } from '$lib/reference';
  import { basisLine, confidenceLine, hasFeature, referenceBasis, updatedWords } from '$lib/throughput';
  import { href, router } from '$lib/router.svelte';
  import { serverCommand, trainCommand } from '$lib/trainCli';
  import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import { HEAD_LABEL, optimizerLabel } from './run/resolved';
  import { CloudPick } from './runnew/cloudPick.svelte';
  import { arches, autoCloseMosaic, closeMosaicEpochs, defaultImgsz, estimate, shortRun, warmup } from './runnew/estimate';
  import { ampBlockOf } from './runnew/amp';
  import { FIELD, JUMP, jumpTarget, type Jump } from './runnew/jump';
  import { roughBytes } from './runnew/summary';
  import DatasetSection from './runnew/DatasetSection.svelte';
  import ModelSection from './runnew/ModelSection.svelte';
  import TrainingSection from './runnew/TrainingSection.svelte';
  import AugmentSection from './runnew/AugmentSection.svelte';
  import ReviewList from './runnew/ReviewList.svelte';
  import EstimateFigures from './runnew/EstimateFigures.svelte';
  import CliCommand from './runnew/CliCommand.svelte';
  import OomNote from './runnew/OomNote.svelte';
  import StartDock from './runnew/StartDock.svelte';

  setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'New run' });
  // This page is the create action: the top bar's New run steps aside instead of showing a pressed, disabled copy.
  ownCreateAction();


  const q = router.route.query;
  const defaults: Augment = { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 };

  let datasetId = $state<string>(q.get('dataset') ?? '');
  // The dock repeats the summary's Start button for narrow screens; it steps aside while that button is on screen.
  let startEl = $state<HTMLElement>();
  let startVisible = $state(false);
  $effect(() => {
    const el = startEl;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const io = new IntersectionObserver(([e]) => (startVisible = !!e?.isIntersecting));
    io.observe(el);
    return () => io.disconnect();
  });
  let arch = $state<Arch>('folo-s');
  let epochs = $state(100);
  let batch = $state('16');
  const IMGSZ = IMGSZ_STEPS.map(String);
  // ?imgsz= (e.g. from a dataset's "Train at 960 px" health finding) preselects a supported size. Until one is chosen
  // (there, by hand or by a retry) the size follows the dataset (see `defaultImgsz`).
  const queryImgsz = IMGSZ.includes(q.get('imgsz') ?? '');
  let imgsz = $state(queryImgsz ? q.get('imgsz')! : '640');
  let imgszChosen = $state(queryImgsz);
  let device = $state<string>('auto');
  let optimizer = $state<Optimizer>('auto');
  /** Mixed precision as asked for; what the run gets is `ampOn` (off where the server cannot train that way). */
  let amp = $state(false);
  /** The end-to-end head (TrainConfig.head `o2o`, docs/MODEL.md §6.6): predicts without NMS. */
  let e2e = $state(false);
  let lr0 = $state(0.01);
  let seed = $state(0);
  let weights = $state<string>('none');
  let name = $state('');
  let nameEdited = $state(false);
  let augment = $state<Augment>(structuredClone(defaults));
  // Close mosaic follows the epoch count (default 10, never more than half the run, the trainer's own cap) until it is
  // set by hand; the trainer caps a hand-set value the same way, so a short run always trains with mosaic.
  let closeEdited = $state(false);
  $effect(() => {
    const e = epochs;
    if (closeEdited || !Number.isFinite(e) || e < 1) return;
    const want = autoCloseMosaic(e);
    if (untrack(() => augment.close_mosaic) !== want) augment.close_mosaic = want;
  });
  let submitting = $state(false);
  /** The short-run warning's reasons, folded behind its "Why?". */
  let shortWhy = $state(false);
  let advancedOpen = $state(false);
  /** The augmentation settings fold behind a one-line summary at every width until opened. */
  let augOpen = $state(false);
  /** Below 960 px the dock opens the summary as a sheet, so the run can be reviewed without scrolling down. */
  let sumOpen = $state(false);

  const datasets = new Resource(async (s) => {
    const list = await hub.api.datasets(s);
    return Promise.all(list.map((d) => hub.api.dataset(d.id, s).catch(() => d)));
  });
  const system = new Resource((s) => hub.api.system(s));
  // Run on (servers with the `cloud` feature): this machine or a cloud connection, its GPU and the time limit.
  const cloudPick = new CloudPick(() => system.data);
  const cloudOn = $derived(hasCloud(system.data));
  /** A connection is chosen: the run trains on its GPU, not this machine's. */
  const onCloud = $derived(cloudOn && cloudPick.cloud);
  const models = new Resource((s) => hub.api.models(s));
  const runs = new Resource((s) => hub.api.runs(s));
  // This server's measured speed (GET /api/system/throughput), where it has one: measured latency in the size table.
  const tp = new Resource((s) => (hasFeature(system.data, 'throughput') ? hub.api.throughput(s).catch(() => null) : Promise.resolve(null)));

  // Prefill from an earlier run (?from=run_x, e.g. "Retry with batch 32"). Loaded through the hub, so it waits for
  // the real server instead of asking the mock that answers before the connection settles.
  const fromId = q.get('from');
  const source = new Resource((s) => (fromId ? hub.api.run(fromId, s) : Promise.resolve(null)));
  let prefilled = $state(!fromId);
  $effect(() => {
    const r = source.data;
    if (!r || untrack(() => prefilled)) return;
    const c = r.config;
    // A retry's fix comes in the query (?batch=, ?imgsz=, ?arch=, ?device=; see $lib/oom) and wins over the run's.
    // A value the trainer would reject is ignored (the run's own stays), never offered with Start enabled.
    const whole = (k: string, lo: number, hi: number, step = 1) => {
      const v = Number(q.get(k));
      return Number.isInteger(v) && v >= lo && v <= hi && v % step === 0 ? String(v) : null;
    };
    datasetId = c.dataset_id;
    arch = arches.find((a) => a.arch === q.get('arch'))?.arch ?? c.arch;
    epochs = Number(whole('epochs', 1, 10000) ?? c.epochs); // e.g. a short run's "Train N epochs"
    batch = whole('batch', 1, 1024) ?? String(c.batch);
    imgsz = whole('imgsz', 64, 4096, 32) ?? String(c.imgsz); // the trainer's limits (docs/API.md, runs)
    imgszChosen = true;
    device = q.get('device') === 'cpu' ? 'cpu' : c.device;
    optimizer = c.optimizer;
    amp = c.amp ?? false;
    e2e = c.head === 'o2o';
    lr0 = c.lr0;
    seed = c.seed;
    weights = c.weights ?? 'none';
    // A cloud run's connection, GPU and limit, where they still exist (else this machine, and Run on says why).
    cloudPick.prefill(c.compute, r.cloud?.offer.gpu ?? 'GPU');
    augment = structuredClone(c.augment);
    closeEdited = true;
    name = `${c.name}-retry`;
    nameEdited = true;
    prefilled = true;
  });
  $effect(() => {
    if (source.error && !untrack(() => prefilled)) {
      // page-level: loading the run to copy, before any dialog opens.
      toast.error(source.error, 'Could not load the run to copy');
      prefilled = true;
    }
  });

  $effect(() => {
    const list = datasets.data;
    if (list?.length && !untrack(() => datasetId)) datasetId = list[0]!.id;
  });

  const dataset = $derived(datasets.data?.find((d) => d.id === datasetId));

  // A fine-tune trains the weights' own architecture (docs/API.md, TrainConfig.weights), so the weights listed are the
  // chosen size's and switching size drops another size's choice. Published weights (`folo weights pull`, `source:
  // published`) are listed whatever the size, in a group of their own: picking them sets the size to theirs. Fused
  // weights (an export) load for inference only, so they are never a starting point.
  const isPublished = (m: Model) => m.source === 'published';
  const startable = $derived((models.data ?? []).filter((m) => m.fused !== true));
  const weightOptions = $derived(startable.filter((m) => !isPublished(m) && m.arch === arch));
  const publishedOptions = $derived(startable.filter(isPublished));
  // Each starting point's score with what it is on (lib/scoreSummary): grouped by dataset and split, best first.
  const scoreCtx = $derived.by(() => {
    const runById = new Map<string, Run>((runs.data ?? []).map((r) => [r.id, r]));
    const dsById = new Map<string, Dataset>((datasets.data ?? []).map((d) => [d.id, d]));
    return (m: Model) => scoreContext(m, runById, dsById);
  });
  const weightChoices = $derived(modelChoices(weightOptions, scoreCtx));
  const picked = $derived(weights === 'none' ? undefined : startable.find((m) => m.id === weights));
  /** Start from's hint: the picked weights' score with all of its context, and what published weights are. */
  const startHint = $derived.by(() => {
    if (!picked) return 'Fine-tuning from a trained model converges much faster.';
    // Published weights: what they are for, in plain words; their provenance and score are in the ⓘ beside the label
    // (pickedAbout). Trained ones are their score with its context.
    if (isPublished(picked)) return publishedSummary(picked);
    return scoreMeta(picked.metrics?.map50_95, scoreCtx(picked)) || 'Fine-tuning from a trained model converges much faster.';
  });
  /** Published weights' provenance: the server's description, then the score with what it is on. */
  const pickedAbout = $derived(
    picked && isPublished(picked) ? [picked.about, scoreMeta(picked.metrics?.map50_95, scoreCtx(picked))].filter(Boolean).join(' · ') : null,
  );
  /** Published weights picked: the run's size is theirs, said under the size picker. */
  const pickedPublished = $derived(picked && isPublished(picked) ? picked : null);
  $effect(() => {
    const a = arch;
    const list = startable;
    if (!models.data) return;
    const w = untrack(() => weights);
    if (w === 'none') return;
    const m = list.find((x) => x.id === w);
    if (!m || m.arch !== a) weights = 'none';
  });
  /** Start from picked: published weights of another size bring their size (the size picker follows). */
  function pickWeights(id: string) {
    const m = startable.find((x) => x.id === id);
    if (m && isPublished(m) && m.arch !== arch) arch = m.arch;
  }
  const gpu = $derived(system.data?.gpus[0]);

  // The suggested name is unique among existing runs (dataset-size, then -2, -3, …), so runs stay tellable apart.
  $effect(() => {
    const base = `${dataset?.name ?? 'run'}-${arch.replace('folo-', '')}`;
    const taken = new Set((runs.data ?? []).map((r) => r.name));
    let auto = base;
    for (let k = 2; taken.has(auto); k++) auto = `${base}-${k}`;
    if (!untrack(() => nameEdited)) name = auto;
  });

  // A few training images for the augmentation preview.
  const sampleImages = new Resource<DatasetImage[]>(async (s) => {
    const id = datasetId;
    if (!id) return [];
    return (await hub.api.datasetImages(id, { split: 'train', offset: 0, limit: 12 }, s)).images.filter((i) => i.boxes.length > 0);
  });

  // The training images' sizes, to train a dataset of small images at its own size. Read only for a dataset small
  // enough to list in a few pages (the stats do not carry image sizes); otherwise null, and the size stays 640.
  const trainSizes = new Resource(async (s) => {
    const id = datasetId;
    if (!id) return null;
    const MAX = 2000;
    const sizes: (readonly [number, number])[] = [];
    let total = Infinity;
    for (let page = 0; page < 8 && sizes.length < Math.min(total, MAX); page++) {
      const r = await hub.api.datasetImages(id, { split: 'train', offset: sizes.length, limit: 1000 }, s);
      total = r.total;
      if (total > MAX || !r.images.length) break;
      sizes.push(...r.images.map((i) => [i.width, i.height] as const));
    }
    return { id, sizes: sizes.length === total ? sizes : null };
  });
  const datasetImgsz = $derived(trainSizes.data && trainSizes.data.id === datasetId ? defaultImgsz(trainSizes.data.sizes) : null);
  // Until the current dataset's sizes are known (or when they cannot be read), an unchosen size is the 640 default:
  // never the size fitted to the dataset picked before.
  $effect(() => {
    const d = datasetImgsz;
    if (!untrack(() => imgszChosen)) imgsz = String(d ?? 640);
  });


  /** Number inputs bind `null` when cleared; accept only finite whole numbers in range. */
  const isInt = (v: unknown, lo: number, hi: number) => typeof v === 'number' && Number.isInteger(v) && v >= lo && v <= hi;

  const batchN = $derived(Math.max(1, Math.round(Number(batch) || 16)));
  const imgszN = $derived(Number(imgsz) || 640);
  const onCpu = $derived(!onCloud && (device === 'cpu' || (device === 'auto' && !gpu)));

  // Mixed precision (TrainConfig.amp): shown only when the server says what `auto` can use (`capabilities`, absent from
  // older servers), and off where a run would fail with `not_supported`: one plain sentence, the command that works
  // (ui/Remedy: the shell, one unwrapped copyable line), the server's own words kept verbatim behind "Driver report".
  const caps = $derived(system.data?.capabilities);
  const ampBlock = $derived(ampBlockOf(caps, device, onCloud, system.data));
  const ampOn = $derived(amp && (onCloud || (!!caps && !ampBlock)));

  /** Epochs without mosaic as the trainer runs them: every epoch when mosaic is 0, else at most half the run. */
  const closeRun = $derived(augment.mosaic <= 0 ? epochs : closeMosaicEpochs(augment.close_mosaic, epochs));
  /** This server's device is the reference machine's, so the reference's narrower range applies. */
  const refMachine = $derived(!onCloud && (onCpu ? system.data?.cpu.name === REF_CPU : gpu?.name === REF_GPU));
  /** The local estimate: docs/BENCHMARKS.md's reference, while the server's loads or where it has none. */
  const est = $derived(
    estimate(arch, dataset?.splits.train ?? 0, epochs, batchN, imgszN, onCpu, ampOn, { valImages: dataset?.splits.val ?? 0, closeEpochs: closeRun, sameMachine: refMachine }),
  );
  /** The trainer's schedule: optimizer steps an epoch, fewer than batches when gradients accumulate. */
  const sched = $derived(warmup(dataset?.splits.train ?? 0, Number.isInteger(epochs) && epochs > 0 ? epochs : 1, batchN));
  let stepsOpen = $state(false);
  // The figures wait for the server's devices and the dataset: until then "Estimating…", never a CPU guess or 0 steps.
  // A cloud run's figures wait for its GPU too.
  const estReady = $derived(!!dataset && (system.data !== undefined || system.error !== undefined) && (!onCloud || !!cloudPick.compute));
  // wgpu cannot report adapter memory (real servers send 0): then there is no meter and no out-of-memory guess. A cloud
  // run's memory is its offer's.
  const gpuMem = $derived(onCloud ? cloudPick.offer?.vram_bytes || null : gpu && gpu.memory_bytes > 0 ? gpu.memory_bytes : null);
  /** Whether GPU memory applies: a GPU of this machine, or the cloud offer's. */
  const memShown = $derived(onCloud ? !!cloudPick.offer : !onCpu && !!gpu);
  // The cheapest offer with the memory these settings need (the reference model's forecast), until one is picked.
  $effect(() => {
    if (onCloud) cloudPick.preselect(dataset ? est.vramBytes : null);
  });

  // A retry of a run that ran out of GPU memory: what changed since, and an estimate that agrees with the failure.
  const oomRun = $derived(source.data && source.data.status === 'failed' && isOom(source.data.error) ? source.data : null);
  const oomChanges = $derived.by(() => {
    const c = oomRun?.config;
    if (!c) return [];
    const out: { key: 'batch' | 'imgsz' | 'model' | 'device'; was: string; now: string }[] = [];
    if (c.batch !== batchN) out.push({ key: 'batch', was: `batch ${c.batch}`, now: `batch ${batchN}` });
    if (c.imgsz !== imgszN) out.push({ key: 'imgsz', was: `${c.imgsz} px`, now: `${imgszN} px` });
    if (c.arch !== arch) out.push({ key: 'model', was: c.arch, now: arch });
    if (c.device !== device) out.push({ key: 'device', was: c.device, now: device });
    return out;
  });
  const was = (key: string) => oomChanges.find((x) => x.key === key)?.was;
  // The estimate is rough; when it thought the failed settings fit, it is scaled up until they no longer do.
  // The run page forecasts its fixes with the same correction (lib/oom), so the retry it offers is shown here as fitting.
  const scale = $derived(oomRun ? oomCalibration(oomRun.config, gpuMem) : 1);

  // Start answered `501 not_supported` for mixed precision: the server cannot train that way after all (restarted
  // without tensor cores since it said it could). Its message stays at the Mixed precision field until the switch or
  // the device changes; the capabilities are read again, so the field shows what works now (off, with the remedy).
  // While the switch is still on, Start waits (a field in error gates the form).
  let ampRefused = $state<{ device: string; message: string } | null>(null);
  const ampErr = $derived(ampRefused && ampRefused.device === device ? ampRefused : null);

  // Start answered `404 device_not_found` (`501 not_implemented` from servers before it): said at the Device field in
  // one line built from the devices this machine lists now, until another device is picked. The server's own words
  // (they name its backends) fold into the field's technical details.
  let deviceError = $state<{ device: string; gone: boolean; detail: string } | null>(null);
  const deviceErr = $derived(deviceError && deviceError.device === device ? deviceError : null);
  /** A device in a sentence, named once with its value: "Auto", "GPU 1 (gpu:1)", "CPU" (values elsewhere: `gpu:1`). */
  const devWord = (d: string) => (d === 'auto' ? 'Auto' : d === 'cpu' ? 'CPU' : `${d.replace(/^gpu:/, 'GPU ')} (${d})`);
  const deviceMsg = $derived.by(() => {
    const e = deviceErr;
    if (!e) return null;
    if (!e.gone) return `This server cannot train on ${devWord(e.device)}. Pick Auto or CPU.`;
    const others = [...(system.data?.gpus ?? []).map((g) => `gpu:${g.ordinal}`).filter((d) => d !== e.device), 'cpu'].map(devWord);
    return `${devWord(e.device)} is no longer available. Pick Auto, ${others.length > 1 ? `${others.slice(0, -1).join(', ')} or ${others.at(-1)}` : others[0]}.`;
  });

  const errors = $derived({
    name: !name.trim() ? 'Give the run a name.' : /[^\w.-]/.test(name.trim()) ? 'Letters, digits, dot, dash and underscore only.' : null,
    epochs: !isInt(epochs, 1, 10000) ? 'A whole number from 1 to 10,000.' : null,
    lr0: !(typeof lr0 === 'number' && lr0 > 0 && lr0 < 10) ? 'Greater than 0 and below 10.' : null,
    seed: !isInt(seed, 0, 2 ** 32 - 1) ? 'A whole number, 0 or more.' : null,
    close: !isInt(augment.close_mosaic, 0, Number.isFinite(epochs) ? epochs : 0) ? 'A whole number up to the epoch count.' : null,
    dataset: !dataset ? 'Pick a dataset.' : null,
    weights: weights !== 'none' && picked?.arch !== arch ? 'Pick weights of the same model size.' : null,
    // The field says what to pick; the line beside Start only names the problem and leads there.
    device: deviceErr && !onCloud ? `${devWord(device)} is not available.` : null,
    amp: ampErr && ampOn && !onCloud ? 'Not available on this server now.' : null,
    runon: cloudOn ? cloudPick.errors.runon : null,
    offer: onCloud ? cloudPick.errors.offer : null,
    hours: onCloud ? cloudPick.errors.hours : null,
  });
  const valid = $derived(prefilled && Object.values(errors).every((e) => e === null));
  /** The first problem, named by its field (shown beside a disabled Start, as a way to that field). */
  const why = $derived.by(() => {
    for (const [k, v] of Object.entries(errors)) if (v) return { text: `${FIELD[k as keyof typeof errors]}: ${v}`, key: JUMP[k as keyof typeof errors] };
    return null;
  });
  const showWhy = $derived(!valid && !!why && prefilled);
  // The trainer's own short-run warnings (too few steps from scratch, mostly warmup): not errors, a warning with the fix.
  const short = $derived(errors.epochs ? null : shortRun(dataset?.splits.train ?? 0, epochs, batchN, weights === 'none'));
  const selArch = $derived(arches.find((a) => a.arch === arch));

  const config = $derived<TrainConfig>({
    name: name.trim(),
    dataset_id: datasetId,
    arch,
    epochs,
    batch: batchN,
    imgsz: imgszN,
    // A cloud run's worker picks its own GPU (docs/API.md, Cloud runs).
    device: onCloud ? 'auto' : (device as Device),
    optimizer,
    lr0,
    weights: weights === 'none' ? null : weights,
    augment: { ...augment, hsv: [...augment.hsv] as [number, number, number], close_mosaic: Math.min(augment.close_mosaic, epochs) },
    seed,
    // Only to servers that report capabilities (older ones predate `amp`), or train it in the cloud.
    ...(caps || onCloud ? { amp: ampOn } : {}),
    // Left out for the default (missing means `o2m`, docs/API.md), so older servers see the request they know.
    ...(e2e ? { head: 'o2o' as const } : {}),
    // Only once a connection and its GPU are chosen, and only to servers with the feature.
    ...(onCloud && cloudPick.compute ? { compute: cloudPick.compute } : {}),
  });

  // The server's estimate (POST /api/runs/estimate, feature `run_estimate`): the first asked at once, every later one
  // 300 ms after the settings stop changing, whether or not the last one was answered (a slow or failing server is not
  // asked once per keystroke); the last answer stays up meanwhile. Its errors are the form's own (a missing device,
  // weights of another size), already said at their fields: the reference figures stand in.
  const canEstimate = $derived(hasFeature(system.data, 'run_estimate'));
  let served = $state.raw<{ key: string; est: RunEstimate } | null>(null);
  /** Whether the latest request is still out: set by the run of the effect that owns it, cleared by every other. */
  let estimating = $state(false);
  /** Whether an estimate has been asked for yet (not state: it only sets the next one's delay). */
  let asked = false;
  /** The latest request's number: an answer to an older (cancelled) one never touches the state. */
  let estSeq = 0;
  // The name is part of the request: the server's command (`cli`) names the run.
  const estKey = $derived(JSON.stringify(config));
  $effect(() => {
    const key = estKey;
    const body = { ...config, name: config.name || 'estimate' };
    const seq = ++estSeq;
    // No request for these settings (none possible, or the answer is already in): none is pending either, even where
    // the one this replaced (since cancelled) was.
    if (!canEstimate || !dataset || !isInt(epochs, 1, 10000) || errors.close || (onCloud && !cloudPick.compute)) {
      served = null;
      estimating = false;
      return;
    }
    if (untrack(() => served?.key) === key) {
      estimating = false;
      return;
    }
    const ctrl = new AbortController();
    estimating = true;
    const t = setTimeout(
      () => {
        asked = true;
        hub.api.estimateRun(body, ctrl.signal).then(
          (e) => {
            if (ctrl.signal.aborted || seq !== estSeq) return;
            served = { key, est: e };
            estimating = false;
          },
          () => {
            if (ctrl.signal.aborted || seq !== estSeq) return;
            served = null;
            estimating = false;
          },
        );
      },
      asked ? 300 : 0,
    );
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  });
  /**
   * Close mosaic as the trainer applies it: the server's count for these settings (`RunEstimate.close_mosaic`), else
   * the trainer's cap of the request (at most half the run), which the server's agrees with.
   */
  const closeEff = $derived.by(() => {
    if (errors.close || errors.epochs) return null;
    const n = served?.key === estKey ? served.est.close_mosaic : undefined;
    return typeof n === 'number' && Number.isFinite(n) ? n : closeMosaicEpochs(augment.close_mosaic, epochs);
  });
  /**
   * Whether the figures shown answer earlier settings: the last answer stays up while the request for these is out, so
   * the form never empties as it is edited, but it is not these settings' figure (DESIGN_DIRECTION §2.8, "a figure is
   * either current or visibly marked as not"): Est. time and Est. GPU memory dim to tertiary, in the Summary and the
   * dock, and the basis line reads "Updating…" until the answer for these settings is in.
   */
  const estStale = $derived(!!served && served.key !== estKey);
  /** What the Summary and the dock show: the server's estimate when there is one, else the reference's. */
  const shown = $derived.by(() => {
    const r = served?.est;
    if (r)
      return {
        lo: r.est_total_s_low,
        hi: r.est_total_s_high,
        vram: r.est_peak_vram_bytes ?? 0,
        vramMeasured: r.vram_source === 'measured',
        basis: onCloud ? 'The reference machine’s speed' : basisLine(r),
        rough: r.confidence === 'low',
        how: onCloud ? sentence(r.basis.note) : confidenceLine(r),
        phases: r.phases,
        // A measured basis whose validation rate is still the reference's says so on that line.
        val: { images: r.val_images, rate: r.val_imgs_per_s, reference: r.basis.source === 'measured' && r.val_source === 'reference' },
        startup: r.startup_s,
        updated: r.basis.source === 'measured' && r.basis.updated ? updatedWords(r.basis) : null,
      };
    return {
      lo: est.lowSeconds,
      hi: est.highSeconds,
      vram: est.vramBytes,
      vramMeasured: false,
      basis: referenceBasis(refMachine, onCpu),
      rough: true,
      how: !canEstimate
        ? `Estimated from ${referenceSource(onCpu)}, scaled to these settings: this server does not estimate runs.`
        : estimating
          ? "Loading this server's estimate."
          : `Estimated from ${referenceSource(onCpu)}, scaled to these settings: this server could not estimate them.`,
      phases: est.phases,
      val: { images: est.valImages, rate: est.valImgsPerS, reference: false },
      startup: est.startupS,
      updated: null,
    };
  });
  // Measured peaks need no scaling: only the reference model is corrected by a failed run.
  const vram = $derived(shown.vram * (shown.vramMeasured ? 1 : scale));
  const vramPct = $derived(gpuMem ? vram / gpuMem : 0);
  /**
   * One condition, one severity, on the meter and the line under it (the figure stays in ink): over the GPU's memory is
   * `danger` (likely won't fit), within 10% of it `warning` (tight; lib/oom's headroom), else neutral.
   */
  const vramLevel = $derived(!memShown || !gpuMem ? null : vramPct > 1 ? ('danger' as const) : vramPct > 0.9 ? ('warning' as const) : null);
  const noFit = $derived(vramLevel === 'danger');
  /**
   * Whether these settings fit the GPU: `fits`, `no` (forecast over its memory) or `unknown` (the adapter does not report
   * its memory, so nothing can be checked); null on the CPU, where GPU memory does not apply.
   */
  const vramFit = $derived(!memShown ? null : !gpuMem ? ('unknown' as const) : noFit ? ('no' as const) : ('fits' as const));
  // The time limit follows the estimate's high end until it is set by hand.
  $effect(() => cloudPick.follow(estReady ? shown.hi : null));
  /**
   * What a cloud run costs: the server's range for these settings (`RunEstimate.cost`), else the offer's price over the
   * time shown; the most it can cost, at its time limit. Null without a price (compute that is not billed).
   */
  const cloudCost = $derived.by(() => {
    const price = onCloud ? (cloudPick.offer?.price_per_hour ?? null) : null;
    if (price === null) return null;
    const c = served?.est.cost;
    return c ? { lo: c.est_usd_low, hi: c.est_usd_high } : { lo: (price * shown.lo) / 3600, hi: (price * shown.hi) / 3600 };
  });

  // Settings forecast not to fit: the one primary becomes the change that fits ("Use batch 8"), chosen by lib/oom's
  // `fitFix` from the same forecast as the figure above (the server's estimate, else the reference model) with 10% of
  // the memory to spare; Start steps back to a secondary with the risk said beside it. Asked once the estimate for
  // these settings is in, so the two never disagree.
  let fitted = $state.raw<{ key: string; fix: OomFix | null } | null>(null);
  $effect(() => {
    const key = estKey;
    if (!noFit || estimating || untrack(() => fitted?.key) === key) return;
    const ctrl = new AbortController();
    const c = untrack(() => config);
    const forecaster = vramForecaster(hub.api, untrack(() => system.data), ctrl.signal);
    fitFix(c, gpuMem, forecaster, untrack(() => scale)).then(
      (fix) => !ctrl.signal.aborted && (fitted = { key, fix }),
      () => {},
    );
    return () => ctrl.abort();
  });
  const fix = $derived(noFit && fitted?.key === estKey ? fitted.fix : null);
  const fixLoading = $derived(noFit && fitted?.key !== estKey);
  /** Applies the change, as a retry's query would have (lib/oom `OomFix.query`). */
  function applyFix(f: OomFix) {
    for (const [k, v] of Object.entries(f.query)) {
      if (k === 'batch') batch = v;
      else if (k === 'imgsz') ((imgsz = v), (imgszChosen = true));
      else if (k === 'arch') arch = v as Arch;
      else if (k === 'device') device = v;
    }
  }

  // The same run from a terminal: the server's command where it builds one (only it knows its files), else Studio's,
  // from the CLI's own mapping ($lib/trainCli). A fine-tune names the weights file by path; where the server does
  // not report it, a placeholder and a note say what to fill in.
  const ownCli = $derived(trainCommand(config, dataset?.path ?? datasetId, models.data?.find((m) => m.id === weights)));
  const servedCli = $derived(served?.key === estKey ? serverCommand(served.est) : null);
  const cliParts = $derived(servedCli ? servedCli.split(/ (?=--)/) : ownCli.parts);
  const cli = $derived(servedCli ?? ownCli.line);
  const cliPlaceholder = $derived(servedCli ? null : ownCli.placeholder);

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    if (!valid || submitting) return;
    // The Summary sheet closes first: the start's result is the page's (a toast, or the Device field's message).
    sumOpen = false;
    submitting = true;
    try {
      const run = await hub.api.createRun(config);
      const where = run.cloud ? `, at ${cloudPick.provider} on ${run.cloud.offer.gpu}` : '';
      // page-level: the sheet is closed (above).
      toast.success(`Started ${run.name}`, `${run.epochs} epochs of ${run.config.arch} on ${dataset?.name}${where}.`);
      router.go(`/runs/${run.id}`);
    } catch (err) {
      if (err instanceof ApiError && (err.code === 'device_not_found' || (err.code === 'not_implemented' && device.startsWith('gpu:')))) {
        deviceError = {
          device,
          gone: err.code === 'device_not_found',
          detail: [err.message, err.hint].filter((x): x is string => !!x).map((x) => sentence(x, true)).join(' '),
        };
        // The adapters may have changed (an eGPU unplugged): list them again, and show the field with its message.
        void system.refresh();
        void jump('device');
        // page-level: the sheet closed when the form was sent.
      } else if (err instanceof ApiError && err.code === 'not_supported' && config.amp) {
        // At the field, not a toast: the message, then what the server can do now (the capabilities read again).
        ampRefused = { device, message: err.message };
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
      submitting = false;
    }
  }


  // ---------- review ----------
  const shortGpu = deviceName;
  const deviceLabel = $derived.by(() => {
    if (device === 'auto') return `Auto${gpu ? ` · ${shortGpu(gpu.name)}` : ''}`;
    if (device === 'cpu') return 'CPU';
    const g = system.data?.gpus.find((x) => `gpu:${x.ordinal}` === device);
    return g ? `${device} · ${shortGpu(g.name)}` : device;
  });
  /** The settings that shape the run, each a way back to its control. Optimizer and head only once not the default. */
  const review = $derived<{ key: Jump; label: string; value: string; mono?: boolean; warn?: boolean }[]>([
    { key: 'data', label: 'Data', value: dataset ? `${dataset.name} · ${compact(dataset.splits.train)} images` : '—', warn: !!errors.dataset },
    { key: 'model', label: 'Model', value: arch, mono: true },
    { key: 'epochs', label: 'Epochs', value: errors.epochs ? '—' : num(epochs), warn: !!errors.epochs },
    { key: 'batch', label: 'Batch', value: String(batchN) },
    { key: 'imgsz', label: 'Image size', value: `${imgszN} px` },
    // A cloud run says where it runs ("RunPod · RTX 4090") in the device's place.
    onCloud
      ? { key: 'runon' as const, label: 'Runs on', value: cloudPick.offer ? `${cloudPick.provider} · ${cloudPick.offer.gpu}` : cloudPick.provider, warn: !!errors.runon || !!errors.offer }
      : { key: 'device' as const, label: 'Device', value: deviceLabel, warn: !!deviceMsg },
    ...(caps || onCloud ? [{ key: 'amp' as const, label: 'Mixed precision', value: ampOn ? 'On' : 'Off' }] : []),
    {
      key: 'weights',
      label: 'Start from',
      value: weights === 'none' ? FROM_SCRATCH : (picked?.name ?? weights),
      mono: weights !== 'none',
      warn: !!errors.weights,
    },
    ...(optimizer !== 'auto' ? [{ key: 'opt' as const, label: 'Optimizer', value: optimizerLabel(optimizer) }] : []),
    ...(e2e ? [{ key: 'head' as const, label: 'Head', value: HEAD_LABEL.o2o }] : []),
  ]);


  async function jump(key: Jump) {
    sumOpen = false;
    if (key === 'opt' || key === 'lr0' || key === 'seed' || key === 'head') advancedOpen = true;
    if (key === 'close') augOpen = true;
    await tick();
    // A closing dialog hands focus back to its opener; move on after that.
    requestAnimationFrame(() => {
      const el = jumpTarget(key);
      if (!el) return;
      const box = el.closest<HTMLElement>('.field, .card, .arch, .ds-pick, .a-pick') ?? el;
      const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
      box.scrollIntoView({ block: 'center', behavior: reduce ? 'auto' : 'smooth' });
      el.focus({ preventScroll: true, focusVisible: true } as FocusOptions);
      // Where the browser keeps the focus ring hidden after a mouse click, mark the landing spot until focus leaves.
      if (el.matches(':focus-visible')) return;
      box.dataset.jumped = '';
      el.addEventListener('blur', () => delete box.dataset.jumped, { once: true });
    });
  }
</script>

{#snippet reviewList(withName: boolean)}
  <ReviewList {withName} {name} {errors} {review} {jump} />
{/snippet}

{#snippet figures(where: string)}
  <EstimateFigures
    {where}
    {estReady}
    {sched}
    {batchN}
    {est}
    bind:stepsOpen
    {shown}
    {estStale}
    {estimating}
    {memShown}
    {vram}
    {onCloud}
    {cloudPick}
    {cloudCost}
    {gpuMem}
    {scale}
    {onCpu}
    {vramLevel}
    {vramPct}
    {noFit}
    {fix}
    {fixLoading}
    {short}
    bind:shortWhy
    bind:epochs
  />
{/snippet}

<!-- Why Start is off, as a way to the field that needs changing (the field may be far away). The mark in front carries
     the danger hue; the link itself is ink, since danger never means navigation. -->
{#snippet whyLine(id: string, cls = '')}
  {#if showWhy && why}
    <p class="why {cls}" {id} role={id === 'start-why' ? 'status' : undefined}>
      <CircleAlert size={14} strokeWidth={2} aria-hidden="true" /><button type="button" class="why-go" onclick={() => why && jump(why.key)}>{why.text}<ArrowRight size={12} strokeWidth={2} aria-hidden="true" /></button>
    </p>
  {/if}
{/snippet}

{#snippet estimates(withName: boolean)}
  {@render reviewList(withName)}
  {@render figures('sheet')}
{/snippet}

<Page title="New training run">
  {#snippet sub()}<span>Pick data and a model size; the defaults are tuned to work well. Everything else is optional.</span>{/snippet}

  {#if datasets.error && !datasets.data}
    <ErrorState error={datasets.error} retry={() => datasets.refresh()} title="Could not load datasets" />
  {:else if datasets.data && datasets.data.length === 0}
    <Panel>
      <EmptyState icon={Images} title="Add a dataset first" description="Runs train on a registered dataset: add a folder or generate a synthetic one.">
        {#snippet actions()}<Button variant="primary" href={href('/datasets')}>Go to Datasets</Button>{/snippet}
      </EmptyState>
    </Panel>
  {:else}
    <form class="layout" onsubmit={submit} novalidate>
      {#if oomRun}
        <OomNote {oomRun} {oomChanges} {vramFit} {vram} {gpuMem} {vramLevel} />
      {/if}
      <div class="form">
        <DatasetSection {datasets} bind:datasetId {dataset} {errors} />

        <ModelSection
          bind:arch
          bind:weights
          {tp}
          {oomRun}
          {was}
          {pickedPublished}
          {selArch}
          {startHint}
          {pickedAbout}
          {errors}
          {weightChoices}
          {publishedOptions}
          {scoreCtx}
          {pickWeights}
        />

        <TrainingSection
          bind:epochs
          bind:batch
          bind:imgsz
          bind:imgszChosen
          bind:device
          bind:optimizer
          bind:amp
          bind:ampRefused
          bind:e2e
          bind:lr0
          bind:seed
          bind:advancedOpen
          {arch}
          {errors}
          {was}
          {oomRun}
          {dataset}
          {datasetImgsz}
          {IMGSZ}
          {cloudOn}
          {onCloud}
          {cloudPick}
          {system}
          {gpu}
          {deviceMsg}
          {deviceErr}
          {caps}
          {ampBlock}
          {ampErr}
          {ampOn}
        />

        <AugmentSection bind:augment bind:augOpen bind:closeEdited {defaults} {errors} {closeEff} {epochs} {dataset} {sampleImages} />
      </div>

      <aside class="summary">
        <Panel title="Summary">
          <!-- Two groups: what the run is (name, settings) and what it costs (estimates, Start). A rail stacks them; a
               full-width summary (below 960 px) sets them side by side. -->
          <div class="sum">
            <div class="sum-col">
              <Field label="Run name" for="name" error={errors.name}>
                <Input id="name" bind:value={name} oninput={() => (nameEdited = true)} invalid={!!errors.name} spellcheck="false" autocomplete="off" />
              </Field>
              {@render reviewList(false)}
            </div>
            <div class="sum-col">
            {@render figures('rail')}
            <div class="start" bind:this={startEl}>
              {#if noFit && (fix || fixLoading)}
                <Button variant="primary" size="lg" block loading={fixLoading} onclick={() => fix && applyFix(fix)}
                  >{fix ? `${useLabel(fix)}${fix.vram !== undefined ? ` · ${roughBytes(fix.vram)}` : ''}` : 'Finding settings that fit'}</Button
                >
              {/if}
              <Button
                type="submit"
                variant={noFit ? 'secondary' : 'primary'}
                size="lg"
                icon={Play}
                block
                loading={submitting}
                disabled={!valid}
                aria-describedby={[showWhy ? 'start-why' : '', noFit ? 'rail-nofit' : ''].filter(Boolean).join(' ') || undefined}>Start training</Button
              >
              {@render whyLine('start-why')}
            </div>
            <CliCommand {cliParts} {cli} {cliPlaceholder} />
            </div>
          </div>
        </Panel>
      </aside>

      <StartDock
        {startVisible}
        {showWhy}
        {why}
        {whyLine}
        bind:sumOpen
        {shown}
        {estReady}
        {estStale}
        {cloudCost}
        {memShown}
        {noFit}
        {gpuMem}
        {vram}
        {short}
        {dataset}
        {arch}
        {fix}
        {fixLoading}
        {applyFix}
        {submitting}
        {valid}
      />

      <!-- Inside the form, so its Start button submits it. -->
      <Dialog bind:open={sumOpen} title="Summary" size="sm">
        <div class="sum">
          {@render estimates(true)}
          {@render whyLine('sheet-why')}
        </div>
        {#snippet footer()}
          <Button variant="ghost" onclick={() => (sumOpen = false)}>Keep editing</Button>
          {#if noFit && (fix || fixLoading)}
            <Button type="submit" icon={Play} loading={submitting} disabled={!valid} aria-describedby={[showWhy ? 'sheet-why' : '', 'sheet-nofit'].filter(Boolean).join(' ')}>Start training</Button>
            <Button variant="primary" loading={fixLoading} onclick={() => fix && applyFix(fix)}>{fix ? useLabel(fix) : 'Finding a fit'}</Button>
          {:else}
            <Button type="submit" variant="primary" icon={Play} loading={submitting} disabled={!valid} aria-describedby={showWhy ? 'sheet-why' : undefined}>Start training</Button>
          {/if}
        {/snippet}
      </Dialog>
    </form>
  {/if}
</Page>

<style>
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
  .why {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    margin-top: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-primary);
  }
  .why > :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
    color: var(--danger-solid);
  }
  /* The reason Start is off leads to the field: an ink link, underlined, the arrow in tertiary. */
  .why-go {
    display: inline;
    min-width: 0;
    color: inherit;
    text-align: left;
    text-decoration: underline;
    text-decoration-color: color-mix(in srgb, currentColor 40%, transparent);
    text-underline-offset: 3px;
    border-radius: var(--radius-xs);
  }
  .why-go:hover {
    text-decoration-color: currentColor;
  }
  .why-go :global(svg) {
    display: inline;
    margin-left: var(--space-1);
    vertical-align: -1px;
    color: var(--text-tertiary);
  }
  @media (pointer: coarse) {
    .why {
      align-items: center;
    }
    .why > :global(svg) {
      margin-top: 0;
    }
    .why-go {
      display: inline-block;
      min-height: 44px;
      padding-block: var(--space-3);
    }
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
    /* The reason sits on the dock's left, where the estimate would be. */
    .dock-why {
      flex: 1;
      min-width: 0;
      margin-top: 0;
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
</style>
