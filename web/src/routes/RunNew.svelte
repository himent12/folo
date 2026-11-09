<script lang="ts">
  import Check from '@lucide/svelte/icons/check';
  import ChevronRight from '@lucide/svelte/icons/chevron-right';
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import ChevronUp from '@lucide/svelte/icons/chevron-up';
  import Copy from '@lucide/svelte/icons/copy';
  import Images from '@lucide/svelte/icons/images';
  import Play from '@lucide/svelte/icons/play';
  import Shuffle from '@lucide/svelte/icons/shuffle';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import CircleAlert from '@lucide/svelte/icons/circle-alert';
  import TriangleAlert from '@lucide/svelte/icons/triangle-alert';
  import { tick, untrack } from 'svelte';
  import { MediaQuery } from 'svelte/reactivity';
  import { ApiError, hub } from '$lib/api/index.svelte';
  import { Resource } from '$lib/api/resource.svelte';
  import type { Arch, Augment, DatasetImage, Device, Optimizer, RunEstimate, TrainConfig } from '$lib/api/types';
  import ClassStrip from '$lib/detect/ClassStrip.svelte';
  import { bytes, compact, deviceName, duration, num, sentence } from '$lib/format';
  import { IMGSZ_STEPS, isOom } from '$lib/oom';
  import { prefs } from '$lib/prefs.svelte';
  import { REF_CPU, REF_GPU } from '$lib/reference';
  import { basisLine, confidenceLine, hasFeature, latencyMs, latencyRange, measuredLatency, timeRange, updatedWords } from '$lib/throughput';
  import { href, router } from '$lib/router.svelte';
  import { serverCommand, trainCommand } from '$lib/trainCli';
  import { ownCreateAction, setCrumbs } from '$lib/shell/crumbs.svelte';
  import { dock } from '$lib/shell/dock';
  import { sidebar } from '$lib/shell/sidebarState.svelte';
  import Page from '$lib/shell/Page.svelte';
  import Button from '$lib/ui/Button.svelte';
  import Dialog from '$lib/ui/Dialog.svelte';
  import EmptyState from '$lib/ui/EmptyState.svelte';
  import ErrorState from '$lib/ui/ErrorState.svelte';
  import Field from '$lib/ui/Field.svelte';
  import Input from '$lib/ui/Input.svelte';
  import Panel from '$lib/ui/Panel.svelte';
  import Remedy from '$lib/ui/Remedy.svelte';
  import Segmented from '$lib/ui/Segmented.svelte';
  import Select from '$lib/ui/Select.svelte';
  import Skeleton from '$lib/ui/Skeleton.svelte';
  import Slider from '$lib/ui/Slider.svelte';
  import Switch from '$lib/ui/Switch.svelte';
  import Ident from '$lib/ui/Ident.svelte';
  import Meta from '$lib/ui/Meta.svelte';
  import { toast } from '$lib/ui/toast.svelte';
  import { tooltip } from '$lib/ui/tooltip';
  import { VULKAN } from './settings/capabilities';
  import { HEAD_LABEL } from './run/resolved';
  import AugPreview, { type AugKey } from './runnew/AugPreview.svelte';
  import { AMP_SPEEDUP, arches, autoCloseMosaic, closeMosaicEpochs, defaultImgsz, estimate, oomScale, shortRun, warmup } from './runnew/estimate';
  import { media } from '$lib/design/tokens';

  setCrumbs({ label: 'Runs', path: '/runs' }, { label: 'New run' });
  // This page is the create action: the top bar's New run steps aside instead of showing a pressed, disabled copy.
  ownCreateAction();

  /** Where the architectures' reference latency was measured (docs/BENCHMARKS.md). */
  const LATENCY_REF = 'RTX 4080 SUPER · 640 px · f32 · end to end';
  /** The same, where the line already says 640 px and end to end. */
  const REF_WHERE = 'RTX 4080 SUPER · f32';
  // Phones pick the model size from a segmented control; the full specification table folds behind "Compare sizes".
  const phone = new MediaQuery(media.phone, false);

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
  const BATCHES = ['8', '16', '32', '64'];
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
  let focus = $state<AugKey | null>(null);
  let shuffle = $state(1);
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
    augment = structuredClone(c.augment);
    closeEdited = true;
    name = `${c.name}-retry`;
    nameEdited = true;
    prefilled = true;
  });
  $effect(() => {
    if (source.error && !untrack(() => prefilled)) {
      toast.error(source.error, 'Could not load the run to copy');
      prefilled = true;
    }
  });

  $effect(() => {
    const list = datasets.data;
    if (list?.length && !untrack(() => datasetId)) datasetId = list[0]!.id;
  });

  const dataset = $derived(datasets.data?.find((d) => d.id === datasetId));

  // Pretrained weights must match the architecture; switching size drops an incompatible choice. Fused weights (an
  // export) load for inference only, so they are never a starting point.
  const weightOptions = $derived((models.data ?? []).filter((m) => m.arch === arch && m.fused !== true));
  $effect(() => {
    const opts = weightOptions;
    if (!models.data) return;
    if (untrack(() => weights) !== 'none' && !opts.some((m) => m.id === untrack(() => weights))) weights = 'none';
  });
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

  // Estimates are ranges (lib/throughput `timeRange`); memory is one figure, in the app's one size format.
  const roughBytes = (b: number) => `~${bytes(b, prefs.units)}`;

  /** Number inputs bind `null` when cleared; accept only finite whole numbers in range. */
  const isInt = (v: unknown, lo: number, hi: number) => typeof v === 'number' && Number.isInteger(v) && v >= lo && v <= hi;

  const batchN = $derived(Math.max(1, Math.round(Number(batch) || 16)));
  const imgszN = $derived(Number(imgsz) || 640);
  const onCpu = $derived(device === 'cpu' || (device === 'auto' && !gpu));

  // Mixed precision (TrainConfig.amp): shown only when the server says what `auto` can use (`capabilities`, absent from
  // older servers), and off where a run would fail with `not_supported`: one plain sentence, the command that works
  // (ui/Remedy: the shell, one unwrapped copyable line), the server's own words kept verbatim behind "Driver report".
  const caps = $derived(system.data?.capabilities);
  const ampBlock = $derived.by((): { why: string; shell?: string; command?: string; raw?: string } | null => {
    if (!caps) return null;
    if (device === 'cpu') return { why: 'The CPU has no tensor cores. Pick a GPU to train in mixed precision.' };
    // Auto on a server without a usable GPU is the CPU too: no GPU advice.
    if (device === 'auto' && caps.device === 'cpu') return { why: 'This server has no usable GPU, so runs train on the CPU in f32.' };
    // `capabilities` describe the device `auto` resolves to; another GPU is left to the server.
    if (caps.amp.available || (device !== 'auto' && device !== caps.device)) return null;
    const raw = caps.amp.reason ?? caps.tensor_cores.reason ?? undefined;
    const t = caps.tensor_cores;
    if (t.available && !t.enabled) {
      if (/self-test/i.test(t.reason ?? '')) return { why: "The server's tensor-core self-test failed. Train in f32 on this GPU.", raw };
      return { why: 'Tensor cores are off on this server; restart it with them on:', command: 'folo serve --tensor-cores', raw };
    }
    if (!t.available) {
      // docs/GPU.md: wgpu has cooperative matrices on Vulkan only (NVIDIA tensor cores through SPV_KHR_cooperative_matrix;
      // an RTX 4080 SUPER has them there), none on DX12 or Metal. So only an NVIDIA RTX card seen through DX12 has a way.
      const g = system.data?.gpus.find((x) => `gpu:${x.ordinal}` === caps.device);
      if (g?.backend === 'dx12' && /nvidia/i.test(g.name) && /\brtx\b/i.test(g.name))
        return { why: 'DirectX 12 does not reach this GPU’s tensor cores; restart the server on Vulkan.', shell: 'PowerShell', command: `${VULKAN} --tensor-cores`, raw };
      return { why: 'This GPU has no tensor cores FOLO can use. Train in f32 on this GPU.', raw };
    }
    return { why: 'Not available on this server. Train in f32 on this GPU.', raw };
  });
  const ampOn = $derived(amp && !!caps && !ampBlock);
  /** What the switch does, with the speed measured for this size (docs/BENCHMARKS.md), where there is one. */
  const ampHint = $derived.by(() => {
    const x = AMP_SPEEDUP[arch];
    return x ? `f16 on tensor cores: about ${x.toFixed(1)}× faster, 40% less memory.` : 'f16 on tensor cores: faster, about 40% less memory.';
  });

  /** Epochs without mosaic as the trainer runs them: every epoch when mosaic is 0, else at most half the run. */
  const closeRun = $derived(augment.mosaic <= 0 ? epochs : closeMosaicEpochs(augment.close_mosaic, epochs));
  /** This server's device is the reference machine's, so the reference's narrower range applies. */
  const refMachine = $derived(onCpu ? system.data?.cpu.name === REF_CPU : gpu?.name === REF_GPU);
  /** The local estimate: docs/BENCHMARKS.md's reference, while the server's loads or where it has none. */
  const est = $derived(
    estimate(arch, dataset?.splits.train ?? 0, epochs, batchN, imgszN, onCpu, ampOn, { valImages: dataset?.splits.val ?? 0, closeEpochs: closeRun, sameMachine: refMachine }),
  );
  /** The trainer's schedule: optimizer steps an epoch, fewer than batches when gradients accumulate. */
  const sched = $derived(warmup(dataset?.splits.train ?? 0, Number.isInteger(epochs) && epochs > 0 ? epochs : 1, batchN));
  /** The batches behind the steps, for the Summary's tooltip and its accessible description: one line in the rail. */
  const batchesOf = $derived(`${num(est.stepsPerEpoch)} ${est.stepsPerEpoch === 1 ? 'batch' : 'batches'} of ${batchN}`);
  const stepsWhy = $derived(sched.accumulate > 1 ? `${batchesOf}; ${sched.accumulate} per optimizer step.` : `${batchesOf}, one step each.`);
  // The figures wait for the server's devices and the dataset: until then "Estimating…", never a CPU guess or 0 steps.
  const estReady = $derived(!!dataset && (system.data !== undefined || system.error !== undefined));
  // wgpu cannot report adapter memory (real servers send 0): then there is no meter and no out-of-memory guess.
  const gpuMem = $derived(gpu && gpu.memory_bytes > 0 ? gpu.memory_bytes : null);

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
  const scale = $derived.by(() => {
    const c = oomRun?.config;
    if (!c || !gpuMem || c.device === 'cpu') return 1;
    return oomScale(estimate(c.arch, dataset?.splits.train ?? 0, c.epochs, c.batch, c.imgsz, false).vramBytes, gpuMem);
  });

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
    weights: weights !== 'none' && !weightOptions.some((m) => m.id === weights) ? 'Pick weights of the same model size.' : null,
    // The field says what to pick; the line beside Start only names the problem and leads there.
    device: deviceErr ? `${devWord(device)} is not available.` : null,
  });
  const valid = $derived(prefilled && Object.values(errors).every((e) => e === null));
  const FIELD: Record<keyof typeof errors, string> = {
    name: 'Run name',
    epochs: 'Epochs',
    lr0: 'Learning rate',
    seed: 'Seed',
    close: 'Close mosaic',
    dataset: 'Dataset',
    weights: 'Start from',
    device: 'Device',
  };
  const JUMP: Record<keyof typeof errors, Jump> = {
    name: 'name',
    epochs: 'epochs',
    lr0: 'lr0',
    seed: 'seed',
    close: 'close',
    dataset: 'data',
    weights: 'weights',
    device: 'device',
  };
  /** The first problem, named by its field (shown beside a disabled Start, as a way to that field). */
  const why = $derived.by(() => {
    for (const [k, v] of Object.entries(errors)) if (v) return { text: `${FIELD[k as keyof typeof errors]}: ${v}`, key: JUMP[k as keyof typeof errors] };
    return null;
  });
  const showWhy = $derived(!valid && !!why && prefilled);
  // The trainer's own short-run warnings (too few steps from scratch, mostly warmup): not errors, a warning with the fix.
  const short = $derived(errors.epochs ? null : shortRun(dataset?.splits.train ?? 0, epochs, batchN, weights === 'none'));
  /** Close mosaic as the trainer applies it: at most half the run. */
  const closeEff = $derived(errors.close || errors.epochs ? null : closeMosaicEpochs(augment.close_mosaic, epochs));
  const selArch = $derived(arches.find((a) => a.arch === arch));
  /** The sizes this server has measured end to end at 640 px (their latency replaces the reference in the table). */
  const latHere = $derived(arches.filter((a) => measuredLatency(tp.data, a.arch)));
  // A folded section must not hide its own error: while one is shown the section stays open whatever was toggled.
  const augShown = $derived(augOpen || errors.close !== null);

  const trim = (v: number) => String(+v.toFixed(3));
  /** The augmentation settings as one line; values that differ from the defaults are emphasised. */
  const augLine = $derived([
    { k: 'mosaic', v: trim(augment.mosaic), changed: augment.mosaic !== defaults.mosaic },
    { k: 'mixup', v: trim(augment.mixup), changed: augment.mixup !== defaults.mixup },
    { k: 'flip', v: trim(augment.fliplr), changed: augment.fliplr !== defaults.fliplr },
    { k: 'scale', v: `±${trim(augment.scale)}`, changed: augment.scale !== defaults.scale },
    { k: 'translate', v: `±${trim(augment.translate)}`, changed: augment.translate !== defaults.translate },
    { k: 'colour', v: trim(augment.hsv[1]), changed: augment.hsv[1] !== defaults.hsv[1] },
    { k: 'close mosaic', v: String(augment.close_mosaic), changed: augment.close_mosaic !== defaults.close_mosaic },
  ]);
  const augChanged = $derived(augLine.filter((it) => it.changed));

  const config = $derived<TrainConfig>({
    name: name.trim(),
    dataset_id: datasetId,
    arch,
    epochs,
    batch: batchN,
    imgsz: imgszN,
    device: device as Device,
    optimizer,
    lr0,
    weights: weights === 'none' ? null : weights,
    augment: { ...augment, hsv: [...augment.hsv] as [number, number, number], close_mosaic: Math.min(augment.close_mosaic, epochs) },
    seed,
    // Only to servers that report capabilities (older ones predate `amp`).
    ...(caps ? { amp: ampOn } : {}),
    // Left out for the default (missing means `o2m`, docs/API.md), so older servers see the request they know.
    ...(e2e ? { head: 'o2o' as const } : {}),
  });

  // The server's estimate (POST /api/runs/estimate, feature `run_estimate`): the first asked at once, every later one
  // 300 ms after the settings stop changing, whether or not the last one was answered (a slow or failing server is not
  // asked once per keystroke); the last answer stays up meanwhile. Its errors are the form's own (a missing device,
  // weights of another size), already said at their fields: the reference figures stand in.
  const canEstimate = $derived(hasFeature(system.data, 'run_estimate'));
  let served = $state.raw<{ key: string; est: RunEstimate } | null>(null);
  let estimating = $state(false);
  /** Whether an estimate has been asked for yet (not state: it only sets the next one's delay). */
  let asked = false;
  // The name is part of the request: the server's command (`cli`) names the run.
  const estKey = $derived(JSON.stringify(config));
  $effect(() => {
    const key = estKey;
    const body = { ...config, name: config.name || 'estimate' };
    if (!canEstimate || !dataset || !isInt(epochs, 1, 10000) || errors.close) {
      served = null;
      return;
    }
    if (untrack(() => served?.key) === key) return;
    const ctrl = new AbortController();
    estimating = true;
    const t = setTimeout(
      () => {
        asked = true;
        hub.api.estimateRun(body, ctrl.signal).then(
          (e) => {
            if (ctrl.signal.aborted) return;
            served = { key, est: e };
            estimating = false;
          },
          () => {
            if (ctrl.signal.aborted) return;
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
  /** What the Summary and the dock show: the server's estimate when there is one, else the reference's. */
  const shown = $derived.by(() => {
    const r = served?.est;
    if (r)
      return {
        lo: r.est_total_s_low,
        hi: r.est_total_s_high,
        vram: r.est_peak_vram_bytes ?? 0,
        vramMeasured: r.vram_source === 'measured',
        basis: basisLine(r),
        rough: r.confidence === 'low',
        how: confidenceLine(r),
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
      basis: refMachine ? `Reference figures, same ${onCpu ? 'CPU' : 'GPU'} model` : 'Reference figures, not measured here',
      rough: true,
      how: !canEstimate
        ? "This server does not estimate runs: docs/BENCHMARKS.md's figures, carried over to these settings."
        : estimating
          ? "Loading this server's estimate."
          : "This server could not estimate these settings: docs/BENCHMARKS.md's figures, carried over to them.",
      phases: est.phases,
      val: { images: est.valImages, rate: est.valImgsPerS, reference: false },
      startup: est.startupS,
      updated: null,
    };
  });
  // Measured peaks need no scaling: only the reference model is corrected by a failed run.
  const vram = $derived(shown.vram * (shown.vramMeasured ? 1 : scale));
  const vramPct = $derived(gpuMem ? vram / gpuMem : 0);

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
    sumOpen = false;
    submitting = true;
    try {
      const run = await hub.api.createRun(config);
      toast.success(`Started ${run.name}`, `${run.epochs} epochs of ${run.config.arch} on ${dataset?.name}.`);
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
      } else toast.error(err, 'Could not start the run');
    } finally {
      submitting = false;
    }
  }

  /** The way out of a device the server cannot use: Auto, then back to the field. */
  function useAuto() {
    device = 'auto';
    document.getElementById('device')?.focus();
  }

  async function copy(text: string, done: string) {
    try {
      await navigator.clipboard.writeText(text);
      toast.success(done);
    } catch (err) {
      toast.error(err, 'Could not copy');
    }
  }

  const pct2 = (v: number) => `${Math.round(v * 100)}%`;
  /** The standard choices plus `v` when it is not one of them (a copied run's batch 4, 1024 px), in order. */
  const withValue = (list: string[], v: string) => (list.includes(v) ? list : [...list, v].sort((a, b) => Number(a) - Number(b)));

  // ---------- review ----------
  const shortGpu = deviceName;
  const deviceLabel = $derived.by(() => {
    if (device === 'auto') return `Auto${gpu ? ` · ${shortGpu(gpu.name)}` : ''}`;
    if (device === 'cpu') return 'CPU';
    const g = system.data?.gpus.find((x) => `gpu:${x.ordinal}` === device);
    return g ? `${device} · ${shortGpu(g.name)}` : device;
  });
  type Jump = 'name' | 'data' | 'model' | 'epochs' | 'batch' | 'imgsz' | 'device' | 'amp' | 'weights' | 'opt' | 'lr0' | 'seed' | 'head' | 'close';
  /** The settings that shape the run, each a way back to its control. Optimizer and head only once not the default. */
  const review = $derived<{ key: Jump; label: string; value: string; mono?: boolean; warn?: boolean }[]>([
    { key: 'data', label: 'Data', value: dataset ? `${dataset.name} · ${compact(dataset.splits.train)} images` : '—', warn: !!errors.dataset },
    { key: 'model', label: 'Model', value: arch, mono: true },
    { key: 'epochs', label: 'Epochs', value: errors.epochs ? '—' : num(epochs), warn: !!errors.epochs },
    { key: 'batch', label: 'Batch', value: String(batchN) },
    { key: 'imgsz', label: 'Image size', value: `${imgszN} px` },
    { key: 'device', label: 'Device', value: deviceLabel, warn: !!deviceMsg },
    ...(caps ? [{ key: 'amp' as const, label: 'Mixed precision', value: ampOn ? 'On' : 'Off' }] : []),
    {
      key: 'weights',
      label: 'Start from',
      value: weights === 'none' ? 'From scratch' : (weightOptions.find((m) => m.id === weights)?.name ?? weights),
      mono: weights !== 'none',
      warn: !!errors.weights,
    },
    ...(optimizer !== 'auto' ? [{ key: 'opt' as const, label: 'Optimizer', value: optimizer === 'sgd' ? 'SGD + momentum' : 'AdamW' }] : []),
    ...(e2e ? [{ key: 'head' as const, label: 'Head', value: HEAD_LABEL.o2o }] : []),
  ]);

  /** The control a review row returns to; the picker that is on screen (cards or the phone select). */
  function jumpTarget(key: Jump): HTMLElement | null {
    const visible = (el: HTMLElement | null) => (el && el.getClientRects().length ? el : null);
    switch (key) {
      case 'data':
        return visible(document.getElementById('dataset-pick')) ?? document.querySelector<HTMLElement>('input[name="dataset"]:checked');
      case 'model':
        return visible(document.querySelector<HTMLElement>('#arch-pick [aria-checked="true"]')) ?? document.querySelector<HTMLElement>('input[name="arch"]:checked');
      case 'batch':
      case 'imgsz':
        return document.querySelector<HTMLElement>(`#${key} [aria-checked="true"]`);
      default:
        return document.getElementById(key);
    }
  }

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
  <ul class="review" aria-label="Settings">
    {#if withName}
      <li><button type="button" class="rv" class:warn={!!errors.name} onclick={() => jump('name')}><span class="rv-k">Run name</span><span class="rv-v">{name.trim() || '—'}</span></button></li>
    {/if}
    {#each review as r (r.key)}
      <li>
        <button type="button" class="rv" class:warn={r.warn} onclick={() => jump(r.key)} aria-label="{r.label}: {r.value}. Edit">
          <span class="rv-k">{r.label}</span><span class="rv-v" class:mono={r.mono}>{r.value}</span>
        </button>
      </li>
    {/each}
  </ul>
{/snippet}

{#snippet figures(where: string)}
  {#if !estReady}
    <dl class="est" aria-busy="true">
      <dt>Schedule</dt><dd class="num"><Skeleton w="88px" inline /></dd>
      <dt>Est. time</dt><dd class="num rough">Estimating…</dd>
    </dl>
  {:else}
  <dl class="est">
    <!-- The trainer's count (the short-run warning's unit). The batches behind it: a tooltip on hover and focus (which
         also describes the value to a screen reader), and read after it in browse mode. -->
    <dt>Schedule</dt>
    <dd class="num sched">
      <!-- svelte-ignore a11y_no_noninteractive_tabindex: focusable so keyboard users get the tooltip too. -->
      <span class="steps" tabindex="0" use:tooltip={stepsWhy}>{num(sched.stepsPerEpoch)} steps <span class="faint">/ epoch</span></span><span class="sr-only">. {stepsWhy}</span>
    </dd>
    <!-- A range, never one precise figure; a rough one (low confidence) in secondary ink. What it stands on is the
         line under it, which opens the phases behind it. -->
    <dt>Est. time</dt><dd class="num" class:rough={shown.rough} aria-busy={estimating || undefined}>{timeRange(shown.lo, shown.hi)}</dd>
    <dd class="basis">
      <details class="est-how">
        <summary><span>{shown.basis}</span><ChevronRight size={12} strokeWidth={2} aria-hidden="true" /></summary>
        <!-- One row each, the value at the right; on a narrow rail it wraps under its label, still at the right. -->
        <dl class="how-list num">
          {#each shown.phases as p (p.phase)}
            <div>
              <dt>{num(p.epochs)} {p.epochs === 1 ? 'epoch' : 'epochs'}, {p.phase === 'mosaic' ? 'mosaic' : 'no mosaic'}</dt>
              <dd>{num(p.epoch_s, p.epoch_s < 10 ? 1 : 0)} s each · {num(p.train_imgs_per_s)} img/s</dd>
            </div>
          {/each}
          <div>
            <dt>Validation{shown.val.reference ? ', reference' : ''}</dt>
            <dd>{num(shown.val.images)} images · {num(shown.val.rate)} img/s</dd>
          </div>
          <div><dt>First epoch</dt><dd>+{num(shown.startup, 1)} s startup</dd></div>
          {#if !onCpu && gpu}
            <div><dt>Peak GPU memory</dt><dd>{shown.vramMeasured ? 'measured here' : 'reference model'}</dd></div>
          {/if}
        </dl>
        <p class="how-note">{shown.how}{#if shown.updated}{' '}Last measured {shown.updated}.{/if}</p>
      </details>
    </dd>
    {#if !onCpu && gpu}
      <dt>Est. GPU memory</dt>
      <dd class="num" class:warn={vramPct > 0.9}>{roughBytes(vram)}{#if gpuMem}{' '}<span class="faint">/ {bytes(gpuMem, prefs.units)}</span>{/if}</dd>
    {/if}
  </dl>
  <!-- Some adapters report no memory size (`memory_bytes: 0`, e.g. through Vulkan): the estimate stands alone, said once. -->
  {#if !onCpu && gpu && !gpuMem}<p class="calib">This GPU’s memory size isn’t reported.</p>{/if}
  {#if scale > 1 && !onCpu}<p class="calib">Scaled up from the failed run's settings, which did not fit.</p>{/if}
  {#if !onCpu && gpuMem}
    <div class="vram" aria-hidden="true"><i class:warn={vramPct > 0.9} style:width={pct2(Math.min(1, vramPct))}></i></div>
  {/if}
  <!-- Warnings in the rail are one-line callouts in the status's text colour with its icon, no tinted fill (fills are
       for banners), right above Start. -->
  {#if vramPct > 1}
    <p class="callout danger" role="alert">
      <TriangleAlert size={14} aria-hidden="true" /><span>This likely runs out of GPU memory. Try batch {Math.max(1, Math.floor(batchN / Math.ceil(vramPct)))} or a smaller image size.</span>
    </p>
  {:else if onCpu}
    <p class="callout"><TriangleAlert size={14} aria-hidden="true" /><span>Training on the CPU is roughly 14× slower than on a GPU.</span></p>
  {/if}
  {/if}
  {#if short}
    <!-- The trainer's warnings (folo_vision::train::schedule_warnings) as one line; the reasons and the fix behind Why?. -->
    <div class="short">
      <p class="callout">
        <TriangleAlert size={14} aria-hidden="true" />
        <span>{#if short.scratch}Too short to learn: {num(short.steps)} steps.{:else}Mostly warmup: {num(short.warmup)} of {num(short.steps)} steps.{/if}</span>
        <button type="button" class="more" aria-expanded={shortWhy} aria-controls="{where}-short-why" onclick={() => (shortWhy = !shortWhy)}>Why?</button>
      </p>
      {#if shortWhy}
        <div class="short-why" id="{where}-short-why">
          {#if short.scratch}
            <p>
              {num(short.steps)} optimizer steps: {num(epochs)} epochs × {num(short.stepsPerEpoch)}{#if short.accumulate > 1}{' '}({batchesOf}, {short.accumulate} per step){/if}.
              Trained from scratch, a detector needs a few hundred steps to learn anything, so expect an mAP near&nbsp;0.
            </p>
          {/if}
          {#if short.warmupHeavy}
            <p>Warmup takes {num(short.warmup)} of the {num(short.steps)} steps, so the learning rate peaks only in the second half of the run.</p>
          {/if}
          <button type="button" class="fix" onclick={() => (epochs = short.epochs)}>Train {num(short.epochs)} epochs</button>
        </div>
      {/if}
    </div>
  {/if}
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

<!-- The model sizes, one specification row each (a table on wide screens; behind "Compare sizes" on phones). -->
{#snippet archTable()}
  <!-- One specification row per size, the column header on the same grid (subgrid), so every value sits
       under its heading at every width. -->
  <div class="arches" role="radiogroup" aria-label="Model size">
    <div class="a-head" aria-hidden="true"><span class="a-head-lead">Size</span><span>Params</span><span>GFLOPs</span><span>Latency</span><span></span></div>
    {#each arches as a (a.arch)}
      {@const on = a.arch === arch}
      {@const here = measuredLatency(tp.data, a.arch)}
      <label class="arch" class:on>
        <input type="radio" name="arch" value={a.arch} bind:group={arch} class="sr-only" />
        <span class="a-size">{a.size}</span>
        <Ident value={a.arch} class="a-name" />
        <span class="a-note">{a.note}</span>
        <span class="a-v num">{compact(a.params)}</span>
        <span class="a-v num">{num(a.flops, 1)}</span>
        {#if here}<span class="a-v num here">{num(here.latency_ms, 2)} ms</span>{:else if a.ms !== undefined}<span class="a-v num">{num(a.ms, 2)} ms</span>{:else}<span class="a-v faint" title="Not measured yet">—</span>{/if}
        <span class="tick" aria-hidden="true">{#if on}<Check size={14} strokeWidth={2.25} />{/if}</span>
      </label>
    {/each}
  </div>
  <!-- At most two lines: what each figure is. Measured on this server where it has predicted with that size (named in
       words, not only by weight), the architecture's reference elsewhere; never these weights' own timing. -->
  {#if latHere.length}
    <p class="ref">
      Latency at 640 px, end to end: measured on this GPU for {latHere.map((a) => a.arch).join(', ')}; the
      {latHere.length === 1 ? 'others are' : 'rest are'} the reference, {REF_WHERE}. A dash: not measured yet.
    </p>
  {:else}
    <p class="ref">Latency: each size's reference, {LATENCY_REF}. A dash: not measured yet.</p>
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
        <!-- Why the form is not a plain copy: the run it retries ran out of memory, and what was changed for that. -->
        <p class="oom-note" role="status">
          <TriangleAlert size={16} aria-hidden="true" />
          <span>
            <a class="link" href={href(`/runs/${oomRun.id}`)}>{oomRun.name}</a> ran out of GPU memory at
            {#if oomChanges.length}{oomChanges.map((x) => x.was).join(', ')}. Now {oomChanges.map((x) => x.now).join(', ')}.{:else}batch {oomRun.config.batch}, {oomRun.config.imgsz} px. These are the same settings: lower the batch or image size, or pick a smaller model.{/if}
          </span>
        </p>
      {/if}
      <div class="form">
        <Panel title="1 · Dataset" id="sec-data">
          {#if !datasets.data}
            <div class="cards">{#each [0, 1, 2] as i (i)}<Skeleton h="88px" r="var(--radius-md)" />{/each}</div>
          {:else}
            <!-- Phones: a compact picker and a one-line summary; the cards are for wider screens. -->
            <div class="ds-pick">
              <Select id="dataset-pick" label="Dataset" bind:value={datasetId} invalid={!!errors.dataset} options={datasets.data.map((d) => ({ value: d.id, label: d.name }))} />
              {#if dataset}
                <p class="ds-line num">
                  <Meta><span>{num(dataset.splits.train)} train</span><span>{num(dataset.splits.val)} val</span><span>{dataset.classes.length} classes</span><span>{dataset.format}</span></Meta>
                </p>
              {/if}
            </div>
            <div class="cards" role="radiogroup" aria-label="Dataset" aria-invalid={errors.dataset ? 'true' : undefined}>
              {#each datasets.data as d (d.id)}
                {@const on = d.id === datasetId}
                <label class="card" class:on>
                  <input type="radio" name="dataset" value={d.id} bind:group={datasetId} class="sr-only" />
                  <span class="c-top"><span class="c-name">{d.name}</span><span class="c-fmt">{d.format}</span></span>
                  {#if on}<span class="tick" aria-hidden="true"><Check size={14} strokeWidth={2.25} /></span>{/if}
                  <!-- Breaks only between the parts, and a wrapped line never ends on "·" (ui/Meta). -->
                  <span class="c-meta num"><Meta><span>{compact(d.splits.train)} train</span><span>{compact(d.splits.val)} val</span><span>{d.classes.length} classes</span></Meta></span>
                  {#if d.stats}<ClassStrip counts={d.stats.class_counts} classes={d.classes} height={4} />{/if}
                </label>
              {/each}
            </div>
          {/if}
        </Panel>

        <Panel title="2 · Model" id="sec-model" meta={was('model') ? `was ${oomRun?.config.arch}` : undefined}>
          {#if phone.current}
            <!-- Phones: the size as one segmented pick and the chosen model in words; the full comparison folds away. -->
            <div class="a-pick">
              <Segmented id="arch-pick" label="Model size" bind:value={arch} options={arches.map((a) => ({ value: a.arch, label: a.size }))} />
              {#if selArch}
                <p class="a-sel"><Ident value={selArch.arch} /> {selArch.note}</p>
                <p class="a-specs num">{compact(selArch.params)} params · {num(selArch.flops, 1)} GFLOPs</p>
                <!-- The architecture's reference figure, named as one, with where it was measured under it: not this
                     device, not these weights. -->
                {@const here = measuredLatency(tp.data, selArch.arch)}
                {#if here}
                  <!-- Measured on this server first, its spread and count beside it; the reference stays, secondary. -->
                  <p class="a-lat num">Latency <span class="a-lat-v">{latencyMs(here.latency_ms)}</span> measured here</p>
                  <p class="a-specs num">{latencyRange(here.latency_ms_low, here.latency_ms_high)} · {num(here.samples)} predictions · 640 px · {here.precision} · end to end</p>
                  {#if selArch.ms !== undefined}<p class="a-specs num">Reference {latencyMs(selArch.ms)} · {REF_WHERE}</p>{/if}
                {:else if selArch.ms !== undefined}
                  <p class="a-lat num">Reference latency <span class="a-lat-v">{num(selArch.ms, 2)} ms</span></p>
                  <p class="a-specs">{LATENCY_REF}</p>
                {:else}
                  <p class="a-lat">Reference latency not measured yet</p>
                {/if}
              {/if}
            </div>
            <details class="advanced a-compare">
              <summary><ChevronRight size={14} aria-hidden="true" /> Compare sizes <span class="faint">params, compute, latency</span></summary>
              {@render archTable()}
            </details>
          {:else}
            {@render archTable()}
          {/if}
          <div class="row2">
            <Field label="Start from" for="weights" hint="Fine-tuning from a trained model converges much faster." error={errors.weights}>
              <Select
                id="weights"
                bind:value={weights}
                invalid={!!errors.weights}
                describedby={errors.weights ? 'weights-error' : 'weights-hint'}
                options={[{ value: 'none', label: 'From scratch' }, ...weightOptions.map((m) => ({ value: m.id, label: `${m.name} · mAP ${m.metrics?.map50_95.toFixed(3) ?? '—'}` }))]}
              />
            </Field>
          </div>
        </Panel>

        <Panel title="3 · Training" id="sec-train">
          <div class="fields">
            <Field label="Epochs" for="epochs" error={errors.epochs}>
              <Input id="epochs" type="number" min="1" max="10000" step="1" bind:value={epochs} invalid={!!errors.epochs} aria-describedby={errors.epochs ? 'epochs-error' : undefined} suffix="epochs" />
            </Field>
            <Field label="Batch size" for="batch" hint="Per step. Lower it if you run out of GPU memory.">
              <!-- A retry's change sits beside the new value: "16 · was 32". -->
              <span class="with-was">
                <Segmented id="batch" label="Batch size" bind:value={batch} options={withValue(BATCHES, batch).map((v) => ({ value: v, label: v }))} />
                {#if was('batch')}<span class="was">was {oomRun?.config.batch}</span>{/if}
              </span>
            </Field>
            <Field
              label="Image size"
              for="imgsz"
              hint={!imgszChosen && datasetImgsz !== null && datasetImgsz < 640
                ? `Longer side in pixels. ${dataset?.name}'s images are all ${datasetImgsz} px.`
                : 'Longer side in pixels, letterboxed.'}
            >
              <span class="with-was">
                <Segmented id="imgsz" label="Image size" bind:value={imgsz} onchange={() => (imgszChosen = true)} options={withValue(IMGSZ, imgsz).map((v) => ({ value: v, label: v }))} />
                {#if was('imgsz')}<span class="was">was {oomRun?.config.imgsz}</span>{/if}
              </span>
            </Field>
            <!-- Not a Field: a device the server cannot use gets a line with a way out and folded details. -->
            <div class="field dev" class:invalid={!!deviceMsg}>
              <div class="dev-top">
                <label for="device">Device</label>
                {#if was('device')}<span class="was">was {oomRun?.config.device}</span>{/if}
              </div>
              <Select
                id="device"
                bind:value={device}
                invalid={!!deviceMsg}
                describedby={deviceMsg ? 'device-error' : undefined}
                options={[
                  { value: 'auto', label: `Auto${gpu ? ` · ${deviceName(gpu.name)}` : ''}` },
                  ...(system.data?.gpus ?? []).map((g) => ({ value: `gpu:${g.ordinal}`, label: `gpu:${g.ordinal} · ${deviceName(g.name)}` })),
                  // A device the run asked for that the machine no longer lists stays selectable, so the form keeps it.
                  ...(device.startsWith('gpu:') && system.data && !system.data.gpus.some((g) => `gpu:${g.ordinal}` === device) ? [{ value: device, label: `${device} · not found` }] : []),
                  { value: 'cpu', label: `CPU${system.data ? ` · ${system.data.cpu.threads} threads` : ''}` },
                ]}
              />
              {#if deviceMsg && deviceErr}
                <div class="dev-err">
                  <div class="dev-line">
                    <p id="device-error" role="alert">{deviceMsg}</p>
                    <Button size="sm" onclick={useAuto}>Use Auto</Button>
                  </div>
                  {#if deviceErr.detail}
                    <details class="dev-detail">
                      <summary><ChevronRight size={12} aria-hidden="true" />Technical details</summary>
                      <p>{deviceErr.detail}</p>
                    </details>
                  {/if}
                </div>
              {:else}
                <p class="dev-hint">Auto prefers the GPU over the CPU.</p>
              {/if}
            </div>
            {#if caps}
              <!-- A field on the same grid: the label's line carries the switch at the field's right edge, the hint (or,
                   where the server cannot train in mixed precision, why not and what to do) under it. Only the control
                   is disabled; the label keeps the field labels' colour. Hidden when the server does not say. -->
              <div class="field sw-field amp" class:off={!!ampBlock}>
                <Switch id="amp" label="Mixed precision" checked={ampOn} onchange={(v) => (amp = v)} disabled={!!ampBlock} describedby={ampBlock ? 'amp-why' : 'amp-hint'} />
                {#if ampBlock}
                  <div class="sw-why"><Remedy compact id="amp-why" cause={ampBlock.why} shell={ampBlock.shell} command={ampBlock.command} report={ampBlock.raw} /></div>
                {:else}
                  <p class="sw-hint" id="amp-hint">{ampHint}</p>
                {/if}
              </div>
            {/if}
          </div>
          <details class="advanced" open={advancedOpen} ontoggle={(e) => (advancedOpen = (e.currentTarget as HTMLDetailsElement).open)}>
            <summary><ChevronRight size={14} /> Advanced <span class="faint">optimizer, learning rate, seed, head</span></summary>
            <div class="fields">
            <Field label="Optimizer" for="opt">
              <Select id="opt" bind:value={optimizer} options={[{ value: 'auto', label: 'Auto (AdamW for short runs, else SGD)' }, { value: 'sgd', label: 'SGD + momentum' }, { value: 'adamw', label: 'AdamW' }]} />
            </Field>
            <Field label="Initial learning rate" for="lr0" error={errors.lr0} hint="Warmup 3 epochs, then cosine to 1%.">
              <Input id="lr0" type="number" step="0.001" min="0" bind:value={lr0} invalid={!!errors.lr0} aria-describedby={errors.lr0 ? 'lr0-error' : 'lr0-hint'} />
            </Field>
            <Field label="Seed" for="seed" hint="Same seed, same run." error={errors.seed}>
              <Input id="seed" type="number" min="0" step="1" bind:value={seed} invalid={!!errors.seed} aria-describedby={errors.seed ? 'seed-error' : 'seed-hint'} />
            </Field>
            <!-- TrainConfig.head `o2o` (docs/MODEL.md §6.6); the cost measured on the shapes gate (docs/BENCHMARKS.md). -->
            <div class="field sw-field">
              <Switch id="head" label="End-to-end (no NMS)" checked={e2e} onchange={(v) => (e2e = v)} describedby="head-hint" />
              <p class="sw-hint" id="head-hint">No NMS step at inference; about 0.015 lower mAP50-95 than NMS on the same model.</p>
            </div>
            </div>
          </details>
        </Panel>

        <Panel title="4 · Augmentation" id="sec-aug">
          {#snippet actions()}
            <span class="aug-actions" class:shut={!augShown}>
              <Button size="sm" variant="ghost" icon={Shuffle} onclick={() => shuffle++}>New samples</Button>
              <Button size="sm" variant="ghost" icon={RotateCcw} onclick={() => ((augment = structuredClone(defaults)), (closeEdited = false))}>Defaults</Button>
            </span>
          {/snippet}
          <button type="button" class="aug-toggle" aria-expanded={augShown} aria-controls="aug-body" onclick={() => (augOpen = !augShown)}>
            <ChevronRight size={14} aria-hidden="true" />
            <span class="aug-line num wide-only">
              {#each augLine as it, i (it.k)}{#if i > 0}<span class="sep" aria-hidden="true">·</span>{/if}<span class:changed={it.changed}>{it.k} {it.v}</span>{/each}
            </span>
            <!-- Phones: the seven settings never fit one line, so the line says what was changed, wrapping whole items. -->
            <span class="aug-line num phone-only">
              {#if augChanged.length}
                <span>{augChanged.length} changed</span>{#each augChanged as it (it.k)}<span class="sep" aria-hidden="true">·</span><span class="changed">{it.k} {it.v}</span>{/each}
              {:else}
                <span>All {augLine.length} settings at their defaults</span>
              {/if}
            </span>
            <span class="aug-verb">{augShown ? 'Hide' : 'Edit'}</span>
          </button>
          <div class="aug" id="aug-body" class:shut={!augShown}>
            <div class="aug-controls">
              {#snippet ctl(k: AugKey, children: import('svelte').Snippet)}
                <!-- svelte-ignore a11y_no_static_element_interactions -->
                <div
                  class="aug-row"
                  class:focused={focus === k}
                  onpointerenter={() => (focus = k)}
                  onpointerleave={() => (focus = null)}
                  onfocusin={() => (focus = k)}
                  onfocusout={() => (focus = null)}
                >
                  {@render children()}
                </div>
              {/snippet}
              {#snippet mosaicCtl()}<Slider label="Mosaic" bind:value={augment.mosaic} min={0} max={1} step={0.05} hint="Stitch 4 images into one: more objects per sample." />{/snippet}
              {#snippet mixupCtl()}<Slider label="Mixup" bind:value={augment.mixup} min={0} max={1} step={0.05} hint="Blend two images and keep both sets of boxes." />{/snippet}
              {#snippet flipCtl()}<Slider label="Horizontal flip" bind:value={augment.fliplr} min={0} max={1} step={0.05} hint="Chance to mirror left-right." />{/snippet}
              {#snippet scaleCtl()}<Slider label="Scale" bind:value={augment.scale} min={0} max={0.9} step={0.05} format={(v) => `±${v.toFixed(2)}`} hint="Random zoom in and out." />{/snippet}
              {#snippet translateCtl()}<Slider label="Translate" bind:value={augment.translate} min={0} max={0.5} step={0.01} format={(v) => `±${v.toFixed(2)}`} hint="Random shift, as a fraction of the size." />{/snippet}
              {#snippet hsvCtl()}<Slider label="Colour jitter (saturation)" bind:value={augment.hsv[1]} min={0} max={1} step={0.05} hint="Hue ±{augment.hsv[0]}, value ±{augment.hsv[2]}: robustness to lighting." />{/snippet}
              {@render ctl('mosaic', mosaicCtl)}
              {@render ctl('mixup', mixupCtl)}
              {@render ctl('fliplr', flipCtl)}
              {@render ctl('scale', scaleCtl)}
              {@render ctl('translate', translateCtl)}
              {@render ctl('hsv', hsvCtl)}
              <div class="aug-row">
                <Field
                  label="Close mosaic"
                  for="close"
                  hint={closeEff !== null && closeEff < augment.close_mosaic && augment.mosaic > 0
                    ? `At most half the run: mosaic closes for the last ${num(closeEff)} ${closeEff === 1 ? 'epoch' : 'epochs'}.`
                    : 'Train the last N epochs without mosaic to settle the boxes.'}
                  error={errors.close}
                >
                  <Input id="close" type="number" min="0" step="1" max={epochs} bind:value={augment.close_mosaic} oninput={() => (closeEdited = true)} suffix="epochs" invalid={!!errors.close} aria-describedby={errors.close ? 'close-error' : 'close-hint'} />
                </Field>
              </div>
            </div>
            <div class="aug-preview">
              <p class="overline">{focus ? `Preview · ${focus === 'fliplr' ? 'flip' : focus} only` : 'Preview · all augmentations'}</p>
              {#if dataset}
                <AugPreview datasetId={dataset.id} images={sampleImages.data ?? []} classes={dataset.classes} {augment} {focus} seed={shuffle} />
              {/if}
              <p class="aug-hint">Hover or focus a setting to see it alone. Boxes move with the pixels, as in training.</p>
            </div>
          </div>
        </Panel>
      </div>

      <aside class="summary">
        <Panel title="Summary">
          <!-- Two groups: what the run is (name, settings) and what it costs (estimates, Start). A rail stacks them; a
               full-width summary (below 960 px) sets them side by side. -->
          <div class="sum">
            <div class="sum-col">
              <Field label="Run name" for="name" error={errors.name}>
                <Input id="name" bind:value={name} oninput={() => (nameEdited = true)} invalid={!!errors.name} aria-describedby={errors.name ? 'name-error' : undefined} spellcheck="false" autocomplete="off" />
              </Field>
              {@render reviewList(false)}
            </div>
            <div class="sum-col">
            {@render figures('rail')}
            <div class="start" bind:this={startEl}>
              <Button
                type="submit"
                variant="primary"
                size="lg"
                icon={Play}
                block
                loading={submitting}
                disabled={!valid}
                aria-describedby={showWhy ? 'start-why' : undefined}>Start training</Button
              >
              {@render whyLine('start-why')}
            </div>
            <details class="cli">
              <summary><ChevronRight size={14} aria-hidden="true" />CLI command <span class="faint">the same run from a terminal</span></summary>
              <div class="cli-body">
                <!-- One flag a line; a line too long for the rail wraps at a space and hangs deeper than the flags. -->
                <code aria-describedby={cliPlaceholder ? 'cli-note' : undefined}
                  >{#each cliParts as p, i (i)}<span class="ln" class:cont={i > 0}
                      >{#if cliPlaceholder && p === `--weights <path to ${cliPlaceholder}>`}--weights <span class="ph">&lt;path to <span class="ph-file">{cliPlaceholder}</span>&gt;</span
                        >{:else}{p}{/if}{i < cliParts.length - 1 ? ' \\' : ''}</span
                    >{/each}</code
                >
                <span class="cli-copy"><Button size="sm" variant="ghost" icon={Copy} aria-label="Copy command" title="Copy command" onclick={() => copy(cli, 'Command copied')} /></span>
              </div>
              {#if cliPlaceholder}<p class="cli-note" id="cli-note">Put the path of {cliPlaceholder} on the server in its place: this server does not report it.</p>{/if}
            </details>
            </div>
          </div>
        </Panel>
      </aside>

      <!-- Narrow screens: the summary sits below the form, so keep the essentials and the action in reach. -->
      <!-- use:dock: the phone tab bar steps aside while this bar is on the page (lib/shell/dock.ts). -->
      <div
        use:dock
        class="dock"
        class:away={startVisible}
        role="region"
        aria-label="Start training"
        inert={startVisible}
        style:--dock-left={sidebar.collapsed ? 'var(--size-sidebar-collapsed)' : 'var(--size-sidebar)'}
      >
        <!-- The reason Start is off travels with it (the Summary's line is out of view here), in the estimate's place:
             one row of chrome, not two. -->
        {#if showWhy && why}
          {@render whyLine('dock-why', 'dock-why')}
        {:else}
          <button type="button" class="dock-est num" aria-haspopup="dialog" aria-label="Show the run summary" onclick={() => (sumOpen = true)}>
            <Meta>
              <span class:rough={shown.rough || !estReady}>{estReady ? timeRange(shown.lo, shown.hi) : 'Estimating…'}</span>
              {#if estReady && !onCpu && gpu}<span class:warn={vramPct > 0.9}>{roughBytes(vram)} GPU memory</span>{/if}
              {#if short}<span class="dock-warn"><TriangleAlert size={12} aria-hidden="true" />{short.scratch ? 'Too short to learn' : 'Mostly warmup'}</span>{/if}
              {#if dataset}<span class="faint"><Meta line><span>{dataset.name}</span><Ident value={arch} /></Meta></span>{/if}
            </Meta>
            <span class="dock-more"><ChevronUp size={14} strokeWidth={1.75} aria-hidden="true" /></span>
          </button>
        {/if}
        <Button type="submit" variant="primary" size="lg" icon={Play} loading={submitting} disabled={!valid} aria-describedby={showWhy ? 'dock-why' : undefined}
          >Start training</Button
        >
      </div>

      <!-- Inside the form, so its Start button submits it. -->
      <Dialog bind:open={sumOpen} title="Summary" size="sm">
        <div class="sum">
          {@render estimates(true)}
          {@render whyLine('sheet-why')}
        </div>
        {#snippet footer()}
          <Button variant="ghost" onclick={() => (sumOpen = false)}>Keep editing</Button>
          <Button type="submit" variant="primary" icon={Play} loading={submitting} disabled={!valid} aria-describedby={showWhy ? 'sheet-why' : undefined}>Start training</Button>
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
  .cards {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-2);
  }
  .card,
  .arch {
    position: relative;
    border-radius: var(--radius-md);
    border: 1px solid var(--line-default);
    background: var(--bg-raised);
    cursor: pointer;
    transition:
      border-color var(--dur-fast) var(--ease-standard),
      background-color var(--dur-fast) var(--ease-standard),
      box-shadow var(--dur-fast) var(--ease-standard);
  }
  .card {
    display: grid;
    gap: 6px;
    align-content: start;
    padding: var(--space-3);
  }
  .card:hover,
  .arch:hover {
    border-color: var(--line-strong);
  }
  /* Selection: a strong edge on a neutral fill and a small ink check; lime stays for the Start button and focus.
     (bg.selected is lime-tinted, for selected rows; a chosen card is a static choice, so it stays neutral.) */
  .card.on {
    border-color: var(--line-strong);
    background: var(--bg-active);
  }
  /* A model size is a row in a list: the one selection rule, bg.active and a 2 px accent bar, plus the check. */
  .arch.on {
    background: var(--bg-active);
    box-shadow: inset 2px 0 0 var(--accent-line);
  }
  .tick {
    display: grid;
    color: var(--text-primary);
  }
  .card .tick {
    position: absolute;
    top: 12px;
    right: 12px;
  }
  .card.on .c-fmt {
    visibility: hidden;
  }
  .card:has(:focus-visible),
  .arch:has(:focus-visible) {
    outline: 2px solid var(--focus);
    outline-offset: 2px;
  }
  .c-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .c-name {
    font-weight: var(--weight-medium);
  }
  .c-fmt {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .c-meta {
    font-size: var(--type-sm-size);
    color: var(--text-secondary);
  }
  /* Model sizes: single-line rows (letter, name, note, the three specs, the check) on one grid with their header.
     A subgrid's edge tracks hold its border and padding, so the first and last are 12 + 1 px wider than their content. */
  .arches {
    display: grid;
    grid-template-columns: 28px auto minmax(0, 1fr) repeat(3, 64px) 28px;
    column-gap: var(--space-3);
    row-gap: var(--space-1);
  }
  .a-head,
  .arch {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: subgrid;
    align-items: center;
  }
  .a-head {
    padding: 0 calc(var(--space-3) + 1px) var(--space-1);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    text-align: right;
  }
  .a-head-lead {
    grid-column: 1 / 4;
    text-align: left;
  }
  .arch {
    min-height: 36px;
    padding: 0 var(--space-3);
  }
  .a-size {
    font-size: var(--type-md-size);
    line-height: var(--type-md-line);
    font-weight: var(--weight-semibold);
  }
  .arch :global(.a-name) {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .arch.on :global(.a-name) {
    color: var(--text-secondary);
  }
  .a-note {
    min-width: 0;
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .a-v {
    font-size: var(--type-sm-size);
    text-align: right;
    white-space: nowrap;
  }
  .a-v.num {
    color: var(--text-secondary);
  }
  .arch.on .a-v.num {
    color: var(--text-primary);
  }
  /* Measured on this server: ink in every row (the note under the table names these sizes in words). */
  .a-v.num.here {
    color: var(--text-primary);
  }
  .ref {
    max-width: 60ch;
    text-wrap: pretty;
    margin-top: var(--space-2);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .row2 {
    margin-top: var(--space-4);
    max-width: 420px;
  }
  /* Two fields a row on one grid: a label row, a control row of one height, then help. Each field spans the three rows
     (subgrid), so labels, controls and help line up across the pair whatever each one holds. */
  .fields {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    grid-auto-rows: auto minmax(var(--control-md), auto) auto;
    gap: 0 var(--space-5);
  }
  .fields > :global(div.field) {
    grid-row: span 3;
    display: grid;
    grid-template-rows: subgrid;
    row-gap: 6px;
    align-items: center;
    padding-bottom: var(--space-4);
  }
  .fields > :global(div.field > :first-child) {
    align-self: end;
  }
  .fields > :global(div.field > p),
  .fields > :global(div.field > .dev-err) {
    align-self: start;
  }
  /* The Device field: a Field's shape, with room for its error's way out. */
  .dev-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: var(--space-2);
  }
  .dev-top label {
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  .dev-top .was {
    font-size: var(--type-sm-size);
  }
  .dev-hint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .dev-err {
    display: grid;
    gap: var(--space-1);
  }
  .dev-line {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .dev-line p {
    padding-top: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--danger-text);
  }
  .dev-line :global(.btn) {
    flex-shrink: 0;
  }
  .dev-detail summary {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
    cursor: pointer;
    list-style: none;
    border-radius: var(--radius-sm);
  }
  .dev-detail summary::-webkit-details-marker {
    display: none;
  }
  .dev-detail summary:hover {
    color: var(--text-primary);
  }
  .dev-detail summary :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .dev-detail[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .dev-detail p {
    margin-top: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    overflow-wrap: anywhere;
  }
  @media (pointer: coarse) {
    .dev-detail summary {
      min-height: 44px;
    }
  }
  .aug {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: var(--space-5);
  }
  .aug-controls {
    display: grid;
    gap: 2px;
  }
  .aug-row {
    padding: var(--space-2) var(--space-3);
    margin: 0 calc(-1 * var(--space-3));
    border-radius: var(--radius-md);
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .aug-row.focused {
    background: var(--bg-hover);
  }
  .aug-preview {
    display: grid;
    gap: var(--space-2);
    align-content: start;
    align-self: start;
    position: sticky;
    top: calc(var(--size-topbar) + var(--space-4));
  }
  .aug-hint {
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  /* Helper text keeps a 12 px measure of at most ~60 characters, however wide the column. */
  .aug-hint,
  .aug-controls :global(.hint),
  .aug-controls :global(.msg) {
    max-width: 360px;
    text-wrap: pretty;
  }
  .sum,
  .sum-col {
    display: grid;
    align-content: start;
    gap: var(--space-4);
    min-width: 0;
  }
  /* Settings to review: 28 px rows, each a way back to its control. */
  .review {
    display: grid;
    margin: calc(-1 * var(--space-1)) calc(-1 * var(--space-2)) 0;
  }
  /* Touch: each review row is a full 44 px target (after the base rule, so it wins). */
  @media (pointer: coarse) {
    .rv.rv {
      height: 44px;
    }
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
  .rv {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    width: 100%;
    height: 28px;
    padding: 0 var(--space-2);
    border-radius: var(--radius-sm);
    font-size: var(--type-sm-size);
    text-align: left;
    transition: background-color var(--dur-fast) var(--ease-standard);
  }
  .rv:hover {
    background: var(--bg-hover);
  }
  .rv:focus-visible {
    outline-offset: -2px;
  }
  .rv-k {
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .rv-v {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text-primary);
  }
  .rv.warn .rv-v {
    color: var(--danger-text);
  }
  .layout :global([data-jumped]) {
    outline: 2px solid var(--focus);
    outline-offset: 4px;
    border-radius: var(--radius-md);
  }
  .est {
    padding-top: var(--space-3);
    border-top: 1px solid var(--line-subtle);
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 6px var(--space-3);
    font-size: var(--type-sm-size);
  }
  .est dt {
    color: var(--text-tertiary);
  }
  .est dd {
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .est dd.warn {
    color: var(--warning-text);
  }
  /* A rough estimate (nothing close measured here) reads one step quieter than a measured one. */
  .est dd.rough {
    color: var(--text-secondary);
  }
  /* What the time stands on: one quiet line under it, across the row, opening the phases behind it. */
  .est dd.basis {
    grid-column: 1 / -1;
    margin-top: -2px;
    overflow: visible;
    white-space: normal;
  }
  .est-how summary {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    max-width: 100%;
    border-radius: var(--radius-xs);
    color: var(--text-tertiary);
    cursor: pointer;
    list-style: none;
  }
  .est-how summary::-webkit-details-marker {
    display: none;
  }
  .est-how summary > span {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .est-how summary :global(svg) {
    flex-shrink: 0;
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .est-how[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .est-how summary:hover {
    color: var(--text-secondary);
  }
  .how-list {
    display: grid;
    gap: 2px;
    margin-top: var(--space-2);
    padding: var(--space-2) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    text-align: left;
  }
  .how-list > div {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 0 var(--space-3);
  }
  .how-list dt {
    color: var(--text-tertiary);
  }
  .how-list dd {
    margin-left: auto;
    color: var(--text-secondary);
    text-align: right;
    white-space: nowrap;
  }
  .how-note {
    margin-top: var(--space-2);
    text-align: left;
    color: var(--text-tertiary);
    text-wrap: pretty;
  }
  @media (pointer: coarse) {
    .est-how summary {
      min-height: 44px;
    }
  }
  /* The steps carry their batches in a tooltip: focusable, so the value keeps room for its focus ring. */
  .est dd.sched {
    overflow: visible;
  }
  .steps {
    border-radius: var(--radius-xs);
    text-decoration: underline dotted var(--line-strong);
    text-underline-offset: 3px;
    cursor: help;
  }
  /* A switch field (Mixed precision, End-to-end): the Switch is the label's line, its track at the field's right edge,
     and the label reads as every field label (text.secondary, medium), also while only the control is disabled. */
  .fields > .sw-field :global(.switch .label) {
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
  }
  /* Mixed precision is the fifth field of a two-column grid: a settings row across the full width (label and hint on
     the left, the track at the section's right edge), one row of its own height, never a lone half-row beside a hole.
     Where the server cannot train in mixed precision, its remedy (ui/Remedy: a sentence, the command on one line) has
     the same full width, so the command fits whole. */
  .fields > .sw-field.amp {
    grid-column: 1 / -1;
    grid-row: auto;
    grid-template-rows: none;
  }
  .fields > .sw-field > .sw-why {
    align-self: start;
    min-width: 0;
  }
  .sw-hint {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
    text-wrap: pretty;
  }
  @media (pointer: coarse) {
    .sw-field :global(.switch) {
      min-height: 44px;
    }
  }
  /* A neutral meter; the warning colour only above 90 % of the GPU's memory. */
  .vram {
    height: 4px;
    margin-top: calc(-1 * var(--space-2));
    border-radius: 2px;
    background: var(--bg-active);
    overflow: hidden;
  }
  .vram i {
    display: block;
    height: 100%;
    background: var(--text-secondary);
    transition: width var(--dur-chart) var(--ease-standard);
  }
  .vram i.warn {
    background: var(--warning-solid);
  }
  .ds-pick {
    display: none;
  }
  /* Phones: the chosen size in words under its segmented pick, its specs in one line. */
  .a-sel {
    margin-top: var(--space-2);
    color: var(--text-primary);
  }
  .a-sel :global(.ident) {
    margin-right: var(--space-1);
    font-size: var(--type-sm-size);
    color: var(--text-tertiary);
  }
  .a-specs {
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  .a-lat {
    margin-top: var(--space-1);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .a-lat-v {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .a-compare[open] .arches {
    margin-top: var(--space-3);
  }
  /* A retry's change beside the control it changed. */
  .with-was {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .with-was .was {
    font-size: var(--type-sm-size);
    white-space: nowrap;
  }
  .ds-line {
    margin-top: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
  }
  .aug-toggle {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-height: var(--control-lg);
    margin-top: calc(-1 * var(--space-1));
    padding: 0 var(--space-2);
    margin-inline: calc(-1 * var(--space-2));
    width: calc(100% + 2 * var(--space-2));
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    text-align: left;
  }
  .aug-toggle:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }
  .aug-toggle > :global(svg) {
    flex-shrink: 0;
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .aug-toggle[aria-expanded='true'] > :global(svg) {
    transform: rotate(90deg);
  }
  .aug-line {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .aug-line .sep {
    margin-inline: var(--space-1-5);
    color: var(--text-tertiary);
  }
  .aug-line.phone-only {
    display: none;
  }
  @media (max-width: 767px) {
    .aug-line.wide-only {
      display: none;
    }
    /* Whole items wrap; none is cut. */
    .aug-line.phone-only {
      display: block;
      padding-block: var(--space-2);
      white-space: normal;
    }
    .aug-line.phone-only > span {
      white-space: nowrap;
    }
  }
  .aug-line .changed {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
  }
  .aug-verb {
    flex-shrink: 0;
    color: var(--text-tertiary);
  }
  .aug-actions {
    display: contents;
  }
  .aug.shut,
  .aug-actions.shut {
    display: none;
  }
  .aug-toggle[aria-expanded='true'] {
    margin-bottom: var(--space-3);
  }
  /* A callout: one line in the status's text colour after its icon, at 12 px, no fill (tinted fills are for banners). */
  .callout {
    display: flex;
    align-items: flex-start;
    gap: var(--space-2);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--warning-text);
    text-wrap: pretty;
  }
  .callout.danger {
    color: var(--danger-text);
  }
  .callout > :global(svg) {
    flex-shrink: 0;
    margin-top: 3px; /* grid-ok: the 14 px icon centred on the 20 px first line */
  }
  .callout > span {
    min-width: 0;
  }
  .short {
    display: grid;
    gap: var(--space-2);
  }
  /* The reasons and the fix, under the line, indented to its text. */
  .short-why {
    display: grid;
    justify-items: start;
    gap: var(--space-2);
    padding-left: 22px; /* grid-ok: the icon (14) and the gap (8) */
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-secondary);
    text-wrap: pretty;
  }
  /* "Why?" opens the reasons and the fix, right after the line's text. */
  .more {
    flex-shrink: 0;
    color: var(--warning-text);
    text-decoration: underline;
    text-decoration-color: color-mix(in srgb, currentColor 40%, transparent);
    text-underline-offset: 3px;
    border-radius: var(--radius-xs);
  }
  .more:hover {
    text-decoration-color: currentColor;
  }
  /* The fix a short-run warning suggests, under its reasons. */
  .fix {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    text-align: left;
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
    border-radius: var(--radius-xs);
  }
  .fix:hover {
    text-decoration-color: currentColor;
  }
  @media (pointer: coarse) {
    .fix,
    .more {
      min-height: 44px;
    }
  }
  .calib {
    margin-top: calc(-1 * var(--space-1));
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
    color: var(--text-tertiary);
  }
  /* The retry note spans the form and the summary; the mark carries the hue, the text stays in ink. */
  .oom-note {
    grid-column: 1 / -1;
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    background: var(--bg-panel);
    border: 1px solid var(--line-panel);
    border-radius: var(--radius-lg);
    color: var(--text-secondary);
  }
  .oom-note > :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
    color: var(--warning-text);
  }
  .oom-note .link {
    color: var(--text-primary);
    font-weight: var(--weight-medium);
    text-decoration: underline;
    text-decoration-color: var(--line-strong);
    text-underline-offset: 3px;
  }
  .oom-note .link:hover {
    text-decoration-color: var(--text-secondary);
  }
  /* A field the retry changed: its earlier value beside the label. */
  .was {
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }
  .cli {
    margin-top: calc(-1 * var(--space-1));
    padding-top: var(--space-2);
    border-top: 1px solid var(--line-subtle);
  }
  .cli summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: var(--control-md);
    color: var(--text-secondary);
    font-size: var(--type-sm-size);
    font-weight: var(--weight-medium);
    cursor: pointer;
    list-style: none;
    border-radius: var(--radius-sm);
  }
  .cli summary::-webkit-details-marker {
    display: none;
  }
  .cli summary :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .cli[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .cli summary:hover {
    color: var(--text-primary);
  }
  .cli summary .faint {
    font-weight: var(--weight-regular);
  }
  .cli-body {
    position: relative;
    margin-top: var(--space-2);
  }
  .ln {
    display: block;
    padding-left: 4ch;
    text-indent: -4ch;
  }
  .ln.cont {
    padding-left: 6ch;
  }
  /* The weights file to fill in, in placeholder ink; its name moves to the next line whole where it fits there. */
  .ph {
    color: var(--text-tertiary);
  }
  .ph-file {
    display: inline-block;
    max-width: 100%;
    text-indent: 0;
  }
  .cli-note {
    margin-top: var(--space-2);
    color: var(--text-tertiary);
    font-size: var(--type-sm-size);
    line-height: var(--type-sm-line);
  }
  .cli-copy {
    position: absolute;
    top: var(--space-1);
    right: var(--space-1);
  }
  code {
    display: block;
    padding: var(--space-2) var(--space-8) var(--space-2) var(--space-3);
    border-radius: var(--radius-md);
    background: var(--bg-inset);
    font-family: var(--font-mono);
    font-size: var(--type-sm-size);
    line-height: 18px;
    color: var(--text-secondary);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .advanced {
    margin-top: var(--space-4);
    border-top: 1px solid var(--line-subtle);
    padding-top: var(--space-3);
  }
  .advanced summary {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    height: var(--control-md);
    color: var(--text-secondary);
    font-weight: var(--weight-medium);
    cursor: pointer;
    list-style: none;
    border-radius: var(--radius-sm);
  }
  .advanced summary::-webkit-details-marker {
    display: none;
  }
  .advanced summary :global(svg) {
    transition: transform var(--dur-fast) var(--ease-standard);
  }
  .advanced[open] summary :global(svg) {
    transform: rotate(90deg);
  }
  .advanced summary:hover {
    color: var(--text-primary);
  }
  .advanced summary .faint {
    font-weight: var(--weight-regular);
  }
  .advanced[open] .fields {
    margin-top: var(--space-3);
  }
  @media (pointer: coarse) {
    .advanced summary {
      height: 44px;
    }
  }
  .dock {
    display: none;
  }
  @media (max-width: 1279px) {
    /* Laptops keep the sticky rail beside the form, 40 px slimmer. */
    .layout {
      grid-template-columns: minmax(0, 1fr) 300px;
    }
    /* The preview takes a little more of the narrower form, so its frames stay large enough to read. */
    .aug {
      grid-template-columns: minmax(0, 5fr) minmax(0, 6fr);
      gap: var(--space-4);
    }
  }
  @media (max-width: 1023px) {
    .layout {
      padding-bottom: 64px;
    }
    /* Beside the sidebar (its content may start at the left edge, so the bar starts where the page does). */
    .dock {
      position: fixed;
      left: var(--dock-left, 0);
      right: 0;
      bottom: 0;
      z-index: var(--z-sticky);
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: var(--space-4);
      padding: var(--space-3) var(--space-4);
      background: color-mix(in srgb, var(--bg-app) 90%, transparent);
      backdrop-filter: blur(12px) saturate(1.4);
      -webkit-backdrop-filter: blur(12px) saturate(1.4);
      border-top: 1px solid var(--line-subtle);
      transition:
        transform var(--dur-base) var(--ease-out),
        opacity var(--dur-base) var(--ease-out);
    }
    /* The reason sits on the dock's left, where the estimate would be. */
    .dock-why {
      flex: 1;
      min-width: 0;
      margin-top: 0;
    }
    .dock.away {
      transform: translateY(100%);
      opacity: 0;
    }
    .dock-est {
      display: flex;
      align-items: center;
      gap: var(--space-3);
      min-height: var(--control-lg);
      padding: 0 var(--space-2);
      border-radius: var(--radius-md);
      color: var(--text-secondary);
      font-size: var(--type-sm-size);
      text-align: left;
    }
    .dock-est:hover {
      background: var(--bg-hover);
      color: var(--text-primary);
    }
    .dock-more {
      display: grid;
      color: var(--text-tertiary);
    }
    .dock-est .warn,
    .dock-warn {
      color: var(--warning-text);
    }
    .dock-warn {
      display: inline-flex;
      align-items: center;
      gap: var(--space-1);
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
    .summary .est {
      padding-top: 0;
      border-top: 0;
    }
  }
  .dock-est .faint {
    display: none;
  }
  @media (max-width: 767px) {
    /* Two columns: the estimate wraps on the left, a 44 px Start button holds the right; 16 px gutters. */
    .dock {
      left: 0;
      bottom: calc(var(--tabbar-h) + env(safe-area-inset-bottom));
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: var(--space-3);
      padding: var(--space-2) var(--space-4);
    }
    .dock-est {
      position: relative;
      flex-wrap: wrap;
      gap: 0 var(--space-3);
      min-width: 0;
      margin-left: calc(-1 * var(--space-2));
      padding: var(--space-1) var(--space-6) var(--space-1) var(--space-2);
      line-height: var(--type-sm-line);
    }
    .dock-more {
      position: absolute;
      right: var(--space-2);
      top: 50%;
      transform: translateY(-50%);
    }
    .dock-est .faint {
      display: block;
      flex-basis: 100%;
      font-size: var(--type-sm-size);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .dock :global(.btn.lg) {
      --h: 44px;
    }
    .ds-pick {
      display: block;
    }
    .ds-pick :global(select) {
      height: 44px;
    }
    .cards {
      display: none;
    }
    /* Model rows: the note gives way (it sits under the rows) so the three specs keep their columns; 44 px rows. */
    .arches {
      grid-template-columns: 28px minmax(0, 1fr) repeat(3, 52px) 28px;
      column-gap: var(--space-2);
    }
    .a-head-lead {
      grid-column: 1 / 3;
    }
    .arch {
      min-height: 44px;
    }
    .a-note {
      display: none;
    }
    .summary .sum {
      grid-template-columns: minmax(0, 1fr);
    }
    .summary .est {
      padding-top: var(--space-3);
      border-top: 1px solid var(--line-subtle);
    }
    .fields,
    .aug {
      grid-template-columns: minmax(0, 1fr);
    }
    /* One column: no neighbour to line up with, so a switch field is one row of its own height. */
    .fields > .sw-field {
      grid-row: auto;
      grid-template-rows: none;
    }

    .aug-preview {
      position: static;
      order: -1;
    }
  }
</style>
