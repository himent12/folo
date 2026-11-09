// In-browser mock of the whole API (docs/API.md), used with `?mock=1` or when the server is unreachable.
// Deterministic: everything derives from ids and the clock, so screenshots are comparable between rounds.
//
// Scenarios (`?scenario=`): `normal` (default), `empty` (nothing registered yet), `error` (every call fails),
// `loading` (calls never resolve), `oom` (server inference runs out of GPU memory for every model but folo-n),
// `gpu-unplugged` (a second GPU is listed until a run asks for it, then `404 device_not_found`), `short-run` (adds a
// finished 5-epoch run the trainer warns is too short to learn from), `tensor-cores-off` and `no-tensor-cores` (no
// mixed-precision training: the server's `FOLO_MOCK_TENSOR_CORES=off` / `none`), `train-split` (adds a finished run on a dataset with no validation images), `finishing` (the run training reaches its
// last epoch 3 s after load, for the toast that says so), `no-throughput` (a server from before measured throughput:
// no `throughput`, `run_estimate`, `benchmark` or `run_cli` feature, and those endpoints answer 404), `stale` (the server
// stops answering, its live stream too, 3 s after load), `stale-lists` (the requests stop, the live stream keeps streaming) and `amp-refused`
// (the server restarted without tensor cores after the page loaded: `/api/system` said mixed precision was available,
// `POST /api/runs` refuses it with `501 not_supported`, and `/api/system` says so from then on), and for cloud compute
// (docs/API.md, Mock mode → Cloud: one RunPod connection, a finished cloud run): `cloud-none` (no connection, the
// server's `FOLO_MOCK_CLOUD=none`), `cloud-off` (no `cloud` feature: `FOLO_MOCK_CLOUD=off`), `cloud-trouble` (the
// connection's key was revoked, so its check fails and the termination of a finished run's instance cannot be
// confirmed), `cloud-training` (a cloud run trains on a rented RTX 4090) and `cloud-upload` (a cloud run uploads its
// dataset), for screenshots of those states.

import { ApiError, type Api, type RunEventHandlers } from '../client';
import type {
  Arch,
  BenchmarkReport,
  BenchmarkRequest,
  Box,
  Capabilities,
  ClassMetrics,
  CliCommand,
  CloudConnection,
  CloudInstance,
  CloudOffer,
  CloudOfferList,
  CloudProvider,
  ConnectCloud,
  CreateDataset,
  CreateSyntheticDataset,
  Dataset,
  DatasetImages,
  DatasetImagesQuery,
  DatasetIssues,
  DatasetIssuesQuery,
  ExportFormat,
  ExportResult,
  GpuPool,
  Health,
  IssueCounts,
  LogLine,
  Model,
  ModelEval,
  OnnxSwitches,
  PredictBox,
  PredictInput,
  PredictOptions,
  Prediction,
  QueueRequest,
  Run,
  RunCloud,
  RunLogs,
  RunEstimate,
  RunMetrics,
  RunSamples,
  Sample,
  Split,
  SystemInfo,
  ThroughputTable,
  TrainConfig,
} from '../types';
import { image, imageIndex, indices, issues, specs, stats, toDataset, total, type MockDatasetSpec } from './datasets';
import { synthEvaluation, toModelEval } from './evaluation';
import { renderImage } from './render';
import { hash, rng, smoothNoise, range } from './rng';
import {
  GPU_NAME,
  archInfo,
  bestOf,
  epochAt,
  cloudFinish,
  cloudOf,
  epochLog,
  history,
  loopStart,
  optimizerSteps,
  progressAt,
  resolveTrain,
  startLogs,
  stepAt,
  toRun,
  trainEnd,
  type RunProfile,
} from './runs';
import { COCO80 } from './coco';
import {
  cloudLog,
  instanceAt,
  PROVIDERS,
  REMOTE_OFFER,
  RTX_4090,
  RUNPOD_OFFERS,
  secretHint,
  seedConnection,
  trainAt,
  type CloudTimeline,
} from './cloud';
import { applyQueueMove } from '../queue';
import { onnxFileName } from '../client';
import { CPU_NAME, MockThroughput } from './throughput';
import { trainCommand } from '../../trainCli';

export type Scenario =
  | 'normal'
  | 'empty'
  | 'error'
  | 'loading'
  | 'oom'
  | 'gpu-unplugged'
  | 'short-run'
  | 'tensor-cores-off'
  | 'no-tensor-cores'
  | 'finishing'
  // A finished run on fasteners-mini, which has no validation images: it validated on its training split.
  | 'train-split'
  | 'no-throughput'
  | 'amp-refused'
  // The server answers the first loads, then stops (STALE_AFTER): pages keep what they have and say it is stale, and
  // the live run's stream drops too, so its card shows the last known state.
  | 'stale'
  // The same, but only the requests stop answering: the live run's stream keeps streaming, so its card stays live
  // and the page says the lists alone are not refreshing.
  | 'stale-lists'
  // A Studio build without folo-wasm: /api/system does not list `web_wasm`, so Playground's browser choices are off.
  | 'no-wasm'
  // Cloud compute: no connection; no `cloud` feature; a revoked key and an unconfirmed termination; a cloud run
  // training; one uploading its dataset.
  | 'cloud-none'
  | 'cloud-off'
  | 'cloud-trouble'
  | 'cloud-training'
  | 'cloud-upload';

const MIN = 60_000;
const HOUR = 60 * MIN;
const DAY = 24 * HOUR;
/** The `stale` scenario: how long the server answers before it stops. */
const STALE_AFTER = 3000;

const defaultAugment = { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4] as [number, number, number], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 };

function config(name: string, dataset_id: string, arch: Arch, epochs: number, extra: Partial<TrainConfig> = {}): TrainConfig {
  return {
    name,
    dataset_id,
    arch,
    epochs,
    batch: 16,
    imgsz: 640,
    device: 'gpu:0',
    optimizer: 'auto',
    lr0: 0.01,
    weights: null,
    augment: { ...defaultAugment },
    seed: 0,
    ...extra,
  };
}

const DEVICE = `${GPU_NAME} (vulkan)`;

/** folo_gpu::context::TENSOR_CORES_NOT_REQUESTED. */
const TC_NOT_REQUESTED = 'off (not requested: FOLO_GPU_TENSOR_CORES=1 or --tensor-cores)';

/**
 * `/api/system`'s `capabilities` as folo-server's mock reports them (`MockTensorCores`), worded as the real server's:
 * tensor cores on and `amp` available; `tensor-cores-off`, available but not requested; `no-tensor-cores`, the adapter
 * through DX12, where wgpu has no cooperative matrices.
 */
export function capabilities(scenario: Scenario): Capabilities {
  const status = scenario === 'tensor-cores-off' ? TC_NOT_REQUESTED : scenario === 'no-tensor-cores' ? `off: ${GPU_NAME} (Dx12) has no cooperative matrices in wgpu` : null;
  return {
    device: 'gpu:0',
    shader_f16: true,
    tensor_cores: { available: scenario !== 'no-tensor-cores', enabled: status === null, reason: status },
    amp:
      status === null
        ? { available: true, reason: null }
        : {
            available: false,
            // folo_vision::train::amp_not_supported
            reason:
              `mixed-precision training (amp) needs a GPU with tensor cores (f16 cooperative matrices): gpu:0 has tensor cores ${status}; ` +
              'train without amp (f32), or on such a GPU (`folo train --amp` asks for them; start `folo serve` or `folo mcp` with `--tensor-cores`)',
          },
  };
}
/** folo-server's hint on `501 not_supported` (engine::engine_error), verbatim. */
const AMP_HINT =
  'set `amp: false` to train in f32, or restart this server with tensor cores (`folo serve --tensor-cores`, or ' +
  "FOLO_GPU_TENSOR_CORES=1) on a GPU that has them; GET /api/system's `capabilities.amp` says when amp is available";

/** What `folo weights list` says of folo-n-coco (folo_api::weights::PUBLISHED), its `Model.about`. */
const COCO_ABOUT = 'folo-n trained from scratch on COCO train2017 at 640 px (EMA weights, f32, unfused: fine-tune or predict); examples/coco';

const FASTENER_EASE = [1.05, 1.1, 1.12, 0.92, 0.78, 1.0, 0.85, 0.7];

function profiles(t0: number, scenario: Scenario = 'normal'): RunProfile[] {
  const live = t0 - 222_000;
  const names = (id: string) => [...(specs.find((s) => s.id === id)?.classes ?? [])];
  const list: Omit<RunProfile, 'classNames'>[] = [
    {
      id: 'run_q4n2',
      name: 'shapes-hard-l',
      config: config('shapes-hard-l', 'ds_shapes_hard', 'folo-l', 120, { batch: 32 }),
      classes: 6,
      classEase: [1.1, 1.05, 1, 0.82, 0.95, 0.75],
      stepsPerEpoch: 125,
      epochSeconds: 30,
      mapMax: 0.7,
      tau: 16,
      imgsPerS: 140,
      createdAt: t0 - 40_000,
      startedAt: null,
      device: DEVICE,
    },
    {
      // Second in the training queue, behind shapes-hard-l: moving it means something.
      id: 'run_q7c5',
      name: 'fasteners-m',
      config: config('fasteners-m', 'ds_fasteners', 'folo-m', 80),
      classes: 8,
      classEase: FASTENER_EASE,
      stepsPerEpoch: 120,
      epochSeconds: 14,
      mapMax: 0.69,
      tau: 15,
      imgsPerS: 170,
      createdAt: t0 - 25_000,
      startedAt: null,
      device: DEVICE,
    },
    {
      id: 'run_7d1e',
      name: 'fasteners-s',
      config: config('fasteners-s', 'ds_fasteners', 'folo-s', 100, { weights: 'm_coco_s' }),
      classes: 8,
      classEase: FASTENER_EASE,
      stepsPerEpoch: 120,
      epochSeconds: 6,
      mapMax: 0.66,
      tau: 14,
      imgsPerS: 318,
      createdAt: live - 5_000,
      startedAt: live,
      loop: true,
      modelId: 'm_fast_s',
      device: DEVICE,
    },
    {
      id: 'run_c19a',
      name: 'fasteners-n',
      // Fine-tuned from shapes-n (the Model page's lineage line), in mixed precision (as the server mock's m_53d1).
      config: config('fasteners-n', 'ds_fasteners', 'folo-n', 150, { weights: 'm_shapes_n', amp: true }),
      classes: 8,
      classEase: FASTENER_EASE,
      stepsPerEpoch: 120,
      epochSeconds: 21.4,
      mapMax: 0.58,
      tau: 18,
      imgsPerS: 512,
      createdAt: t0 - DAY - 2 * HOUR,
      startedAt: t0 - DAY - 2 * HOUR + 4_000,
      terminal: { status: 'completed', atEpoch: 150 },
      modelId: 'm_fast_n',
      device: DEVICE,
    },
    {
      // A short fine-tune (DOGFOOD-3 S2, S4): 15 epochs ask for the default close mosaic 10, which the trainer caps at
      // 7 (half the run); in mixed precision with the end-to-end head, so the run's header says both.
      id: 'run_f15c',
      name: 'fasteners-n-ft',
      config: config('fasteners-n-ft', 'ds_fasteners', 'folo-n', 15, { weights: 'm_fast_n', amp: true, head: 'o2o' }),
      classes: 8,
      classEase: FASTENER_EASE,
      stepsPerEpoch: 120,
      epochSeconds: 21.4,
      mapMax: 0.6,
      tau: 4,
      imgsPerS: 90,
      createdAt: t0 - DAY - 4 * HOUR,
      startedAt: t0 - DAY - 4 * HOUR + 3_000,
      terminal: { status: 'completed', atEpoch: 15 },
      device: DEVICE,
    },
    {
      id: 'run_a2f0',
      name: 'shapes-hard-m',
      config: config('shapes-hard-m', 'ds_shapes_hard', 'folo-m', 100, { batch: 32, imgsz: 960, augment: { ...defaultAugment, close_mosaic: 15 } }),
      classes: 6,
      classEase: [1.1, 1.05, 1, 0.82, 0.95, 0.75],
      stepsPerEpoch: 125,
      epochSeconds: 41,
      mapMax: 0.68,
      tau: 15,
      imgsPerS: 98,
      createdAt: t0 - DAY - 6 * HOUR,
      startedAt: t0 - DAY - 6 * HOUR + 3_000,
      terminal: {
        status: 'failed',
        atEpoch: 23.4,
        // As folo-vision's `Error::summary` words it: the first line of the driver's report in parentheses.
        error: 'the GPU ran out of memory (wgpu backend: Out of Memory)',
      },
      device: DEVICE,
    },
    {
      id: 'run_5be3',
      name: 'shapes-hard-s',
      // Fine-tuned from shapes-s: Compare's Start from row and legend show it next to the runs from scratch.
      config: config('shapes-hard-s', 'ds_shapes_hard', 'folo-s', 100, { weights: 'm_shapes_s' }),
      classes: 6,
      classEase: [1.1, 1.05, 1, 0.82, 0.95, 0.75],
      stepsPerEpoch: 250,
      epochSeconds: 19,
      mapMax: 0.61,
      tau: 13,
      imgsPerS: 330,
      createdAt: t0 - 2 * DAY - 3 * HOUR,
      startedAt: t0 - 2 * DAY - 3 * HOUR + 2_000,
      terminal: { status: 'stopped', atEpoch: 61 },
      modelId: 'm_hard_s',
      device: DEVICE,
    },
    {
      id: 'run_0e77',
      name: 'shapes-s',
      // The NMS-free one-to-one head (as the server mock's m_31ab, also shapes-s-best).
      config: config('shapes-s', 'ds_shapes', 'folo-s', 100, { head: 'o2o' }),
      classes: 3,
      classEase: [1.04, 1.0, 0.95],
      stepsPerEpoch: 50,
      epochSeconds: 8.2,
      mapMax: 0.83,
      tau: 9,
      imgsPerS: 335,
      createdAt: t0 - 11 * DAY,
      startedAt: t0 - 11 * DAY + 2_000,
      terminal: { status: 'completed', atEpoch: 100 },
      modelId: 'm_shapes_s',
      device: DEVICE,
    },
    {
      id: 'run_31aa',
      name: 'shapes-n',
      config: config('shapes-n', 'ds_shapes', 'folo-n', 100),
      classes: 3,
      classEase: [1.04, 1.0, 0.95],
      stepsPerEpoch: 50,
      epochSeconds: 5.1,
      mapMax: 0.77,
      tau: 10,
      imgsPerS: 540,
      createdAt: t0 - 12 * DAY,
      startedAt: t0 - 12 * DAY + 2_000,
      terminal: { status: 'completed', atEpoch: 100 },
      modelId: 'm_shapes_n',
      device: 'cpu (AMD Ryzen 9 7950X)',
    },
    // `short-run`: 800 images at batch 32 are 25 batches an epoch, 2 accumulated per optimizer step (docs/MODEL.md
    // §8.3), so 13 steps an epoch and 65 in 5 epochs: from scratch, the trainer warns it is too short (and its warmup
    // too long). New run's `run-new-short` scene asks for the same run and suggests the same 16 epochs.
    ...(scenario === 'short-run'
      ? [
          {
            id: 'run_s5q1',
            name: 'shapes-quick',
            config: config('shapes-quick', 'ds_shapes', 'folo-n', 5, { batch: 32 }),
            classes: 3,
            classEase: [1.04, 1.0, 0.95],
            stepsPerEpoch: optimizerSteps(800, 32),
            epochSeconds: 6,
            mapMax: 0.3,
            tau: 40,
            imgsPerS: 540,
            createdAt: t0 - 2 * HOUR,
            startedAt: t0 - 2 * HOUR + 2_000,
            terminal: { status: 'completed' as const, atEpoch: 5 },
            device: DEVICE,
          },
        ]
      : []),
  ];
  // `finishing`: the run in the training slot completes 3 s from now. It is a run of its own, shapes-hard-n, not the
  // demo's live fasteners-s (epoch 38 of 100 on every other screen), so the toast that says it finished, with its real
  // best epoch, contradicts nothing another shot shows; fasteners-s is not in this scenario.
  const finishing: Omit<RunProfile, 'classNames'> = {
    id: 'run_e6b9',
    name: 'shapes-hard-n',
    config: config('shapes-hard-n', 'ds_shapes_hard', 'folo-n', 60),
    classes: 6,
    classEase: [1.1, 1.05, 1, 0.82, 0.95, 0.75],
    stepsPerEpoch: 250,
    epochSeconds: 6,
    mapMax: 0.52,
    tau: 12,
    imgsPerS: 520,
    createdAt: t0 - 60 * 6000 - 5000,
    startedAt: t0 - 60 * 6000 + 3000,
    device: DEVICE,
  };
  // `train-split`: a fine-tune on fasteners-mini (24 training images, none for validation), so every epoch's scores are
  // on its training split; the run page says so and names that split (the `run-train-split` scene).
  const trainSplit: Omit<RunProfile, 'classNames'> = {
    id: 'run_m1n1',
    name: 'fasteners-mini-n',
    config: config('fasteners-mini-n', 'ds_fasteners_mini', 'folo-n', 60, { weights: 'm_fast_n' }),
    classes: 2,
    classEase: [1.02, 0.97],
    stepsPerEpoch: optimizerSteps(24, 16),
    epochSeconds: 2.5,
    mapMax: 0.74,
    tau: 9,
    imgsPerS: 480,
    createdAt: t0 - 3 * HOUR,
    startedAt: t0 - 3 * HOUR + 2_000,
    terminal: { status: 'completed', atEpoch: 60 },
    device: DEVICE,
    // Its best weights, scored on the training split: Overview and Models tag the score and rank it apart.
    modelId: 'm_mini_run',
  };
  const all = [
    ...(scenario === 'finishing' ? [...list.filter((p) => p.id !== 'run_7d1e'), finishing] : scenario === 'train-split' ? [...list, trainSplit] : list),
    ...cloudRuns(t0, scenario),
  ];
  return all.map((p) => withResolved({ ...p, classNames: names(p.config.dataset_id) }));
}

/** A cloud run's timeline as the seeded runs have it: a secure-cloud RTX 4090 on the demo's connection. */
function rented(id: string, nonce: string, ref: string, launchedAt: number, ms: { provision: number; boot: number; sync: number }, upload: number, weights: number, maxHours: number): CloudTimeline {
  return {
    connectionId: 'cc_4d2a',
    provider: 'runpod',
    offer: RTX_4090,
    maxHours,
    instanceId: `ci_${nonce}`,
    instanceName: `folo-${id}-${nonce}`,
    providerRef: ref,
    launchedAt,
    provisionMs: ms.provision,
    bootMs: ms.boot,
    syncMs: ms.sync,
    uploadBytes: upload,
    weightsBytes: weights,
  };
}

/**
 * The seeded cloud runs (docs/API.md, Mock mode → Cloud): shapes-m, finished on a rented RTX 4090 two days ago (in
 * `cloud-trouble` two hours ago, its instance's termination never confirmed); `cloud-training` adds fasteners-l
 * training there, `cloud-upload` shapes-hard-x uploading its dataset.
 */
function cloudRuns(t0: number, scenario: Scenario): Omit<RunProfile, 'classNames'>[] {
  // Without the feature, or before any connection was made (`cloud-none`: a cloud never used), there are none.
  if (scenario === 'cloud-off' || scenario === 'cloud-none' || scenario === 'no-throughput' || scenario === 'empty') return [];
  const compute = (hours: number) => ({ device: 'auto' as const, compute: { connection_id: 'cc_4d2a', offer_id: RTX_4090.id, max_hours: hours } });
  const GPU_4090 = 'NVIDIA GeForce RTX 4090 (vulkan)';
  const MiB = 2 ** 20;
  const finishedAt = scenario === 'cloud-trouble' ? t0 - 2 * HOUR : t0 - 2 * DAY - 5 * HOUR;
  const done = rented('run_c3d5', '4c1a', 'k2x9wq7m3fz1', finishedAt, { provision: 40_000, boot: 3 * MIN, sync: 70_000 }, 64 * MiB, 330 * MiB, 2);
  if (scenario === 'cloud-trouble') done.stuck = 'RunPod rejected the API key (401 Unauthorized)';
  const finished: Omit<RunProfile, 'classNames'> = {
    id: 'run_c3d5',
    name: 'shapes-m',
    config: config('shapes-m', 'ds_shapes', 'folo-m', 100, compute(2)),
    classes: 3,
    classEase: [1.04, 1.0, 0.95],
    stepsPerEpoch: optimizerSteps(800, 16),
    epochSeconds: 9.4,
    mapMax: 0.8,
    tau: 9,
    imgsPerS: 410,
    createdAt: finishedAt,
    startedAt: trainAt(done),
    modelId: 'm_shapes_m',
    device: GPU_4090,
    cloud: done,
  };
  finished.terminal = { status: 'completed', atEpoch: 100, at: cloudFinish({ ...finished, cloud: done }) };
  const out = [finished];
  if (scenario === 'cloud-training') {
    const c = rented('run_c7b1', '9d03', 'p7m2v8x4qk1c', t0 - 40 * MIN, { provision: 40_000, boot: 3 * MIN, sync: 140_000 }, 515 * MiB, 410 * MiB, 3);
    out.push({
      id: 'run_c7b1',
      name: 'fasteners-l',
      config: config('fasteners-l', 'ds_fasteners', 'folo-l', 100, compute(3)),
      classes: 8,
      classEase: FASTENER_EASE,
      stepsPerEpoch: optimizerSteps(1920, 16),
      epochSeconds: 60,
      mapMax: 0.7,
      tau: 16,
      imgsPerS: 160,
      createdAt: c.launchedAt,
      startedAt: trainAt(c),
      device: GPU_4090,
      cloud: c,
    });
  }
  if (scenario === 'cloud-upload') {
    // A quarter of the upload is through at load: 90 s of six minutes.
    const c = rented('run_c9e4', '2b7e', 'q4w8e2r6t1y5', t0 - 3 * MIN, { provision: 30_000, boot: 60_000, sync: 6 * MIN }, 460 * MiB, 1100 * MiB, 8);
    out.push({
      id: 'run_c9e4',
      name: 'shapes-hard-x',
      config: config('shapes-hard-x', 'ds_shapes_hard', 'folo-x', 100, compute(8)),
      classes: 6,
      classEase: [1.1, 1.05, 1, 0.82, 0.95, 0.75],
      stepsPerEpoch: optimizerSteps(4000, 16),
      epochSeconds: 45,
      mapMax: 0.72,
      tau: 14,
      imgsPerS: 120,
      createdAt: c.launchedAt,
      startedAt: trainAt(c),
      device: GPU_4090,
      cloud: c,
    });
  }
  return out;
}

/** The profile with `resolved`: what the real trainer resolves its config to on its dataset's training split. */
function withResolved(p: RunProfile, train?: number): RunProfile {
  const images = train ?? specs.find((s) => s.id === p.config.dataset_id)?.splits.train ?? 0;
  return { ...p, resolved: resolveTrain(p.config, images, p.classes, p.stepsPerEpoch) };
}

/** A prediction before it gets its class name. */
type Scored = Box & { score: number };

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export class MockApi implements Api {
  readonly mock = true;
  readonly #t0: number;
  #runs: RunProfile[];
  #datasets: MockDatasetSpec[];
  #extraModels: Model[] = [];
  #images = new Map<string, Promise<string>>();
  #nextId = 1;
  /**
   * The training queue: ids of the runs waiting for the one training slot, front first (docs/API.md, POST
   * /api/runs/{id}/queue). Runs train one at a time; the worker takes the front run once the slot is free.
   */
  #queue: string[];
  /** Live subscribers per run, for log lines of actions outside a run's timeline (queue moves). */
  #subs = new Map<string, Set<RunEventHandlers>>();
  /** Those log lines, kept for `/logs`. */
  #notes = new Map<string, LogLine[]>();
  #throughput: MockThroughput;
  /** Runs created through the API: while one is queued or running, a benchmark probe answers `409 run_active`. */
  #created = new Set<string>();
  #probing = false;
  /** Cloud connections. A secret is never kept: only its hint (`••••a1b2`), as the API answers it. */
  #connections: CloudConnection[];
  /** Whether the server has the `cloud` feature (`/api/cloud`, `TrainConfig.compute`); older servers do not. */
  readonly #cloudOn: boolean;

  constructor(
    readonly scenario: Scenario = 'normal',
    readonly latency = 140,
  ) {
    this.#t0 = Date.now();
    this.#runs = scenario === 'empty' ? [] : profiles(this.#t0, scenario);
    this.#datasets = scenario === 'empty' ? [] : [...specs];
    this.#throughput = new MockThroughput(this.#t0);
    this.#cloudOn = !['cloud-off', 'no-throughput'].includes(scenario);
    this.#connections = this.#cloudOn && !['empty', 'cloud-none'].includes(scenario) ? [seedConnection(this.#t0, scenario === 'cloud-trouble')] : [];
    this.#queue = this.#runs
      .filter((p) => p.startedAt === null && !p.terminal)
      .sort((a, b) => a.createdAt - b.createdAt)
      .map((p) => p.id);
  }

  /**
   * The worker: once no run holds the training slot (running, stopping, or taken and still being built), it takes
   * the front of the queue, which starts 1.5 s later (at the earliest 1.5 s after it was created).
   */
  #schedule(now: number) {
    if (!this.#queue.length) return;
    // Cloud runs train elsewhere: they never hold the slot.
    const busy = this.#runs.some(
      (p) => !p.cloud && !this.#queue.includes(p.id) && ((p.startedAt !== null && p.startedAt > now) || ['running', 'stopping'].includes(progressAt(p, now).status)),
    );
    if (busy) return;
    const p = this.#runs.find((r) => r.id === this.#queue[0]);
    this.#queue.shift();
    if (p) p.startedAt = Math.max(now + 1500, p.createdAt + 1500);
  }

  /** A run snapshot with its place in the queue. */
  #toRun(p: RunProfile, now: number): Run {
    const i = this.#queue.indexOf(p.id);
    return { ...toRun(p, now), queue_position: i < 0 ? null : i + 1 };
  }

  /** Logs `message` for run `id`: kept for `/logs` and sent to its live subscribers. */
  #note(id: string, message: string) {
    const line: LogLine = { t: new Date().toISOString(), level: 'info', message };
    this.#notes.set(id, [...(this.#notes.get(id) ?? []), line]);
    for (const h of this.#subs.get(id) ?? []) h.log?.(line);
  }

  /** Runs `fn` after a simulated network delay (`latency` overrides the default; a mock built with 0 never waits). */
  async #call<T>(fn: () => T, latency?: number): Promise<T> {
    if (this.scenario === 'loading') return new Promise<T>(() => {});
    if (this.latency > 0 && (latency ?? this.latency) > 0) await sleep((latency ?? this.latency) * (0.7 + 0.6 * Math.random()));
    if (this.scenario === 'error' || ((this.scenario === 'stale' || this.scenario === 'stale-lists') && Date.now() - this.#t0 > STALE_AFTER))
      throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
    this.#schedule(Date.now());
    return fn();
  }

  #spec(id: string): MockDatasetSpec {
    const s = this.#datasets.find((d) => d.id === id);
    if (!s) throw new ApiError(404, 'dataset_not_found', `No dataset with id ${id}.`, 'List datasets with GET /api/datasets.');
    return s;
  }

  #profile(id: string): RunProfile {
    const p = this.#runs.find((r) => r.id === id);
    if (!p) throw new ApiError(404, 'run_not_found', `No run with id ${id}.`);
    return p;
  }

  // ---------- system ----------

  health(): Promise<Health> {
    return this.#call(() => ({ status: 'ok' as const, version: '0.1.0', api: 'v1', mock: true }), 20);
  }

  /** `gpu-unplugged`: the second adapter is gone once a run asked for it. */
  #unplugged = false;

  /** Whether a run trains on this machine now (cloud runs use another's GPU). */
  #trainingHere(): boolean {
    return this.#runs.some((p) => !p.cloud && progressAt(p, Date.now()).status === 'running');
  }

  #gpus(): SystemInfo['gpus'] {
    const t = Date.now() / 1000;
    const training = this.#trainingHere();
    const gpus: SystemInfo['gpus'] = [
      {
        ordinal: 0,
        name: GPU_NAME,
        backend: this.scenario === 'no-tensor-cores' ? ('dx12' as const) : ('vulkan' as const),
        memory_bytes: 16 * 2 ** 30,
        util: training ? round(0.91 + 0.04 * smoothNoise(5, t / 2), 3) : 0.02,
      },
    ];
    if (this.scenario === 'gpu-unplugged' && !this.#unplugged)
      gpus.push({ ordinal: 1, name: 'NVIDIA GeForce RTX 3060', backend: 'vulkan' as const, memory_bytes: 12 * 2 ** 30, util: 0 });
    return gpus;
  }

  /**
   * FOLO's GPU buffer pool (`gpu_pool`), as a real server reports it once it has used the GPU: the training run's
   * tensors in use and a few freed buffers kept for reuse; between runs the cache is given back and only the last run's
   * peak stays. Absent from a server that never used the GPU (`empty`) and from older ones (`no-throughput`).
   */
  #gpuPool(): GpuPool | undefined {
    if (this.scenario === 'empty' || this.scenario === 'no-throughput') return undefined;
    const training = this.#trainingHere();
    const peak = Math.round(5.31 * 2 ** 30);
    return training
      ? { device: 'gpu:0', live_bytes: Math.round(4.62 * 2 ** 30), cached_bytes: Math.round(0.88 * 2 ** 30), peak_live_bytes: peak }
      : { device: 'gpu:0', live_bytes: 0, cached_bytes: 0, peak_live_bytes: peak };
  }

  /** `amp-refused`: set by the refused run; `/api/system` then says what the restarted server can do. */
  #ampRefused = false;

  #capabilities(): Capabilities {
    return capabilities(this.#ampRefused ? 'tensor-cores-off' : this.scenario);
  }

  system(): Promise<SystemInfo> {
    return this.#call(() => {
      const t = Date.now() / 1000;
      const pool = this.#gpuPool();
      return {
        cpu: { name: CPU_NAME, threads: 32, util: round(0.27 + 0.08 * smoothNoise(3, t / 4), 3) },
        memory: { used_bytes: Math.round((21.4 + 0.3 * smoothNoise(4, t / 9)) * 2 ** 30), total_bytes: 64 * 2 ** 30 },
        gpus: this.#gpus(),
        features: [
          ...(this.scenario === 'no-throughput'
            ? ['export_onnx_switches', 'queue_control', 'weights_onnx']
            : ['export_onnx_switches', 'queue_control', 'weights_onnx', 'throughput', 'run_estimate', 'benchmark', 'run_cli']),
          // The in-browser runtime in this build (the name requested of the server; lib/inference `wasmFromSystem`).
          ...(this.scenario === 'no-wasm' ? [] : ['web_wasm']),
          ...(this.#cloudOn ? ['cloud'] : []),
        ],
        capabilities: this.#capabilities(),
        ...(pool ? { gpu_pool: pool } : {}),
      };
    }, 60);
  }

  /** A server from before these endpoints: an unknown route. */
  #noRoute(path: string): ApiError {
    return new ApiError(404, 'not_found', `No route for ${path}.`);
  }

  throughput(): Promise<ThroughputTable> {
    return this.#call(() => {
      if (this.scenario === 'no-throughput') throw this.#noRoute('GET /api/system/throughput');
      return this.#throughput.table();
    }, 80);
  }

  /** About 1.5 s, the shortest a real probe takes (docs/API.md: the budget, 1–10 s, plus setup). */
  async benchmark(body: BenchmarkRequest = {}): Promise<BenchmarkReport> {
    if (this.scenario === 'no-throughput') return this.#call(() => Promise.reject(this.#noRoute('POST /api/system/benchmark')));
    const busy = () => {
      const now = Date.now();
      const active = this.#runs.find((p) => this.#created.has(p.id) && !p.cloud && ['queued', 'running', 'stopping'].includes(progressAt(p, now).status));
      if (this.#probing) return new ApiError(409, 'run_active', 'A benchmark probe is already running.', 'Wait for it to finish.');
      if (active) return new ApiError(409, 'run_active', `Run ${active.name} is ${progressAt(active, now).status}.`, 'Measure when it has finished, or stop it first.');
      return null;
    };
    const err = busy();
    if (err) return this.#call(() => Promise.reject(err), 60);
    this.#probing = true;
    const t = Date.now();
    try {
      if (this.scenario === 'loading') return await new Promise<BenchmarkReport>(() => {});
      if (this.latency > 0) await sleep(1500);
      if (this.scenario === 'error') throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
      return this.#throughput.probe(body, Date.now(), (Date.now() - t) / 1000);
    } finally {
      this.#probing = false;
    }
  }

  // ---------- datasets ----------

  datasets(): Promise<Dataset[]> {
    return this.#call(() => this.#datasets.map((s) => toDataset(s, this.#t0, false)));
  }

  dataset(id: string): Promise<Dataset> {
    return this.#call(() => toDataset(this.#spec(id), this.#t0, true));
  }

  createDataset(body: CreateDataset): Promise<Dataset> {
    return this.#call(() => {
      // As on the server: a folder that is already registered (by its path, whatever the spelling) answers with the
      // existing dataset, its name kept.
      const key = (p: string) => p.trim().replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase();
      const known = this.#datasets.find((d) => key(d.path) === key(body.path));
      if (known) return toDataset(known, this.#t0, true);
      throw new ApiError(400, 'dataset_not_found_on_disk', `No YOLO or COCO dataset found at ${body.path}.`, 'The mock server cannot read your disk; connect to folo serve to register folders.');
    }, 400);
  }

  createSyntheticDataset(body: CreateSyntheticDataset): Promise<Dataset> {
    return this.#call(() => {
      const n = Math.max(10, Math.min(20000, Math.round(body.images)));
      const train = Math.round(n * 0.8);
      const classes = (body.classes.length ? body.classes : ['circle', 'square', 'triangle']).slice(0, 6);
      const spec: MockDatasetSpec = {
        ...(specs.find((s) => s.id === 'ds_shapes') as MockDatasetSpec),
        id: `ds_${(hash(body.name + this.#nextId++) & 0xffff).toString(16).padStart(4, '0')}`,
        name: body.name,
        path: `D:/data/${body.name}`,
        classes,
        classWeights: classes.map(() => 1),
        classShape: classes.map(() => 'square' as const),
        splits: { train, val: n - train, test: 0 },
        width: body.imgsz,
        height: body.imgsz,
        seed: body.seed,
        createdAgoS: (this.#t0 - Date.now()) / 1000,
      };
      this.#datasets.unshift(spec);
      return toDataset(spec, this.#t0, true);
    }, 900);
  }

  datasetImages(id: string, q: DatasetImagesQuery): Promise<DatasetImages> {
    return this.#call(() => {
      const s = this.#spec(id);
      const idx = indices(s, q.split, q.class);
      const count = idx ? idx.length : total(s);
      const offset = Math.max(0, q.offset ?? 0);
      const limit = Math.max(0, Math.min(500, q.limit ?? 100));
      const images = [];
      for (let i = offset; i < Math.min(count, offset + limit); i++) images.push(image(s, idx ? (idx[i] as number) : i));
      return { total: count, images };
    }, 90);
  }

  datasetIssues(id: string, q: DatasetIssuesQuery): Promise<DatasetIssues> {
    return this.#call(() => {
      const all = issues(this.#spec(id));
      const matching = all.filter((i) => (!q.kind || i.kind === q.kind) && (!q.severity || i.severity === q.severity));
      const offset = Math.max(0, q.offset ?? 0);
      const limit = Math.max(0, Math.min(1000, q.limit ?? 100));
      // `counts` covers the whole dataset whatever the filter, as on the server.
      const counts: IssueCounts = { error: 0, warning: 0, info: 0, by_kind: {} };
      for (const i of all) {
        counts[i.severity]++;
        counts.by_kind[i.kind] = (counts.by_kind[i.kind] ?? 0) + 1;
      }
      return { total: matching.length, issues: matching.slice(offset, offset + limit), counts };
    }, 120);
  }

  imageUrl(datasetId: string, imageId: string, thumb?: number): Promise<string> {
    const key = `${datasetId}/${imageId}/${thumb ?? 0}`;
    const hit = this.#images.get(key);
    if (hit) {
      this.#images.delete(key);
      this.#images.set(key, hit);
      return hit;
    }
    const p = (async () => {
      const s = this.#spec(datasetId);
      const index = imageIndex(imageId);
      const blob = await renderImage(s, image(s, index), index, thumb);
      return URL.createObjectURL(blob);
    })();
    this.#images.set(key, p);
    if (this.#images.size > 900) {
      const [oldKey, old] = this.#images.entries().next().value as [string, Promise<string>];
      this.#images.delete(oldKey);
      void old.then((u) => setTimeout(() => URL.revokeObjectURL(u), 30_000));
    }
    return p;
  }

  deleteDataset(id: string): Promise<void> {
    return this.#call(() => {
      this.#spec(id);
      this.#datasets = this.#datasets.filter((d) => d.id !== id);
    });
  }

  // ---------- models ----------

  #models(): Model[] {
    const now = Date.now();
    const out: Model[] = [...this.#extraModels];
    for (const p of this.#runs) {
      const run = toRun(p, now);
      if (!run.model_id) continue;
      const best = bestOf(p, Math.floor(progressAt(p, now).done));
      const ep = best ? epochAt(p, best.epoch - 1) : null;
      const info = archInfo[p.config.arch];
      out.push({
        id: run.model_id,
        name: `${p.name}-${p.terminal?.status === 'stopped' ? 'last' : 'best'}`,
        arch: p.config.arch,
        imgsz: p.config.imgsz,
        classes: [...p.classNames],
        params: info.params,
        flops_g: info.flops,
        bytes: Math.round(info.params * 2.06),
        created: run.finished ?? run.created,
        run_id: p.id,
        metrics: ep
          ? { map50: ep.val.map50, map50_95: ep.val.map50_95, precision: ep.val.precision, recall: ep.val.recall, best_f1_conf: ep.val.best_f1_conf ?? null }
          : null,
        // As a server reads them from the checkpoint: folo.config's head, folo.train's settings (planned epochs).
        head: p.config.head ?? 'o2m',
        trained_with: { amp: p.config.amp ?? false, imgsz: p.config.imgsz, epochs: p.config.epochs },
        precision: 'f32',
        fused: false,
        source: 'run',
      });
    }
    // fasteners-n's f16 export, kept as a model (docs/API.md: the mock's m_53d2): fused, f16, no training record.
    const src = out.find((m) => m.id === 'm_fast_n');
    if (src) {
      out.push({
        ...src,
        id: 'm_fast_n_f16',
        name: 'fasteners-n-f16',
        bytes: Math.round(src.params * 2.02),
        created: new Date(Date.parse(src.created) + 3_600_000).toISOString(),
        trained_with: null,
        precision: 'f16',
        fused: true,
        source: 'file',
      });
    }
    if (this.scenario !== 'empty') {
      out.push({
        id: 'm_coco_s',
        name: 'folo-s-coco80',
        arch: 'folo-s',
        imgsz: 640,
        classes: [...COCO80],
        params: archInfo['folo-s'].params,
        flops_g: archInfo['folo-s'].flops,
        bytes: 18_870_000,
        created: new Date(this.#t0 - 20 * DAY).toISOString(),
        run_id: null,
        // Imported weights from before `best_f1_conf` (docs/FORMAT.md): no training record, and no word on the
        // threshold precision and recall were read at.
        metrics: { map50: 0.611, map50_95: 0.448, precision: 0.672, recall: 0.571, best_f1_conf: null },
        head: 'o2m',
        trained_with: null,
        precision: 'f32',
        fused: false,
        source: 'file',
      });
      // Published weights from `folo weights pull folo-n-coco` (docs/API.md, Published weights): listed by their name,
      // with what they are; their metrics are the COCO val2017 figures FOLO publishes (docs/BENCHMARKS.md).
      out.push({
        id: 'folo-n-coco',
        name: 'folo-n-coco',
        arch: 'folo-n',
        imgsz: 640,
        classes: [...COCO80],
        params: archInfo['folo-n'].params,
        flops_g: archInfo['folo-n'].flops,
        bytes: Math.round(archInfo['folo-n'].params * 4.03),
        created: new Date(this.#t0 - 9 * DAY).toISOString(),
        run_id: null,
        metrics: { map50: 0.455, map50_95: 0.314, precision: 0.583, recall: 0.427, best_f1_conf: null },
        head: 'o2m',
        trained_with: { amp: false, imgsz: 640, epochs: 50 },
        precision: 'f32',
        fused: false,
        source: 'published',
        about: COCO_ABOUT,
      });
      // An imported file with fasteners-mini's classes: /eval runs on that dataset (the first with its classes, as a
      // real server picks one), which has no validation images, so on its training split.
      out.push({
        id: 'm_mini_n',
        name: 'fasteners-mini-n',
        arch: 'folo-n',
        imgsz: 640,
        classes: ['bolt', 'nut'],
        params: archInfo['folo-n'].params,
        flops_g: archInfo['folo-n'].flops,
        bytes: Math.round(archInfo['folo-n'].params * 2.06),
        created: new Date(this.#t0 - 4 * DAY).toISOString(),
        run_id: null,
        metrics: { map50: 0.912, map50_95: 0.701, precision: 0.884, recall: 0.861, best_f1_conf: 0.402 },
        head: 'o2m',
        trained_with: { amp: false, imgsz: 640, epochs: 60 },
        precision: 'f32',
        fused: false,
        source: 'file',
      });
    }
    return out.sort((a, b) => b.created.localeCompare(a.created));
  }

  models(): Promise<Model[]> {
    return this.#call(() => this.#models());
  }

  model(id: string): Promise<Model> {
    return this.#call(() => {
      const m = this.#models().find((x) => x.id === id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${id}.`);
      return m;
    });
  }

  modelEval(id: string): Promise<ModelEval> {
    return this.#call(() => {
      const m = this.#models().find((x) => x.id === id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${id}.`);
      const p = m.run_id ? this.#runs.find((r) => r.id === m.run_id) : undefined;
      let perClass: ClassMetrics[] | null = null;
      let instances: number[] | null = null;
      let images = 500;
      if (p) {
        const best = bestOf(p, Math.floor(progressAt(p, Date.now()).done));
        perClass = best ? epochAt(p, best.epoch - 1).val.per_class : null;
      }
      // The run's dataset; an imported model's, the first with its classes (as a real server picks it).
      const same = (a: readonly string[]) => a.length === m.classes.length && a.every((c, i) => c === m.classes[i]);
      const ds = p ? this.#datasets.find((d) => d.id === p.config.dataset_id) : this.#datasets.find((d) => same(d.classes));
      // Its val split, its train split when val is empty (docs/API.md, Split fallback).
      const split: Split = ds && ds.splits.val === 0 ? 'train' : 'val';
      if (ds) {
        const share = ds.splits[split] / Math.max(1, total(ds));
        instances = stats(ds).class_counts.map((c) => c * share);
        images = ds.splits[split];
      }
      const evaluation = toModelEval(synthEvaluation({ seed: m.id, classes: m.classes, metrics: m.metrics, perClass, instances }), images);
      // folo-s-coco80 has no dataset here: a real server answers 404 (no registered dataset has its 80 classes). The
      // demo keeps its curves and answers as a server from before the provenance fields, which leaves them out.
      if (!ds) return { ...evaluation, best_f1_conf: m.metrics?.best_f1_conf ?? null };
      // The same operating point as the model's metrics (the epoch that wrote its weights), as on folo-server's mock.
      return {
        ...evaluation,
        best_f1_conf: m.metrics?.best_f1_conf ?? null,
        dataset_id: ds.id,
        dataset_name: ds.name,
        split,
        // folo_api::SplitFallback::empty_split's sentence.
        split_fallback:
          split === 'val'
            ? null
            : {
                requested: 'val',
                reason: 'empty_split',
                message: `dataset '${ds.name}' has no val images, so this is its train split (${images} images): training-set numbers, not held-out ones`,
              },
        // folo-vision's EvalConfig defaults (docs/METRICS.md §7), not the NMS settings or best_f1_conf.
        confusion_conf: 0.25,
        confusion_iou: 0.45,
      };
    }, 180);
  }

  /**
   * `/weights?format=onnx`, named as the server names it. The demo's models are not real networks, so the file is
   * empty: Studio says so instead of saving it.
   */
  onnxFile(id: string, switches: OnnxSwitches): Promise<File> {
    return this.#call(() => {
      const m = this.#models().find((x) => x.id === id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${id}.`);
      return new File([], onnxFileName(m.name, !!switches.half), { type: 'application/octet-stream' });
    }, 1200);
  }

  exportModel(id: string, format: ExportFormat, switches?: OnnxSwitches): Promise<ExportResult> {
    return this.#call(() => {
      const m = this.#models().find((x) => x.id === id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${id}.`);
      const on = { decode: !!switches?.decode, dynamic: !!switches?.dynamic, half: !!switches?.half };
      if (format !== 'onnx' && (on.decode || on.dynamic || on.half))
        throw new ApiError(400, 'invalid_body', 'decode, dynamic and half apply to ONNX exports only.');
      // The real server's names (docs/API.md): `<name>-f16.folo`, `<name>.onnx`, `<name>-f16.onnx` with `half`.
      const file = format === 'onnx' ? `${m.name}${on.half ? '-f16' : ''}.onnx` : `${m.name}-f16.folo`;
      const perParam = format === 'onnx' ? (on.half ? 2.05 : 4.1) : 2.03;
      return { path: `D:/folo/exports/${file}`, bytes: Math.round(m.params * perParam) };
    }, 1200);
  }

  // ---------- runs ----------

  runs(): Promise<Run[]> {
    return this.#call(() => {
      const now = Date.now();
      return this.#runs.map((p) => this.#toRun(p, now)).sort((a, b) => b.created.localeCompare(a.created));
    });
  }

  run(id: string): Promise<Run> {
    return this.#call(() => this.#toRun(this.#profile(id), Date.now()));
  }

  /**
   * What the server does to a request's weights before it trains (folo-server's create_run): they must exist, and a
   * fine-tune trains their architecture, as `folo train --weights` does.
   */
  #withWeights(c: TrainConfig): TrainConfig {
    if (!c.weights) return c;
    const m = this.#models().find((x) => x.id === c.weights);
    if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${c.weights}.`);
    return { ...c, arch: m.arch };
  }

  /**
   * `501 not_supported` for an `amp` run this server cannot train (docs/API.md): refused before anything is queued,
   * with `capabilities.amp.reason` as the message, as the real server and folo-server's mock do.
   */
  #checkAmp(c: TrainConfig) {
    if (!c.amp) return;
    // `amp-refused`: the server was restarted without tensor cores since the page asked.
    if (this.scenario === 'amp-refused') this.#ampRefused = true;
    const caps = this.#capabilities();
    if (c.device !== 'cpu' && caps.amp.available) return;
    const reason =
      c.device === 'cpu'
        ? 'mixed-precision training (amp) needs a GPU with tensor cores (f16 cooperative matrices): the run is on the CPU; train without amp (f32), or on such a GPU'
        : (caps.amp.reason ?? 'mixed-precision training (amp) is not available on this server');
    throw new ApiError(501, 'not_supported', reason, AMP_HINT);
  }

  /**
   * `folo train` for `c` as the server writes it (`RunEstimate.cli`, `GET /api/runs/{id}/cli`, quoted for PowerShell):
   * published weights by their name, another model by its file under the data dir (made up: mock models have no
   * files, as on folo-server's mock).
   */
  #cli(c: TrainConfig): CliCommand {
    const ds = this.#spec(c.dataset_id);
    const m = c.weights ? this.#models().find((x) => x.id === c.weights) : undefined;
    const cmd = trainCommand(c, ds.path, m && m.source !== 'published' ? { ...m, path: `D:/folo/models/${m.name}.folo` } : m);
    return { cli: cmd.line, argv: ['folo', ...cmd.argv] };
  }

  runCli(id: string): Promise<CliCommand> {
    return this.#call(() => {
      if (this.scenario === 'no-throughput') throw this.#noRoute(`GET /api/runs/${id}/cli`);
      return this.#cli(this.#profile(id).config);
    }, 80);
  }

  createRun(body: TrainConfig): Promise<Run> {
    return this.#call(() => {
      const ds = this.#spec(body.dataset_id);
      const c = this.#withWeights(body);
      const cloud = this.#compute(c);
      if (cloud) return this.#createCloudRun(c, ds, cloud);
      // `gpu-unplugged`: the second adapter goes away as soon as a run asks for it.
      if (this.scenario === 'gpu-unplugged' && c.device === 'gpu:1') this.#unplugged = true;
      const gpu = /^gpu:(\d+)$/.exec(c.device);
      if (gpu && !this.#gpus().some((g) => g.ordinal === Number(gpu[1]))) throw this.#noDevice(c.device);
      // The real server's body error for a head it does not know (serde: `o2m` or `o2o`; missing means `o2m`).
      if (c.head !== undefined && c.head !== 'o2m' && c.head !== 'o2o')
        throw new ApiError(400, 'invalid_body', `field \`head\`: unknown variant \`${String(c.head)}\`, expected \`o2m\` or \`o2o\``);
      this.#checkAmp(c);
      const now = Date.now();
      const id = `run_${(hash(c.name + now + this.#nextId++) & 0xffff).toString(16).padStart(4, '0')}`;
      const sizeIdx = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'].indexOf(c.arch);
      const p: RunProfile = withResolved({
        id,
        name: c.name,
        config: c,
        classes: ds.classes.length,
        classNames: [...ds.classes],
        classEase: ds.classes.map((_, i) => 0.85 + 0.3 * rng(hash(id) + i)()),
        stepsPerEpoch: optimizerSteps(ds.splits.train, c.batch),
        epochSeconds: 600 / Math.max(1, c.epochs),
        mapMax: 0.55 + sizeIdx * 0.04,
        tau: Math.max(4, c.epochs / 7),
        imgsPerS: [520, 320, 170, 130, 70][sizeIdx] ?? 200,
        createdAt: now,
        // Queued: the worker starts it once the training slot is free (#schedule).
        startedAt: null,
        modelId: `m_${id.slice(4)}`,
        device: c.device === 'cpu' ? 'cpu (AMD Ryzen 9 7950X)' : DEVICE,
      }, ds.splits.train);
      this.#runs.unshift(p);
      this.#created.add(p.id);
      this.#queue.push(p.id);
      this.#schedule(now);
      return this.#toRun(p, now);
    }, 300);
  }

  estimateRun(body: TrainConfig): Promise<RunEstimate> {
    return this.#call(() => {
      if (this.scenario === 'no-throughput') throw this.#noRoute('POST /api/runs/estimate');
      // The same checks as POST /api/runs; nothing is created.
      const ds = this.#spec(body.dataset_id);
      const gpu = /^gpu:(\d+)$/.exec(body.device);
      if (gpu && !this.#gpus().some((g) => g.ordinal === Number(gpu[1]))) throw this.#noDevice(body.device);
      const c = this.#withWeights(body);
      const cloud = this.#compute(c, false);
      const est = { ...this.#throughput.estimate(c, ds.splits.train, ds.splits.val, cloud?.offer.gpu), cli: this.#cli(c).cli };
      const price = cloud?.offer.price_per_hour;
      if (!cloud || price === null || price === undefined) return est;
      // docs/API.md, `RunEstimate.cost`: the price over the training time (booting and the upload come on top).
      const usd = (s: number) => Math.round(price * (s / 3600) * 100) / 100;
      return {
        ...est,
        cost: {
          price_per_hour: price,
          est_usd: usd(est.est_total_s),
          est_usd_low: usd(est.est_total_s_low),
          est_usd_high: usd(est.est_total_s_high),
          max_usd: Math.round(price * cloud.hours * 100) / 100,
        },
      };
    }, 120);
  }

  /**
   * A config's `compute`, checked as `POST /api/runs` checks it (docs/API.md, Cloud runs): the connection, its offer,
   * the time limit and `device: auto`. `refuse`: an offer with nothing available is `409 cloud_refused`, as launching
   * it would be (an estimate launches nothing). Null for a run on this machine, and on a server without the feature,
   * which ignores a field it does not know.
   */
  #compute(c: TrainConfig, refuse = true): { conn: CloudConnection; offer: CloudOffer; hours: number } | null {
    const want = c.compute;
    if (!want || !this.#cloudOn) return null;
    const conn = this.#connection(want.connection_id);
    if (conn.status === 'error') throw new ApiError(422, 'cloud_auth_failed', conn.error ?? `${conn.name} rejected the credentials`, 'Check the connection in Settings › Cloud.');
    const offer = this.#offersOf(conn).find((o) => o.id === want.offer_id);
    if (!offer) throw new ApiError(404, 'offer_not_found', `${conn.name} has no offer '${want.offer_id}' (any more)`, 'List its offers again: GET /api/cloud/connections/{id}/offers.');
    const hours = want.max_hours ?? 12;
    if (!(hours >= 0.25 && hours <= 720)) throw new ApiError(400, 'invalid_config', `\`compute.max_hours\` must be from 0.25 to 720 hours, not ${hours}`);
    if (c.device !== 'auto') throw new ApiError(400, 'invalid_config', `\`device\` must be \`auto\` for a cloud run (the worker picks its GPU), not \`${c.device}\``);
    if (refuse && offer.availability === 'none')
      throw new ApiError(409, 'cloud_refused', `${conn.name} has no ${offer.gpu} (${offer.detail ?? offer.id}) available right now`, 'Pick another GPU, or try again later.');
    return { conn, offer, hours };
  }

  /**
   * A cloud run: out of the training queue, through every phase (about 2 s each of provisioning, booting and the
   * upload), then training like any mock run, on the worker, then its weights collected and its instance released.
   */
  #createCloudRun(c: TrainConfig, ds: MockDatasetSpec, cloud: { conn: CloudConnection; offer: CloudOffer; hours: number }): Run {
    const now = Date.now();
    const id = `run_${(hash(c.name + now + this.#nextId++) & 0xffff).toString(16).padStart(4, '0')}`;
    const nonce = (hash(`${id}:nonce`) & 0xffff).toString(16).padStart(4, '0');
    const sizeIdx = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'].indexOf(c.arch);
    const provisions = PROVIDERS.find((x) => x.id === cloud.conn.provider)?.provisions ?? false;
    const timeline: CloudTimeline = {
      connectionId: cloud.conn.id,
      provider: cloud.conn.provider,
      offer: cloud.offer,
      maxHours: cloud.hours,
      instanceId: provisions ? `ci_${nonce}` : null,
      instanceName: `folo-${id}-${nonce}`,
      providerRef: (hash(`${id}:pod`) >>> 0).toString(36).padStart(7, '0'),
      launchedAt: now,
      // A remote server is not provisioned: it is asked at once.
      provisionMs: provisions ? 2000 : 0,
      bootMs: 2000,
      syncMs: 2000,
      // The dataset as FOLO sends it (about 90 kB an image), and best.folo with last.folo (16 bytes a parameter each).
      uploadBytes: total(ds) * 90_000,
      weightsBytes: archInfo[c.arch].params * 32,
    };
    const p: RunProfile = withResolved(
      {
        id,
        name: c.name,
        config: { ...c, compute: { ...c.compute!, max_hours: cloud.hours } },
        classes: ds.classes.length,
        classNames: [...ds.classes],
        classEase: ds.classes.map((_, i) => 0.85 + 0.3 * rng(hash(id) + i)()),
        stepsPerEpoch: optimizerSteps(ds.splits.train, c.batch),
        epochSeconds: 600 / Math.max(1, c.epochs),
        mapMax: 0.55 + sizeIdx * 0.04,
        tau: Math.max(4, c.epochs / 7),
        imgsPerS: [520, 320, 170, 130, 70][sizeIdx] ?? 200,
        createdAt: now,
        startedAt: trainAt(timeline),
        modelId: `m_${id.slice(4)}`,
        device: `NVIDIA GeForce ${cloud.offer.gpu} (vulkan)`,
        cloud: timeline,
      },
      ds.splits.train,
    );
    this.#runs.unshift(p);
    this.#created.add(p.id);
    return this.#toRun(p, now);
  }

  /** `404 device_not_found` in the real server's words. */
  #noDevice(device: string): ApiError {
    const gpus = this.#gpus();
    return new ApiError(
      404,
      'device_not_found',
      `device ${device} is not available: this machine has no such GPU; FOLO can use ${['cpu', ...gpus.map((g) => `gpu:${g.ordinal}`)].join(', ')}`,
      `this machine has ${gpus.map((g) => `gpu:${g.ordinal} (${g.name}, ${g.backend})`).join(', ')}, cpu; use device \`auto\` (the GPU when there is one) or \`cpu\``,
    );
  }

  /** Like `folo serve --mock`: resuming needs real checkpoints. */
  resumeRun(id: string): Promise<Run> {
    return this.#call(() => {
      this.#profile(id);
      throw new ApiError(501, 'not_implemented', 'Resuming runs needs a real server.', 'Start folo serve without --mock to resume from checkpoints.');
    }, 200);
  }

  stopRun(id: string): Promise<Run> {
    return this.#call(() => {
      const p = this.#profile(id);
      const now = Date.now();
      const { done, status } = progressAt(p, now);
      if (status !== 'running' && status !== 'queued')
        throw new ApiError(409, 'run_not_running', `Run ${p.name} is ${status}; only queued or running runs can be stopped.`);
      if (p.cloud) {
        // The worker's run stops, its checkpoint is collected and the instance released (before training, released at
        // once); the run says `stopping` until then (docs/CLOUD.md, Stopping).
        p.cloud.endAt = Math.min(now, trainEnd(p));
        const end = cloudFinish({ ...p, cloud: p.cloud });
        if (status === 'queued') {
          p.startedAt = null;
          p.neverStarted = true;
        }
        p.terminal = { status: 'stopped', atEpoch: done, at: end, stoppingUntil: end };
        return this.#toRun(p, now);
      }
      // A run that waits leaves the queue and stops at once; its trainer was never built.
      const waiting = status === 'queued';
      this.#queue = this.#queue.filter((x) => x !== p.id);
      const startedAt = loopStart(p, now);
      p.startedAt = startedAt;
      p.loop = false;
      p.neverStarted = waiting;
      p.terminal = waiting ? { status: 'stopped', atEpoch: 0, at: now } : { status: 'stopped', atEpoch: done, at: now + 1200, stoppingUntil: now + 1200 };
      return this.#toRun(p, now);
    });
  }

  deleteRun(id: string): Promise<void> {
    return this.#call(() => {
      const p = this.#profile(id);
      const status = progressAt(p, Date.now()).status;
      if (status === 'queued' || status === 'running' || status === 'stopping')
        throw new ApiError(409, 'run_active', `Run ${p.name} is ${status}.`, 'Stop it first.');
      this.#runs = this.#runs.filter((r) => r.id !== id);
    });
  }

  /** POST /api/runs/{id}/queue, with the real server's errors and log lines (docs/API.md). */
  moveRun(id: string, req: QueueRequest): Promise<Run> {
    return this.#call(() => {
      const p = this.#profile(id);
      const action = 'action' in req;
      const position = 'position' in req;
      if (action === position || (position && !(Number.isInteger(req.position) && req.position >= 1)))
        throw new ApiError(400, 'invalid_body', 'give exactly one of `action` and `position` (a place from 1)');
      const now = Date.now();
      const from = this.#queue.indexOf(id);
      const order = applyQueueMove(this.#queue, id, req);
      if (!order) {
        const status = progressAt(p, now).status;
        throw new ApiError(
          409,
          'run_not_queued',
          `run '${id}' ${status === 'queued' ? 'is starting' : `is ${status}`}, not waiting in the queue`,
          'only runs waiting in the queue (with a queue_position) can be moved',
        );
      }
      this.#queue = order;
      const start = action && req.action === 'start';
      // `start` on a free slot: the worker takes it now.
      if (start) this.#schedule(now);
      const to = this.#queue.indexOf(id);
      const place = `${to + 1} of ${this.#queue.length}`;
      this.#note(
        id,
        start
          ? to < 0
            ? 'start requested: starting now'
            : `start requested: moved to the front (${place}); it starts when the active run ends`
          : from === to
            ? `queue position unchanged (${place})`
            : `moved in the queue to position ${place}`,
      );
      return this.#toRun(p, now);
    }, 160);
  }

  runMetrics(id: string, _signal?: AbortSignal, every = 4): Promise<RunMetrics> {
    return this.#call(() => {
      const p = this.#profile(id);
      const now = Date.now();
      // A looping run's history restarts with each loop.
      if (p.loop && p.startedAt !== null) {
        const shifted = { ...p, startedAt: loopStart(p, now), loop: false };
        return history(shifted, now, every);
      }
      return history(p, now, every);
    }, 220);
  }

  runSamples(id: string, epoch?: number): Promise<RunSamples> {
    return this.#call(() => {
      const p = this.#profile(id);
      const done = Math.floor(progressAt(p, Date.now()).done);
      // `epoch` is 1-based like the API; `e` indexes the run model (0-based).
      const e = Math.max(0, Math.min(done - 1, (epoch ?? done) - 1));
      const ds = this.#spec(p.config.dataset_id);
      const quality = done > 0 ? epochAt(p, e).val.map50_95 : 0;
      const samples: Sample[] = [];
      // A fixed set of up to 16 validation images, spread evenly over the split; a dataset with no validation images
      // validates on its training split (docs/API.md, Split fallback), so its samples come from there.
      const fromTrain = ds.splits.val === 0;
      const pool = fromTrain ? ds.splits.train : ds.splits.val;
      const first = fromTrain ? 0 : ds.splits.train;
      const n = Math.min(16, pool);
      for (let i = 0; i < n; i++) {
        const index = first + Math.floor((i * pool) / n);
        const img = image(ds, index);
        samples.push({
          image_id: img.id,
          dataset_id: ds.id,
          width: img.width,
          height: img.height,
          truth: img.boxes,
          pred: predictBoxes(img.boxes, img.width, img.height, quality, hash(p.id) + i * 977 + e * 13, ds.classes.length)
            .filter((b) => b.score >= 0.25)
            .map((b) => ({ ...b, name: ds.classes[b.class] ?? `class ${b.class}` })),
        });
      }
      return { epoch: e + 1, samples };
    }, 260);
  }

  runEvents(id: string, h: RunEventHandlers): () => void {
    if (this.scenario === 'loading' || this.scenario === 'error') {
      if (this.scenario === 'error') setTimeout(() => h.connection?.('reconnecting'), 500);
      return () => {};
    }
    let p: RunProfile;
    try {
      p = this.#profile(id);
    } catch {
      return () => {};
    }
    const ds = specs.find((s) => s.id === p.config.dataset_id);
    const subs = this.#subs.get(id) ?? new Set<RunEventHandlers>();
    this.#subs.set(id, subs);
    subs.add(h);
    // The run's place in the queue and how many wait, as last reported (the snapshot carries the first one).
    const place = () => {
      const i = this.#queue.indexOf(id);
      return i < 0 ? null : { position: i + 1, waiting: this.#queue.length };
    };
    let lastPlace = JSON.stringify(place());
    let last = progressAt(p, Date.now());
    let lastStart = loopStart(p, Date.now());
    let lastStep = Math.floor(last.done * p.stepsPerEpoch);
    let lastEpoch = Math.floor(last.done);
    // A cloud run's state, sent when it changes: a phase or what it names at once, progress at most once a second and
    // the cost about once a minute (docs/API.md, SSE `cloud`).
    const cloudKey = (c: RunCloud) => JSON.stringify([c.phase, c.instance_id, c.deadline, c.synced]);
    let lastCloud = p.cloud ? cloudOf({ ...p, cloud: p.cloud }, Date.now()) : null;
    let cloudSent = Date.now();
    const t0 = setTimeout(() => {
      h.connection?.('open');
      h.status?.({ status: last.status, error: p.terminal?.error ?? null });
    }, 30);
    let dropped = false;
    const timer = setInterval(() => {
      const now = Date.now();
      // `stale`: the server stops, its stream with it (EventSource then reports each failed retry).
      if (this.scenario === 'stale' && now - this.#t0 > STALE_AFTER) {
        if (!dropped) h.connection?.('reconnecting');
        dropped = true;
        return;
      }
      this.#schedule(now);
      // A `queue` event whenever the place or the count changed (docs/API.md: state, not changes); before `running`.
      const q = place();
      if (JSON.stringify(q) !== lastPlace) {
        lastPlace = JSON.stringify(q);
        if (q) h.queue?.(q);
      }
      const cur = progressAt(p, now);
      const start = loopStart(p, now);
      if (p.loop && p.startedAt !== null && start !== lastStart) {
        // The demo loop restarted: behave like a fresh run.
        lastStart = start;
        lastStep = -1;
        lastEpoch = 0;
        h.status?.({ status: 'running', error: null });
        for (const l of startLogs(p, start, ds?.name ?? p.config.dataset_id, ds?.splits.train ?? 0)) h.log?.(l);
        h.reconnect?.();
      }
      if (cur.status === 'running') {
        const step = Math.floor(cur.done * p.stepsPerEpoch);
        if (step !== lastStep) {
          lastStep = step;
          h.step?.(stepAt(p, step));
        }
      }
      // Flush finished epochs before any terminal status, so the last one is never skipped.
      if (cur.status !== 'queued') {
        const ep = Math.floor(cur.done);
        while (lastEpoch < ep) {
          const e = epochAt(p, lastEpoch);
          h.epoch?.(e);
          for (const l of epochLog(p, e, start)) h.log?.(l);
          lastEpoch++;
        }
      }
      if (p.cloud && lastCloud) {
        const c = cloudOf({ ...p, cloud: p.cloud }, now);
        const moved = c.progress?.done_bytes !== lastCloud.progress?.done_bytes && now - cloudSent >= 1000;
        const billed = c.cost_usd !== lastCloud.cost_usd && now - cloudSent >= 60_000;
        if (cloudKey(c) !== cloudKey(lastCloud) || moved || billed) {
          lastCloud = c;
          cloudSent = now;
          h.cloud?.(c);
        }
      }
      if (cur.status !== last.status) h.status?.({ status: cur.status, error: p.terminal?.error ?? null, ...(p.terminal?.errorCode ? { error_code: p.terminal.errorCode } : {}) });
      if (cur.status === 'stopped' && last.status !== 'stopped')
        h.log?.({ t: new Date(now).toISOString(), level: 'warn', message: `stopped by user at epoch ${Math.floor(cur.done) + 1}; last checkpoint kept` });
      last = cur;
    }, 100);
    return () => {
      clearTimeout(t0);
      clearInterval(timer);
      subs.delete(h);
    };
  }

  runLogs(id: string, limit = 500): Promise<RunLogs> {
    return this.#call(() => {
      this.#profile(id);
      return { lines: this.#logHistory(id).slice(-Math.max(1, Math.min(5000, Math.floor(limit)))) };
    }, 120);
  }

  /** Log lines so far for a run, reconstructed from its profile and the clock. */
  #logHistory(id: string): LogLine[] {
    const p = this.#runs.find((r) => r.id === id);
    const notes = this.#notes.get(id) ?? [];
    if (!p) return [...notes];
    const now = Date.now();
    const ds = specs.find((s) => s.id === p.config.dataset_id);
    const cloud = p.cloud ? cloudLog(p.cloud, trainEnd(p), now, ds?.name ?? p.config.dataset_id) : [];
    if (p.startedAt === null || now < p.startedAt) return [...cloud, ...notes].sort((a, b) => a.t.localeCompare(b.t));
    const start = loopStart(p, now);
    const lines = startLogs(p, start, ds?.name ?? '', ds?.splits.train ?? 0);
    const done = Math.floor(progressAt(p, now).done);
    for (let e = 0; e < done; e++) lines.push(...epochLog(p, epochAt(p, e), start));
    if (p.terminal?.status === 'failed') lines.push({ t: new Date(start + done * p.epochSeconds * 1000).toISOString(), level: 'error', message: p.terminal.error ?? 'failed' });
    return [...cloud, ...lines, ...notes].sort((a, b) => a.t.localeCompare(b.t));
  }

  // ---------- cloud ----------

  /** A server without the `cloud` feature (`cloud-off`, older ones): `/api/cloud` is an unknown route. */
  #cloudRoute(path: string) {
    if (!this.#cloudOn) throw this.#noRoute(path);
  }

  #connection(id: string): CloudConnection {
    const c = this.#connections.find((x) => x.id === id);
    if (!c) throw new ApiError(404, 'connection_not_found', `no cloud connection '${id}'`, 'GET /api/cloud/connections lists them.');
    return c;
  }

  #offersOf(c: CloudConnection): CloudOffer[] {
    return c.provider === 'runpod' ? RUNPOD_OFFERS : [REMOTE_OFFER];
  }

  /** Every instance the mock's cloud runs launched, as at `now`: the live ones first, then the newest. */
  #instances(now: number): { instance: CloudInstance; run: RunProfile }[] {
    const out: { instance: CloudInstance; run: RunProfile }[] = [];
    for (const p of this.#runs) {
      const instance = p.cloud ? instanceAt(p.cloud, p.id, trainEnd(p), now) : null;
      if (instance) out.push({ instance, run: p });
    }
    const alive = (i: CloudInstance) => i.status !== 'terminated' && i.status !== 'lost';
    return out.sort((a, b) => Number(alive(b.instance)) - Number(alive(a.instance)) || b.instance.created.localeCompare(a.instance.created));
  }

  cloudProviders(): Promise<CloudProvider[]> {
    return this.#call(() => {
      this.#cloudRoute('GET /api/cloud/providers');
      return structuredClone(PROVIDERS);
    }, 60);
  }

  cloudConnections(): Promise<CloudConnection[]> {
    return this.#call(() => {
      this.#cloudRoute('GET /api/cloud/connections');
      return structuredClone(this.#connections);
    }, 80);
  }

  /**
   * `POST /api/cloud/connections`: any values but a secret containing `bad` (`422 cloud_auth_failed`), as on
   * folo-server's mock. The secrets are kept as their hint only; the provider takes a moment to answer.
   */
  connectCloud(body: ConnectCloud): Promise<CloudConnection> {
    return this.#call(() => {
      this.#cloudRoute('POST /api/cloud/connections');
      const provider = PROVIDERS.find((x) => x.id === body.provider);
      if (!provider) throw new ApiError(404, 'provider_not_found', `no cloud provider '${body.provider}'`, 'GET /api/cloud/providers lists them.');
      const value = (key: string) => (body.values[key] ?? '').trim();
      for (const f of provider.fields) if (f.required && !value(f.key)) throw new ApiError(400, 'invalid_body', `\`${f.key}\` is required to connect ${provider.name}`);
      const url = provider.id === 'remote' ? value('url') : '';
      if (provider.id === 'remote' && !/^https?:\/\/[^\s/]+/i.test(url)) throw new ApiError(400, 'invalid_body', '`url` must be an http:// or https:// URL');
      if (provider.fields.some((f) => f.secret && value(f.key).includes('bad')))
        throw new ApiError(
          422,
          'cloud_auth_failed',
          provider.id === 'remote' ? `the server at ${url} refused the token (401 unauthorized)` : `${provider.name} rejected the API key (401 Unauthorized)`,
          provider.id === 'remote' ? 'Use the token the server was started with (`folo serve --token`).' : 'Create a key with write access to pods: RunPod console › Settings › API Keys.',
        );
      const taken = new Set(this.#connections.map((c) => c.name));
      let name = body.name?.trim() || provider.name;
      for (let k = 2; !body.name?.trim() && taken.has(name); k++) name = `${provider.name} ${k}`;
      const host = url.replace(/^https?:\/\//i, '').replace(/[/?#].*$/, '');
      const local = /^(localhost|127\.|\[::1\]|10\.|192\.168\.)/i.test(host);
      const now = Date.now();
      const conn: CloudConnection = {
        id: `cc_${(hash(`${name}:${now}:${this.#nextId++}`) & 0xffff).toString(16).padStart(4, '0')}`,
        provider: provider.id,
        name,
        created: new Date(now).toISOString(),
        status: 'ok',
        account: provider.id === 'remote' ? host : 'you@example.com',
        balance_usd: provider.id === 'remote' ? null : 42.17,
        error: null,
        note:
          provider.id === 'remote' && /^http:/i.test(url) && !local
            ? 'Reached over http://: the token and the data travel unencrypted. Use https:// across networks you do not control.'
            : null,
        checked: new Date(now).toISOString(),
        values: Object.fromEntries(
          provider.fields.filter((f) => value(f.key)).map((f) => [f.key, f.secret ? secretHint(value(f.key)) : value(f.key)]),
        ),
      };
      this.#connections.push(conn);
      return structuredClone(conn);
    }, 1200);
  }

  /** `POST …/check`: the stored credentials again; the revoked key of `cloud-trouble` still fails (as `200`). */
  checkCloudConnection(id: string): Promise<CloudConnection> {
    return this.#call(() => {
      this.#cloudRoute(`POST /api/cloud/connections/${id}/check`);
      const c = this.#connection(id);
      c.checked = new Date().toISOString();
      return structuredClone(c);
    }, 900);
  }

  disconnectCloud(id: string): Promise<void> {
    return this.#call(() => {
      this.#cloudRoute(`DELETE /api/cloud/connections/${id}`);
      const c = this.#connection(id);
      const now = Date.now();
      const live = this.#instances(now).find(({ instance }) => instance.connection_id === id && instance.status !== 'terminated' && instance.status !== 'lost');
      const active = this.#runs.find((p) => p.cloud?.connectionId === id && ['queued', 'running', 'stopping'].includes(progressAt(p, now).status));
      if (live || active)
        throw new ApiError(
          409,
          'connection_in_use',
          live ? `${c.name} has a live instance, ${live.instance.name}` : `${c.name} has an active run, ${active!.name}`,
          'Terminate its instances, or wait for its runs to end, then disconnect it.',
        );
      this.#connections = this.#connections.filter((x) => x.id !== id);
    }, 300);
  }

  cloudOffers(id: string): Promise<CloudOfferList> {
    return this.#call(() => {
      this.#cloudRoute(`GET /api/cloud/connections/${id}/offers`);
      const c = this.#connection(id);
      if (c.status === 'error') throw new ApiError(422, 'cloud_auth_failed', c.error ?? `${c.name} rejected the credentials`, 'Check the connection in Settings › Cloud.');
      return { offers: structuredClone(this.#offersOf(c)), fetched: new Date().toISOString() };
    }, 400);
  }

  cloudInstances(): Promise<CloudInstance[]> {
    return this.#call(() => {
      this.#cloudRoute('GET /api/cloud/instances');
      return this.#instances(Date.now()).map((x) => x.instance);
    }, 100);
  }

  /**
   * `DELETE /api/cloud/instances/{id}`: terminating, confirmed a moment later; a run still using it fails with
   * `cloud_instance_lost` (its copied checkpoints stay). One already gone answers its record, and one whose termination
   * keeps failing (`cloud-trouble`) stays `terminating` with its error.
   */
  terminateCloudInstance(id: string): Promise<CloudInstance> {
    return this.#call(() => {
      this.#cloudRoute(`DELETE /api/cloud/instances/${id}`);
      const now = Date.now();
      const hit = this.#instances(now).find((x) => x.instance.id === id);
      if (!hit) throw new ApiError(404, 'instance_not_found', `no cloud instance '${id}'`, 'GET /api/cloud/instances lists them.');
      const { instance, run: p } = hit;
      if (instance.status === 'terminated' || instance.status === 'lost' || instance.error || !p.cloud) return instance;
      const { done, status } = progressAt(p, now);
      if (['queued', 'running', 'stopping'].includes(status)) {
        p.terminal = {
          status: 'failed',
          atEpoch: done,
          error: `the instance ${instance.name} was terminated before the run ended; the checkpoints copied so far are kept`,
          errorCode: 'cloud_instance_lost',
          at: now,
        };
        if (status === 'queued') {
          p.startedAt = null;
          p.neverStarted = true;
        }
      }
      p.cloud.endAt = Math.min(now, p.cloud.endAt ?? now);
      p.cloud.lost = true;
      return instanceAt(p.cloud, p.id, trainEnd(p), now) ?? instance;
    }, 500);
  }

  // ---------- inference ----------

  modelWeightsTag(): Promise<string | null> {
    return Promise.resolve(null);
  }

  modelWeights(id: string): Promise<ArrayBuffer> {
    // The demo's models are not real networks: there is no file to run in the browser.
    return Promise.reject(new ApiError(501, 'not_implemented', `Demo model ${id} has no weights file.`));
  }

  predict(input: PredictInput, opts: PredictOptions): Promise<Prediction> {
    return this.#predict(input, opts, true);
  }

  /** The demo's in-browser path (the Playground's WebGPU / CPU): the same boxes, never the server's GPU. */
  predictInPage(input: PredictInput, opts: PredictOptions): Promise<Prediction> {
    return this.#predict(input, opts, false);
  }

  #predict(input: PredictInput, opts: PredictOptions, server: boolean): Promise<Prediction> {
    return (async () => {
      const models = await this.models();
      const m = models.find((x) => x.id === opts.model_id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${opts.model_id}.`);
      if (server && this.scenario === 'oom' && m.arch !== 'folo-n')
        throw new ApiError(
          507,
          'gpu_out_of_memory',
          'the GPU ran out of memory (wgpu backend: Out of Memory)',
          'lower `batch` or `imgsz`, or use a smaller `arch` (folo-n, folo-s)',
        );
      let width: number;
      let height: number;
      let boxes: Scored[];
      const quality = (m.metrics?.map50_95 ?? 0.5) + 0.12;
      if (input.kind === 'dataset') {
        const s = this.#spec(input.dataset_id);
        const img = image(s, imageIndex(input.image_id));
        width = img.width;
        height = img.height;
        // Map dataset classes into the model's class space by name; objects of classes the model does not know
        // get a stable but arbitrary model class (as a real model would mislabel them).
        const toModel = s.classes.map((n, i) => {
          const j = m.classes.indexOf(n);
          return j >= 0 ? j : hash(`${n}:${i}`) % Math.max(1, m.classes.length);
        });
        const truth = img.boxes.map((b) => ({ ...b, class: toModel[b.class] ?? 0 }));
        boxes = predictBoxes(truth, width, height, quality, hash(input.image_id + m.id), m.classes.length);
      } else {
        const dims = await imageSize(input.file);
        width = dims.width;
        height = dims.height;
        boxes = inventBoxes(width, height, hash(`${input.file.size}:${width}x${height}`), m.classes.length);
      }
      const conf = opts.conf ?? 0.25;
      // A loose NMS threshold lets near-duplicate boxes through, as on the real server.
      if ((opts.iou ?? 0.7) > 0.8) boxes = withDuplicates(boxes, width, height, hash(m.id) ^ (width * 31 + height));
      const r = rng(hash(m.id) + width + height);
      const scale = (m.imgsz / 640) ** 2;
      const inference = archInfo[m.arch].ms * scale * (1 + (r() - 0.5) * 0.08);
      const pre = 0.55 + r() * 0.35;
      const post = 0.12 + boxes.length * 0.004 + r() * 0.05;
      const out: PredictBox[] = boxes
        .filter((b) => b.score >= conf)
        .slice(0, opts.max_det ?? 300)
        .map((b) => ({ ...b, name: m.classes[b.class] ?? `class ${b.class}` }));
      return {
        model_id: m.id,
        image: { width, height },
        boxes: out,
        timing: { preprocess_ms: round(pre, 2), inference_ms: round(inference, 2), postprocess_ms: round(post, 2), total_ms: round(pre + inference + post, 2) },
        device: 'gpu:0',
      };
    })();
  }
}

function round(x: number, d: number) {
  const k = 10 ** d;
  return Math.round(x * k) / k;
}

async function imageSize(file: Blob): Promise<{ width: number; height: number }> {
  try {
    const bmp = await createImageBitmap(file);
    const out = { width: bmp.width, height: bmp.height };
    bmp.close();
    return out;
  } catch {
    throw new ApiError(400, 'bad_image', 'That file is not an image FOLO can decode.', 'Use a JPEG, PNG or WebP.');
  }
}

/** Predictions for an image with known ground truth: misses, jitter, confusions and low-score false positives. */
export function predictBoxes(truth: Box[], W: number, H: number, quality: number, seed: number, classes: number): Scored[] {
  const r = rng(seed);
  const q = Math.max(0, Math.min(1, quality));
  const out: Scored[] = [];
  for (const t of truth) {
    if (r() > 0.25 + 0.75 * Math.min(1, q * 1.25)) continue;
    const w = t.x2 - t.x1;
    const h = t.y2 - t.y1;
    const j = (1 - q) * 0.22;
    const nx1 = t.x1 + w * j * (r() - 0.5) * 2;
    const ny1 = t.y1 + h * j * (r() - 0.5) * 2;
    const nx2 = t.x2 + w * j * (r() - 0.5) * 2;
    const ny2 = t.y2 + h * j * (r() - 0.5) * 2;
    const cls = r() < (1 - q) * 0.25 ? Math.floor(r() * classes) : t.class;
    out.push(clampBox({ class: cls, x1: nx1, y1: ny1, x2: nx2, y2: ny2, score: Math.min(0.99, 0.3 + 0.62 * q + (r() - 0.5) * 0.3) }, W, H));
  }
  const fps = Math.round((1 - q) * 4 * r() + r() * 1.5);
  for (let i = 0; i < fps; i++) {
    const w = range(r, W * 0.04, W * 0.2);
    const h = w * range(r, 0.6, 1.5);
    const x1 = range(r, 0, W - w);
    const y1 = range(r, 0, H - h);
    out.push(clampBox({ class: Math.floor(r() * classes), x1, y1, x2: x1 + w, y2: y1 + h, score: range(r, 0.04, 0.42) }, W, H));
  }
  return out.sort((a, b) => b.score - a.score);
}

/** Adds lower-scored, slightly shifted copies of some boxes (IoU ≈ 0.75–0.9 with the original). */
function withDuplicates(boxes: Scored[], W: number, H: number, seed: number): Scored[] {
  const r = rng(seed);
  const out = [...boxes];
  for (const b of boxes) {
    if (b.score < 0.3 || r() > 0.55) continue;
    const w = b.x2 - b.x1;
    const h = b.y2 - b.y1;
    const j = () => (r() - 0.5) * 0.08;
    out.push(clampBox({ ...b, x1: b.x1 + w * j(), y1: b.y1 + h * j(), x2: b.x2 + w * j(), y2: b.y2 + h * j(), score: b.score * range(r, 0.45, 0.8) }, W, H));
  }
  return out.sort((a, b) => b.score - a.score);
}

/** Plausible boxes for an uploaded image the mock knows nothing about. */
function inventBoxes(W: number, H: number, seed: number, classes: number): Scored[] {
  const r = rng(seed);
  const n = 3 + Math.floor(r() * 6);
  const out: Scored[] = [];
  for (let i = 0; i < n; i++) {
    const w = range(r, W * 0.08, W * 0.35);
    const h = range(r, H * 0.1, H * 0.45);
    const x1 = range(r, 0, W - w);
    const y1 = range(r, 0, H - h);
    out.push(clampBox({ class: Math.floor(r() * Math.min(classes, 12)), x1, y1, x2: x1 + w, y2: y1 + h, score: i < 3 ? range(r, 0.55, 0.96) : range(r, 0.05, 0.5) }, W, H));
  }
  return out.sort((a, b) => b.score - a.score);
}

function clampBox<T extends Scored>(b: T, W: number, H: number): T {
  const x1 = Math.max(0, Math.min(W, b.x1));
  const x2 = Math.max(0, Math.min(W, b.x2));
  const y1 = Math.max(0, Math.min(H, b.y1));
  const y2 = Math.max(0, Math.min(H, b.y2));
  return { ...b, x1: round(Math.min(x1, x2), 1), y1: round(Math.min(y1, y2), 1), x2: round(Math.max(x1, x2), 1), y2: round(Math.max(y1, y2), 1), score: round(b.score, 4) };
}
