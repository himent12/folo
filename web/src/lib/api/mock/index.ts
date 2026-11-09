// In-browser mock of the whole API (docs/API.md), used with `?mock=1` or when the server is unreachable.
// Deterministic: everything derives from ids and the clock, so screenshots are comparable between rounds.
//
// Scenarios (`?scenario=`): `normal` (default), `empty` (nothing registered yet), `error` (every call fails),
// `loading` (calls never resolve), `oom` (server inference runs out of GPU memory for every model but folo-n),
// `gpu-unplugged` (a second GPU is listed until a run asks for it, then `404 device_not_found`), `short-run` (adds a
// finished 5-epoch run the trainer warns is too short to learn from), `tensor-cores-off` and `no-tensor-cores` (no
// mixed-precision training: the server's `FOLO_MOCK_TENSOR_CORES=off` / `none`), for screenshots of those states.

import { ApiError, type Api, type RunEventHandlers } from '../client';
import type {
  Arch,
  Box,
  Capabilities,
  ClassMetrics,
  CreateDataset,
  CreateSyntheticDataset,
  Dataset,
  DatasetImages,
  DatasetImagesQuery,
  DatasetIssues,
  DatasetIssuesQuery,
  ExportFormat,
  ExportResult,
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
  Run,
  RunLogs,
  RunMetrics,
  RunSamples,
  Sample,
  SystemInfo,
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
  epochLog,
  history,
  loopStart,
  optimizerSteps,
  progressAt,
  startLogs,
  stepAt,
  toRun,
  type RunProfile,
} from './runs';
import { COCO80 } from './coco';

export type Scenario = 'normal' | 'empty' | 'error' | 'loading' | 'oom' | 'gpu-unplugged' | 'short-run' | 'tensor-cores-off' | 'no-tensor-cores';

const MIN = 60_000;
const HOUR = 60 * MIN;
const DAY = 24 * HOUR;

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
              'train without amp (f32), or on such a GPU (`folo train --amp` asks for them; start `folo serve --tensor-cores`)',
          },
  };
}
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
      // Fine-tuned from shapes-n (the Model page's lineage line).
      config: config('fasteners-n', 'ds_fasteners', 'folo-n', 150, { weights: 'm_shapes_n' }),
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
      config: config('shapes-s', 'ds_shapes', 'folo-s', 100),
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
  return list.map((p) => ({ ...p, classNames: names(p.config.dataset_id) }));
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

  constructor(
    readonly scenario: Scenario = 'normal',
    readonly latency = 140,
  ) {
    this.#t0 = Date.now();
    this.#runs = scenario === 'empty' ? [] : profiles(this.#t0, scenario);
    this.#datasets = scenario === 'empty' ? [] : [...specs];
  }

  /** Runs `fn` after a simulated network delay (`latency` overrides the default; a mock built with 0 never waits). */
  async #call<T>(fn: () => T, latency?: number): Promise<T> {
    if (this.scenario === 'loading') return new Promise<T>(() => {});
    if (this.latency > 0 && (latency ?? this.latency) > 0) await sleep((latency ?? this.latency) * (0.7 + 0.6 * Math.random()));
    if (this.scenario === 'error')
      throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
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

  #gpus(): SystemInfo['gpus'] {
    const t = Date.now() / 1000;
    const training = this.#runs.some((p) => progressAt(p, Date.now()).status === 'running');
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

  system(): Promise<SystemInfo> {
    return this.#call(() => {
      const t = Date.now() / 1000;
      return {
        cpu: { name: 'AMD Ryzen 9 7950X 16-Core Processor', threads: 32, util: round(0.27 + 0.08 * smoothNoise(3, t / 4), 3) },
        memory: { used_bytes: Math.round((21.4 + 0.3 * smoothNoise(4, t / 9)) * 2 ** 30), total_bytes: 64 * 2 ** 30 },
        gpus: this.#gpus(),
        features: ['export_onnx_switches'],
        capabilities: capabilities(this.scenario),
      };
    }, 60);
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
        metrics: ep ? { map50: ep.val.map50, map50_95: ep.val.map50_95, precision: ep.val.precision, recall: ep.val.recall } : null,
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
        metrics: { map50: 0.611, map50_95: 0.448, precision: 0.672, recall: 0.571 },
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
        const ds = this.#datasets.find((d) => d.id === p.config.dataset_id);
        if (ds) {
          const share = ds.splits.val / Math.max(1, total(ds));
          instances = stats(ds).class_counts.map((c) => c * share);
          images = ds.splits.val;
        }
      }
      return toModelEval(synthEvaluation({ seed: m.id, classes: m.classes, metrics: m.metrics, perClass, instances }), images);
    }, 180);
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
      return this.#runs.map((p) => toRun(p, now)).sort((a, b) => b.created.localeCompare(a.created));
    });
  }

  run(id: string): Promise<Run> {
    return this.#call(() => toRun(this.#profile(id), Date.now()));
  }

  createRun(c: TrainConfig): Promise<Run> {
    return this.#call(() => {
      const ds = this.#spec(c.dataset_id);
      // `gpu-unplugged`: the second adapter goes away as soon as a run asks for it.
      if (this.scenario === 'gpu-unplugged' && c.device === 'gpu:1') this.#unplugged = true;
      const gpu = /^gpu:(\d+)$/.exec(c.device);
      if (gpu && !this.#gpus().some((g) => g.ordinal === Number(gpu[1]))) throw this.#noDevice(c.device);
      // The real server's body error for a head it does not know (serde: `o2m` or `o2o`; missing means `o2m`).
      if (c.head !== undefined && c.head !== 'o2m' && c.head !== 'o2o')
        throw new ApiError(400, 'invalid_body', `field \`head\`: unknown variant \`${String(c.head)}\`, expected \`o2m\` or \`o2o\``);
      const now = Date.now();
      const id = `run_${(hash(c.name + now + this.#nextId++) & 0xffff).toString(16).padStart(4, '0')}`;
      const sizeIdx = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'].indexOf(c.arch);
      const p: RunProfile = {
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
        startedAt: now + 1500,
        modelId: `m_${id.slice(4)}`,
        device: c.device === 'cpu' ? 'cpu (AMD Ryzen 9 7950X)' : DEVICE,
      };
      this.#runs.unshift(p);
      return toRun(p, now);
    }, 300);
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
      const startedAt = loopStart(p, now);
      p.startedAt = startedAt;
      p.loop = false;
      p.terminal = { status: 'stopped', atEpoch: done, at: now + 1200, stoppingUntil: now + 1200 };
      return toRun(p, now);
    });
  }

  deleteRun(id: string): Promise<void> {
    return this.#call(() => {
      const p = this.#profile(id);
      const status = progressAt(p, Date.now()).status;
      if (status === 'running' || status === 'stopping')
        throw new ApiError(409, 'run_active', `Run ${p.name} is ${status}.`, 'Stop it first.');
      this.#runs = this.#runs.filter((r) => r.id !== id);
    });
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
      // A fixed set of up to 16 validation images, spread evenly over the split.
      const n = Math.min(16, ds.splits.val);
      for (let i = 0; i < n; i++) {
        const index = ds.splits.train + Math.floor((i * ds.splits.val) / n);
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
    let last = progressAt(p, Date.now());
    let lastStart = loopStart(p, Date.now());
    let lastStep = Math.floor(last.done * p.stepsPerEpoch);
    let lastEpoch = Math.floor(last.done);
    const t0 = setTimeout(() => {
      h.connection?.('open');
      h.status?.({ status: last.status, error: p.terminal?.error ?? null });
    }, 30);
    const timer = setInterval(() => {
      const now = Date.now();
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
      if (cur.status !== last.status) h.status?.({ status: cur.status, error: p.terminal?.error ?? null });
      if (cur.status === 'stopped' && last.status !== 'stopped')
        h.log?.({ t: new Date(now).toISOString(), level: 'warn', message: `stopped by user at epoch ${Math.floor(cur.done) + 1}; last checkpoint kept` });
      last = cur;
    }, 100);
    return () => {
      clearTimeout(t0);
      clearInterval(timer);
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
    if (!p || p.startedAt === null) return [];
    const now = Date.now();
    const start = loopStart(p, now);
    const ds = specs.find((s) => s.id === p.config.dataset_id);
    const lines = startLogs(p, start, ds?.name ?? '', ds?.splits.train ?? 0);
    const done = Math.floor(progressAt(p, now).done);
    for (let e = 0; e < done; e++) lines.push(...epochLog(p, epochAt(p, e), start));
    if (p.terminal?.status === 'failed') lines.push({ t: new Date(start + done * p.epochSeconds * 1000).toISOString(), level: 'error', message: p.terminal.error ?? 'failed' });
    return lines;
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
