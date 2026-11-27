// The mock server (./index): its state and the request gate every endpoint answers through (`call`: latency, the
// scenarios that fail or stall, the training queue's worker), with the System, Datasets and Models endpoints. Runs and
// cloud compute are in ./trainingApi; agent labeling, benchmarks and inference in ./index.

import { ApiError, f16FileName, onnxFileName } from '../client';
import type {
  BenchmarkReport,
  BenchmarkRequest,
  Capabilities,
  ClassMetrics,
  CloudConnection,
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
  Model,
  ModelEval,
  OnnxSwitches,
  Split,
  SystemInfo,
  ThroughputTable,
} from '../types';
import { round } from './boxes';
import { image, imageIndex, indices, issues, specs, stats, toDataset, total, type MockDatasetSpec } from './datasets';
import { synthEvaluation, toModelEval } from './evaluation';
import { MockLabeling, mockAgents } from './label';
import { MockBench } from './bench';
import { BENCH_FEATURE, BENCH_GENERATED_FEATURE } from '../../bench/rules';
import { renderImage } from './render';
import { hash, smoothNoise } from './rng';
import { GPU_NAME, archInfo, bestOf, epochAt, progressAt, toRun, type RunProfile } from './runs';
import { COCO80 } from './coco';
import { seedConnection } from './cloud';
import { CPU_NAME, MockThroughput } from './throughput';
import { LABEL_FEATURE } from '../../label';
import { type Scenario, STALE_AFTER, capabilities } from './scenario';
import { DAY, profiles } from './runSeed';

/** What `folo weights list` says of folo-n-coco (folo_api::weights::PUBLISHED), its `Model.about`. */
const COCO_ABOUT = 'folo-n trained from scratch on COCO train2017 at 640 px (EMA weights, f32, unfused: fine-tune or predict); examples/coco';

export const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/**
 * The mock server's state that more than one of its classes reads or changes: MockServer, MockTrainingApi
 * (./trainingApi) and MockApi (./index). The fields without a value here are set by MockServer's constructor, in
 * this order.
 */
export class MockState {
  t0!: number;
  runs!: RunProfile[];
  datasets!: MockDatasetSpec[];
  nextId = 1;
  /**
   * The training queue: ids of the runs waiting for the one training slot, front first (docs/API.md, POST
   * /api/runs/{id}/queue). Runs train one at a time; the worker takes the front run once the slot is free.
   */
  queue!: string[];
  throughput!: MockThroughput;
  /** Runs created through the API: while one is queued or running, a benchmark probe answers `409 run_active`. */
  created = new Set<string>();
  /** Agent labeling: its jobs, runs and streams (./label). */
  label!: MockLabeling;
  /** Vision benchmarks: benchmarks, entries, loops and their streams (./bench). */
  bench!: MockBench;
  /** Cloud connections. A secret is never kept: only its hint (`••••a1b2`), as the API answers it. */
  connections!: CloudConnection[];
  /** Whether the server has the `cloud` feature (`/api/cloud`, `TrainConfig.compute`); older servers do not. */
  cloudOn!: boolean;
  /** `gpu-unplugged`: the second adapter is gone once a run asked for it. */
  unplugged = false;
  /** `amp-refused`: set by the refused run; `/api/system` then says what the restarted server can do. */
  ampRefused = false;
}

export class MockServer {
  readonly mock = true;
  /** The state the endpoint classes extending this one share with it (see MockState). */
  protected readonly state = new MockState();
  #extraModels: Model[] = [];
  #images = new Map<string, Promise<string>>();
  #probing = false;

  constructor(
    readonly scenario: Scenario = 'normal',
    readonly latency = 140,
  ) {
    this.state.t0 = Date.now();
    this.state.runs = scenario === 'empty' ? [] : profiles(this.state.t0, scenario);
    this.state.datasets = scenario === 'empty' ? [] : [...specs];
    this.state.throughput = new MockThroughput(this.state.t0);
    this.state.label = new MockLabeling(this.state.t0, scenario !== 'empty', scenario === 'all-agents');
    this.state.bench = new MockBench(this.state.t0, {
      models: () => this.allModels(),
      datasets: () => this.state.datasets,
      labeling: this.state.label,
      agents: () => mockAgents(this.scenario === 'all-agents'),
    });
    this.state.bench.designFails = scenario === 'bench-agent-fails';
    this.state.cloudOn = !['cloud-off', 'no-throughput'].includes(scenario);
    this.state.connections = this.state.cloudOn && !['empty', 'cloud-none'].includes(scenario) ? [seedConnection(this.state.t0, scenario === 'cloud-trouble')] : [];
    this.state.queue = this.state.runs
      .filter((p) => p.startedAt === null && !p.terminal)
      .sort((a, b) => a.createdAt - b.createdAt)
      .map((p) => p.id);
  }


  /**
   * The worker: once no run holds the training slot (running, stopping, or taken and still being built), it takes
   * the front of the queue, which starts 1.5 s later (at the earliest 1.5 s after it was created).
   */
  protected schedule(now: number) {
    if (!this.state.queue.length) return;
    // Cloud runs train elsewhere: they never hold the slot.
    const busy = this.state.runs.some(
      (p) => !p.cloud && !this.state.queue.includes(p.id) && ((p.startedAt !== null && p.startedAt > now) || ['running', 'stopping'].includes(progressAt(p, now).status)),
    );
    if (busy) return;
    const p = this.state.runs.find((r) => r.id === this.state.queue[0]);
    this.state.queue.shift();
    if (p) p.startedAt = Math.max(now + 1500, p.createdAt + 1500);
  }

  /** Runs `fn` after a simulated network delay (`latency` overrides the default; a mock built with 0 never waits). */
  protected async call<T>(fn: () => T, latency?: number): Promise<T> {
    if (this.scenario === 'loading') return new Promise<T>(() => {});
    if (this.latency > 0 && (latency ?? this.latency) > 0) await sleep((latency ?? this.latency) * (0.7 + 0.6 * Math.random()));
    if (this.scenario === 'error' || ((this.scenario === 'stale' || this.scenario === 'stale-lists') && Date.now() - this.state.t0 > STALE_AFTER))
      throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
    this.schedule(Date.now());
    return fn();
  }

  protected spec(id: string): MockDatasetSpec {
    const s = this.state.datasets.find((d) => d.id === id);
    if (!s) throw new ApiError(404, 'dataset_not_found', `No dataset with id ${id}.`, 'List datasets with GET /api/datasets.');
    return s;
  }

  // ---------- system ----------

  health(): Promise<Health> {
    return this.call(() => ({ status: 'ok' as const, version: '0.1.0', api: 'v1', mock: true }), 20);
  }

  /** Whether a run trains on this machine now (cloud runs use another's GPU). */
  #trainingHere(): boolean {
    return this.state.runs.some((p) => !p.cloud && progressAt(p, Date.now()).status === 'running');
  }

  protected gpus(): SystemInfo['gpus'] {
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
    if (this.scenario === 'gpu-unplugged' && !this.state.unplugged)
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

  protected capabilities(): Capabilities {
    return capabilities(this.state.ampRefused ? 'tensor-cores-off' : this.scenario);
  }

  /** Whether this server labels: not `folo serve --mock` (`no-labeling`), nor a server from before it (`no-throughput`). */
  protected labeling() {
    return this.scenario !== 'no-labeling' && this.scenario !== 'no-throughput';
  }

  /** Whether this server has them: not `folo serve --mock` (`no-bench`, `no-labeling`), nor a server from before them. */
  protected benching() {
    return this.scenario !== 'no-bench' && this.scenario !== 'no-labeling' && this.scenario !== 'no-throughput';
  }

  system(): Promise<SystemInfo> {
    return this.call(() => {
      const t = Date.now() / 1000;
      const pool = this.#gpuPool();
      return {
        cpu: { name: CPU_NAME, threads: 32, util: round(0.27 + 0.08 * smoothNoise(3, t / 4), 3) },
        memory: { used_bytes: Math.round((21.4 + 0.3 * smoothNoise(4, t / 9)) * 2 ** 30), total_bytes: 64 * 2 ** 30 },
        gpus: this.gpus(),
        features: [
          ...(this.scenario === 'no-throughput'
            ? ['export_onnx_switches', 'queue_control', 'weights_onnx']
            : ['export_onnx_switches', 'export_dynamic_axes', 'queue_control', 'weights_onnx', 'throughput', 'run_estimate', 'benchmark', 'run_cli']),
          // The in-browser runtime in this build (the name requested of the server; lib/inference `wasmFromSystem`).
          ...(this.scenario === 'no-wasm' ? [] : ['web_wasm']),
          ...(this.labeling() ? [LABEL_FEATURE] : []),
          ...(this.benching() ? [BENCH_FEATURE, BENCH_GENERATED_FEATURE] : []),
          ...(this.state.cloudOn ? ['cloud'] : []),
        ],
        capabilities: this.capabilities(),
        ...(pool ? { gpu_pool: pool } : {}),
      };
    }, 60);
  }

  /** A server from before these endpoints: an unknown route. */
  protected noRoute(path: string): ApiError {
    return new ApiError(404, 'not_found', `No route for ${path}.`);
  }

  throughput(): Promise<ThroughputTable> {
    return this.call(() => {
      if (this.scenario === 'no-throughput') throw this.noRoute('GET /api/system/throughput');
      return this.state.throughput.table();
    }, 80);
  }

  /** About 1.5 s, the shortest a real probe takes (docs/API.md: the budget, 1–10 s, plus setup). */
  async benchmark(body: BenchmarkRequest = {}): Promise<BenchmarkReport> {
    if (this.scenario === 'no-throughput') return this.call(() => Promise.reject(this.noRoute('POST /api/system/benchmark')));
    const busy = () => {
      const now = Date.now();
      const active = this.state.runs.find((p) => this.state.created.has(p.id) && !p.cloud && ['queued', 'running', 'stopping'].includes(progressAt(p, now).status));
      if (this.#probing) return new ApiError(409, 'run_active', 'A benchmark probe is already running.', 'Wait for it to finish.');
      if (active) return new ApiError(409, 'run_active', `Run ${active.name} is ${progressAt(active, now).status}.`, 'Measure when it has finished, or stop it first.');
      return null;
    };
    const err = busy();
    if (err) return this.call(() => Promise.reject(err), 60);
    this.#probing = true;
    const t = Date.now();
    try {
      if (this.scenario === 'loading') return await new Promise<BenchmarkReport>(() => {});
      if (this.latency > 0) await sleep(1500);
      if (this.scenario === 'error') throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
      return this.state.throughput.probe(body, Date.now(), (Date.now() - t) / 1000);
    } finally {
      this.#probing = false;
    }
  }

  // ---------- datasets ----------

  datasets(): Promise<Dataset[]> {
    return this.call(() => this.state.datasets.map((s) => toDataset(s, this.state.t0, false)));
  }

  dataset(id: string): Promise<Dataset> {
    return this.call(() => toDataset(this.spec(id), this.state.t0, true));
  }

  createDataset(body: CreateDataset): Promise<Dataset> {
    return this.call(() => {
      // As on the server: surrounding quotes and whitespace are dropped, a relative path is refused, and a folder that
      // is already registered (by its path, whatever the spelling) answers with the existing dataset, its name kept.
      const path = body.path.trim().replace(/^(["'])(.*)\1$/s, '$2').trim();
      if (!/^([A-Za-z]:[\\/]|[\\/])/.test(path))
        throw new ApiError(400, 'invalid_body', `'${path}' is not a full path`, "use the folder's full path on the machine running folo serve, such as D:/data/my-dataset");
      const key = (p: string) => p.trim().replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase();
      const known = this.state.datasets.find((d) => key(d.path) === key(path));
      if (known) return toDataset(known, this.state.t0, true);
      throw new ApiError(400, 'dataset_not_found_on_disk', `No YOLO or COCO dataset found at ${path}.`, 'The mock server cannot read your disk; connect to folo serve to register folders.');
    }, 400);
  }

  createSyntheticDataset(body: CreateSyntheticDataset): Promise<Dataset> {
    return this.call(() => {
      // As the real server: a count that is no whole number fails its body's decoding (a u64), a 422; one outside its
      // limits is a 400; never clamped or rounded.
      const n = body.images;
      if (!Number.isInteger(n) || n < 0) throw new ApiError(422, 'invalid_body', 'images: invalid type: expected u64');
      if (n < 1 || n > 100_000) throw new ApiError(400, 'invalid_body', 'images must be in 1..=100000');
      const train = Math.round(n * 0.8);
      const classes = (body.classes.length ? body.classes : ['circle', 'square', 'triangle']).slice(0, 6);
      const spec: MockDatasetSpec = {
        ...(specs.find((s) => s.id === 'ds_shapes') as MockDatasetSpec),
        id: `ds_${(hash(body.name + this.state.nextId++) & 0xffff).toString(16).padStart(4, '0')}`,
        name: body.name,
        path: `D:/data/${body.name}`,
        classes,
        classWeights: classes.map(() => 1),
        classShape: classes.map(() => 'square' as const),
        splits: { train, val: n - train, test: 0 },
        width: body.imgsz,
        height: body.imgsz,
        seed: body.seed,
        createdAgoS: (this.state.t0 - Date.now()) / 1000,
      };
      this.state.datasets.unshift(spec);
      return toDataset(spec, this.state.t0, true);
    }, 900);
  }

  datasetImages(id: string, q: DatasetImagesQuery): Promise<DatasetImages> {
    return this.call(() => {
      const s = this.spec(id);
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
    return this.call(() => {
      const all = issues(this.spec(id));
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
      const s = this.spec(datasetId);
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

  deleteDataset(id: string, opts: { files?: boolean } = {}): Promise<void> {
    return this.call(() => {
      // As folo serve --mock: no data dir, so no folder is ever this server's to delete.
      if (opts.files) throw new ApiError(400, 'invalid_query', "this dataset's folder is not one this server wrote; its files are never deleted");
      this.spec(id);
      this.state.datasets = this.state.datasets.filter((d) => d.id !== id);
    });
  }

  // ---------- models ----------

  protected allModels(): Model[] {
    const now = Date.now();
    const out: Model[] = [...this.#extraModels];
    for (const p of this.state.runs) {
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
        created: new Date(this.state.t0 - 20 * DAY).toISOString(),
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
        created: new Date(this.state.t0 - 9 * DAY).toISOString(),
        run_id: null,
        metrics: { map50: 0.455, map50_95: 0.314, precision: 0.583, recall: 0.427, best_f1_conf: null },
        head: 'o2m',
        trained_with: { amp: false, imgsz: 640, epochs: 50 },
        precision: 'f32',
        fused: false,
        source: 'published',
        about: COCO_ABOUT,
      });
      // An early checkpoint of shapes-n (5 epochs), imported as a file: on Shapes val its confidences are 90% right only
      // where it keeps almost nothing, so as a loop's first teacher it labels too little (the bench mock's failed loop,
      // and New loop's warning).
      out.push({
        id: 'm_shapes_e5',
        name: 'shapes-n-early',
        arch: 'folo-n',
        imgsz: 640,
        classes: ['circle', 'square', 'triangle'],
        params: archInfo['folo-n'].params,
        flops_g: archInfo['folo-n'].flops,
        bytes: Math.round(archInfo['folo-n'].params * 2.06),
        created: new Date(this.state.t0 - 10 * DAY).toISOString(),
        run_id: null,
        metrics: { map50: 0.341, map50_95: 0.198, precision: 0.412, recall: 0.377, best_f1_conf: 0.12 },
        head: 'o2m',
        trained_with: { amp: false, imgsz: 640, epochs: 5 },
        precision: 'f32',
        fused: false,
        source: 'file',
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
        created: new Date(this.state.t0 - 4 * DAY).toISOString(),
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
    return this.call(() => this.allModels());
  }

  model(id: string): Promise<Model> {
    return this.call(() => {
      const m = this.allModels().find((x) => x.id === id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${id}.`);
      return m;
    });
  }

  modelEval(id: string): Promise<ModelEval> {
    return this.call(() => {
      const m = this.allModels().find((x) => x.id === id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${id}.`);
      const p = m.run_id ? this.state.runs.find((r) => r.id === m.run_id) : undefined;
      let perClass: ClassMetrics[] | null = null;
      let instances: number[] | null = null;
      let images = 500;
      if (p) {
        const best = bestOf(p, Math.floor(progressAt(p, Date.now()).done));
        perClass = best ? epochAt(p, best.epoch - 1).val.per_class : null;
      }
      // The run's dataset; an imported model's, the first with its classes (as a real server picks it).
      const same = (a: readonly string[]) => a.length === m.classes.length && a.every((c, i) => c === m.classes[i]);
      const ds = p ? this.state.datasets.find((d) => d.id === p.config.dataset_id) : this.state.datasets.find((d) => same(d.classes));
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
    return this.call(() => {
      const m = this.allModels().find((x) => x.id === id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${id}.`);
      return new File([], onnxFileName(m.name, !!switches.half, !!switches.decode), { type: 'application/octet-stream' });
    }, 1200);
  }

  /** `/weights?format=f16`, named as the server names it; empty, as `onnxFile`. */
  f16File(id: string): Promise<File> {
    return this.call(() => {
      const m = this.allModels().find((x) => x.id === id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${id}.`);
      return new File([], f16FileName(m.name), { type: 'application/octet-stream' });
    }, 1200);
  }

  exportModel(id: string, format: ExportFormat, switches?: OnnxSwitches): Promise<ExportResult> {
    return this.call(() => {
      const m = this.allModels().find((x) => x.id === id);
      if (!m) throw new ApiError(404, 'model_not_found', `No model with id ${id}.`);
      const on = {
        decode: !!switches?.decode,
        dynamic: !!switches?.dynamic,
        dynamic_batch: switches?.dynamic_batch,
        dynamic_hw: switches?.dynamic_hw,
        half: !!switches?.half,
      };
      if (format !== 'onnx' && (on.decode || on.dynamic || on.half || on.dynamic_batch !== undefined || on.dynamic_hw !== undefined))
        throw new ApiError(400, 'invalid_body', 'decode, dynamic, dynamic_batch, dynamic_hw and half apply to ONNX exports only.');
      if (on.dynamic && (on.dynamic_batch === false || on.dynamic_hw === false))
        throw new ApiError(400, 'invalid_body', '`dynamic: true` cannot be combined with `dynamic_batch: false` or `dynamic_hw: false`');
      // The real server's names (docs/API.md): `<name>-f16.folo`, `<name>.onnx`, `-decoded` with `decode`, `-f16` with `half`.
      const file = format === 'onnx' ? `${m.name}${on.decode ? '-decoded' : ''}${on.half ? '-f16' : ''}.onnx` : `${m.name}-f16.folo`;
      const perParam = format === 'onnx' ? (on.half ? 2.05 : 4.1) : 2.03;
      return { path: `D:/folo/exports/${file}`, bytes: Math.round(m.params * perParam) };
    }, 1200);
  }
}
