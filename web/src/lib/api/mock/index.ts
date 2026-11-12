// In-browser mock of the whole API (docs/API.md), used only when asked for with `?mock=1`.
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
// `POST /api/runs` refuses it with `501 not_supported`, and `/api/system` says so from then on), `no-labeling` (`folo serve
// --mock`: no `agent_labeling` feature, and the labeling endpoints answer `501 not_implemented`), `all-agents` (both
// labeling agents installed; elsewhere Codex is not), and for cloud compute (docs/API.md, Mock mode → Cloud: one RunPod
// connection, a finished cloud run): `cloud-none` (no connection, the server's `FOLO_MOCK_CLOUD=none`), `cloud-off` (no
// `cloud` feature: `FOLO_MOCK_CLOUD=off`), `cloud-trouble` (the connection's key was revoked, so its check fails and the
// termination of a finished run's instance cannot be confirmed), `cloud-training` (a cloud run trains on a rented RTX
// 4090) and `cloud-upload` (a cloud run uploads its dataset), for screenshots of those states. Vision benchmarks
// (./bench, ./benchSeed): `no-bench` (no `vision_bench` feature; the endpoints answer `501 not_implemented`, as
// `folo serve --mock` does, which `no-labeling` stands for too), `bench-agent-fails` (a design request's answer is not
// a plan: `502 agent_failed`), `bench-loop-gen0` (a loop on Fasteners val that has just started: generation 0, its
// teacher's entry waiting for the GPU) and `bench-failed-only` (the fresh benchmark with one failed entry: its plan
// still open).
//
// MockApi (here: agent labeling, vision benchmarks, inference) extends ./trainingApi (runs, cloud compute), which
// extends ./server (the state, the request gate, system, datasets, models). The seeded runs: ./runSeed; the scenarios'
// type and capabilities: ./scenario.

import { ApiError, type Api, type BenchEventHandlers, type LabelEventHandlers } from '../client';
import type {
  Bench,
  BenchCatalog,
  BenchDesign,
  BenchEntry,
  BenchEntryImagePage,
  BenchEntryImagesQuery,
  BenchImageOptions,
  BenchImagePage,
  BenchImagesQuery,
  BenchLoop,
  BenchPrediction,
  BenchResult,
  Box,
  CreateBench,
  DesignBench,
  StartBenchEntry,
  StartBenchLoop,
  UpdateBench,
  CreateLabelJob,
  Dataset,
  LabelAgent,
  LabelDatasetRequest,
  LabelImage,
  LabelImagePage,
  LabelImagesAdded,
  LabelImagesQuery,
  LabelJob,
  LogLine,
  PredictBox,
  PredictInput,
  PredictOptions,
  Prediction,
  StartLabeling,
  UpdateLabelJob,
} from '../types';
import { imageSize, inventBoxes, predictBoxes, round, withDuplicates, type Scored } from './boxes';
import { image, imageIndex, toDataset } from './datasets';
import { mockAgents } from './label';
import { MockBench } from './bench';
import { seedBenches } from './benchSeed';
import { hash, rng } from './rng';
import { archInfo } from './runs';
import { uploadInChunks, type UploadProgress } from '../upload';
import { MockTrainingApi } from './trainingApi';
import { sleep } from './server';
import { STALE_AFTER } from './scenario';

export type { Scenario } from './scenario';
export { capabilities } from './scenario';

export class MockApi extends MockTrainingApi implements Api {
  // ---------- agent labeling ----------

  /** A labeling call: `folo serve --mock` answers 501, an older server has no such route. */
  async #labelCall<T>(path: string, fn: () => T | Promise<T>, latency?: number): Promise<T> {
    return this.call(() => {
      if (this.scenario === 'no-throughput') throw this.noRoute(path);
      if (!this.labeling()) throw new ApiError(501, 'not_implemented', 'Agent labeling needs a real server.', 'Start folo serve without --mock to label images.');
      return fn();
    }, latency);
  }

  labelAgents(): Promise<LabelAgent[]> {
    return this.#labelCall('GET /api/label/agents', () => mockAgents(this.scenario === 'all-agents'), 80);
  }

  labelJobs(): Promise<LabelJob[]> {
    return this.#labelCall('GET /api/label/jobs', () => this.state.label.jobs(Date.now()));
  }

  labelJob(id: string): Promise<LabelJob> {
    return this.#labelCall(`GET /api/label/jobs/${id}`, () => this.state.label.job(id, Date.now()));
  }

  createLabelJob(body: CreateLabelJob): Promise<LabelJob> {
    return this.#labelCall('POST /api/label/jobs', () => this.state.label.create(body, Date.now()), 300);
  }

  updateLabelJob(id: string, body: UpdateLabelJob): Promise<LabelJob> {
    return this.#labelCall(`PUT /api/label/jobs/${id}`, () => this.state.label.update(id, body, Date.now()));
  }

  deleteLabelJob(id: string): Promise<void> {
    return this.#labelCall(`DELETE /api/label/jobs/${id}`, () => this.state.label.delete(id, Date.now()));
  }

  labelImages(jobId: string, q: LabelImagesQuery = {}): Promise<LabelImagePage> {
    return this.#labelCall(`GET /api/label/jobs/${jobId}/images`, () => this.state.label.images(jobId, q, Date.now()), 90);
  }

  addLabelImages(jobId: string, files: File[], onprogress?: (p: UploadProgress) => void): Promise<LabelImagesAdded> {
    // One request per chunk, as the server takes them (lib/api/upload): progress moves as a real upload's does.
    return this.#labelCall(`POST /api/label/jobs/${jobId}/images`, async () => {
      const send = async (chunk: File[]) => {
        const part = await this.state.label.upload(jobId, chunk, Date.now());
        if (this.latency > 0) await sleep(this.latency * 2);
        return part;
      };
      return (await uploadInChunks(files, send, onprogress)) ?? { added: [], skipped: [], job: this.state.label.job(jobId, Date.now()) };
    });
  }

  importLabelImages(jobId: string, path: string): Promise<LabelImagesAdded> {
    return this.#labelCall(`POST /api/label/jobs/${jobId}/images`, () => this.state.label.importPath(jobId, path, Date.now()), 500);
  }

  updateLabelImage(jobId: string, imageId: string, boxes: Box[]): Promise<LabelImage> {
    return this.#labelCall(`PUT /api/label/jobs/${jobId}/images/${imageId}`, () => this.state.label.updateImage(jobId, imageId, boxes, Date.now()));
  }

  reviewLabelImage(jobId: string, imageId: string, reviewed: boolean): Promise<LabelImage> {
    return this.#labelCall(`PUT /api/label/jobs/${jobId}/images/${imageId}`, () => this.state.label.reviewImage(jobId, imageId, reviewed, Date.now()));
  }

  deleteLabelImage(jobId: string, imageId: string): Promise<void> {
    return this.#labelCall(`DELETE /api/label/jobs/${jobId}/images/${imageId}`, () => this.state.label.deleteImage(jobId, imageId, Date.now()));
  }

  startLabeling(jobId: string, body: StartLabeling): Promise<LabelJob> {
    return this.#labelCall(`POST /api/label/jobs/${jobId}/start`, () => this.state.label.start(jobId, body, Date.now()), 200);
  }

  stopLabeling(jobId: string): Promise<LabelJob> {
    return this.#labelCall(`POST /api/label/jobs/${jobId}/stop`, () => this.state.label.stop(jobId, Date.now()));
  }

  labelLogs(jobId: string, limit = 500): Promise<{ lines: LogLine[] }> {
    return this.#labelCall(`GET /api/label/jobs/${jobId}/logs`, () => this.state.label.logs(jobId, limit, Date.now()), 120);
  }

  createLabelDataset(jobId: string, body: LabelDatasetRequest): Promise<Dataset> {
    return this.#labelCall(
      `POST /api/label/jobs/${jobId}/dataset`,
      () => {
        const spec = this.state.label.dataset(jobId, body, this.state.datasets.map((d) => d.name), Date.now());
        this.state.datasets.unshift(spec);
        return toDataset(spec, this.state.t0, true);
      },
      700,
    );
  }

  async labelImageUrl(jobId: string, imageId: string, thumb?: number): Promise<string> {
    return this.state.label.imageUrl(jobId, imageId, thumb);
  }

  labelEvents(jobId: string, h: LabelEventHandlers): () => void {
    if (this.scenario === 'loading' || this.scenario === 'error' || !this.labeling()) {
      if (this.scenario === 'error') setTimeout(() => h.connection?.('reconnecting'), 500);
      return () => {};
    }
    const unsub = this.state.label.subscribe(jobId, h);
    const t0 = setTimeout(() => {
      h.connection?.('open');
      try {
        h.job?.(this.state.label.job(jobId, Date.now()));
      } catch {
        // A job deleted meanwhile: its stream has nothing to say.
      }
    }, 30);
    let dropped = false;
    const timer = setInterval(() => {
      // `stale`: the server stops, its stream with it.
      if (this.scenario === 'stale' && Date.now() - this.state.t0 > STALE_AFTER) {
        if (!dropped) h.connection?.('reconnecting');
        dropped = true;
        return;
      }
      this.state.label.tick(jobId, Date.now());
    }, 250);
    return () => {
      clearTimeout(t0);
      clearInterval(timer);
      unsub();
    };
  }

  // ---------- vision benchmarks ----------

  /** The demo's benchmarks, made on the first call (they read the models, which read the runs). */
  #benchSeeded = false;
  #benchState(): MockBench {
    if (!this.#benchSeeded) {
      this.#benchSeeded = true;
      if (this.scenario !== 'empty') seedBenches(this.state.bench, this.state.t0, { loopAtGen0: this.scenario === 'bench-loop-gen0', failedOnly: this.scenario === 'bench-failed-only' });
    }
    return this.state.bench;
  }

  /** A benchmark call: `folo serve --mock` answers 501, an older server has no such route. */
  async #benchCall<T>(path: string, fn: (b: MockBench) => T | Promise<T>, latency?: number): Promise<T> {
    return this.call(() => {
      if (this.scenario === 'no-throughput') throw this.noRoute(path);
      if (!this.benching()) throw new ApiError(501, 'not_implemented', 'Vision benchmarks need a real server.', 'Start folo serve without --mock to make benchmarks.');
      return fn(this.#benchState());
    }, latency);
  }

  benchCatalog(): Promise<BenchCatalog> {
    return this.#benchCall('GET /api/bench/catalog', (b) => b.catalog(), 60);
  }

  benches(): Promise<Bench[]> {
    return this.#benchCall('GET /api/bench/benchmarks', (b) => b.list(Date.now()));
  }

  bench(id: string): Promise<Bench> {
    return this.#benchCall(`GET /api/bench/benchmarks/${id}`, (b) => b.get(id, Date.now()));
  }

  createBench(body: CreateBench): Promise<Bench> {
    return this.#benchCall('POST /api/bench/benchmarks', (b) => b.create(body, Date.now()), 600);
  }

  updateBench(id: string, body: UpdateBench): Promise<Bench> {
    return this.#benchCall(`PUT /api/bench/benchmarks/${id}`, (b) => b.update(id, body, Date.now()));
  }

  deleteBench(id: string): Promise<void> {
    return this.#benchCall(`DELETE /api/bench/benchmarks/${id}`, (b) => b.delete(id, Date.now()));
  }

  /** One agent call: answered a few seconds later (a real one takes up to two minutes). */
  designBench(id: string, body: DesignBench): Promise<BenchDesign> {
    return this.#benchCall(
      `POST /api/bench/benchmarks/${id}/design`,
      async (b) => {
        // Checked at once (an unknown benchmark, an agent that is not installed), then the call's time.
        b.design(id, body, Date.now());
        if (this.latency > 0) await sleep(this.scenario === 'bench-agent-fails' ? 2200 : 3800);
        return b.design(id, body, Date.now());
      },
      80,
    );
  }

  benchImages(id: string, q: BenchImagesQuery = {}): Promise<BenchImagePage> {
    return this.#benchCall(`GET /api/bench/benchmarks/${id}/images`, (b) => b.images(id, q, Date.now()), 90);
  }

  async benchImageUrl(id: string, imageId: string, opts: BenchImageOptions = {}): Promise<string> {
    return this.#benchState().imageUrl(id, imageId, opts);
  }

  benchEntries(id: string): Promise<BenchEntry[]> {
    return this.#benchCall(`GET /api/bench/benchmarks/${id}/entries`, (b) => b.entries(id, Date.now()));
  }

  startBenchEntry(id: string, body: StartBenchEntry): Promise<BenchEntry> {
    return this.#benchCall(`POST /api/bench/benchmarks/${id}/entries`, (b) => b.start(id, body, Date.now()), body.contestant.kind === 'external' ? 900 : 250);
  }

  benchEntry(id: string, entryId: string): Promise<BenchEntry> {
    return this.#benchCall(`GET /api/bench/benchmarks/${id}/entries/${entryId}`, (b) => b.entry(id, entryId, Date.now()));
  }

  benchResult(id: string, entryId: string): Promise<BenchResult> {
    return this.#benchCall(`GET /api/bench/benchmarks/${id}/entries/${entryId}/result`, (b) => b.result(id, entryId, Date.now()), 160);
  }

  benchEntryImages(id: string, entryId: string, q: BenchEntryImagesQuery = {}): Promise<BenchEntryImagePage> {
    return this.#benchCall(`GET /api/bench/benchmarks/${id}/entries/${entryId}/images`, (b) => b.entryImages(id, entryId, q, Date.now()), 160);
  }

  stopBenchEntry(id: string, entryId: string): Promise<BenchEntry> {
    return this.#benchCall(`POST /api/bench/benchmarks/${id}/entries/${entryId}/stop`, (b) => b.stopEntry(id, entryId, Date.now()));
  }

  sendBenchPredictions(id: string, entryId: string, predictions: BenchPrediction[]): Promise<BenchEntry> {
    return this.#benchCall(`POST /api/bench/benchmarks/${id}/entries/${entryId}/predictions`, (b) => b.sendPredictions(id, entryId, predictions, Date.now()), 300);
  }

  finishBenchEntry(id: string, entryId: string): Promise<BenchEntry> {
    return this.#benchCall(`POST /api/bench/benchmarks/${id}/entries/${entryId}/finish`, (b) => b.finishEntry(id, entryId, Date.now()), 600);
  }

  deleteBenchEntry(id: string, entryId: string): Promise<void> {
    return this.#benchCall(`DELETE /api/bench/benchmarks/${id}/entries/${entryId}`, (b) => b.deleteEntry(id, entryId, Date.now()));
  }

  benchLoops(id: string): Promise<BenchLoop[]> {
    return this.#benchCall(`GET /api/bench/benchmarks/${id}/loops`, (b) => b.loops(id, Date.now()));
  }

  startBenchLoop(id: string, body: StartBenchLoop): Promise<BenchLoop> {
    return this.#benchCall(`POST /api/bench/benchmarks/${id}/loops`, (b) => b.startLoop(id, body, Date.now()), 300);
  }

  benchLoop(id: string, loopId: string): Promise<BenchLoop> {
    return this.#benchCall(`GET /api/bench/benchmarks/${id}/loops/${loopId}`, (b) => b.loop(id, loopId, Date.now()));
  }

  stopBenchLoop(id: string, loopId: string): Promise<BenchLoop> {
    return this.#benchCall(`POST /api/bench/benchmarks/${id}/loops/${loopId}/stop`, (b) => b.stopLoop(id, loopId, Date.now()));
  }

  deleteBenchLoop(id: string, loopId: string): Promise<void> {
    return this.#benchCall(`DELETE /api/bench/benchmarks/${id}/loops/${loopId}`, (b) => b.deleteLoop(id, loopId, Date.now()));
  }

  benchEvents(id: string, h: BenchEventHandlers): () => void {
    if (this.scenario === 'loading' || this.scenario === 'error' || !this.benching() || this.scenario === 'no-throughput') {
      if (this.scenario === 'error') setTimeout(() => h.connection?.('reconnecting'), 500);
      return () => {};
    }
    const state = this.#benchState();
    const unsub = state.subscribe(id, h);
    const t0 = setTimeout(() => {
      h.connection?.('open');
      try {
        h.bench?.(state.get(id, Date.now()));
      } catch {
        // A benchmark deleted meanwhile: its stream has nothing to say.
      }
    }, 30);
    let dropped = false;
    const timer = setInterval(() => {
      if (this.scenario === 'stale' && Date.now() - this.state.t0 > STALE_AFTER) {
        if (!dropped) h.connection?.('reconnecting');
        dropped = true;
        return;
      }
      state.tick(id, Date.now());
    }, 250);
    return () => {
      clearTimeout(t0);
      clearInterval(timer);
      unsub();
    };
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

  /** Models the demo's server has predicted with: its first call of each is `cold` (the warm-up is in its timing). */
  #warm = new Set<string>();

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
        const s = this.spec(input.dataset_id);
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
      const cold = server && !this.#warm.has(m.id);
      if (server) this.#warm.add(m.id);
      // A model's first prediction on the server includes its warm-up (GPU kernels compiled, buffers allocated).
      const inference = archInfo[m.arch].ms * scale * (1 + (r() - 0.5) * 0.08) + (cold ? 180 * scale : 0);
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
        ...(cold ? { cold: true } : {}),
      };
    })();
  }
}
