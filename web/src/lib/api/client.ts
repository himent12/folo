// The Studio API surface (docs/API.md) and its HTTP implementation. The in-browser mock (./mock) implements the
// same interface, so screens never know which one they talk to.

import type {
  Bench,
  BenchCatalog,
  BenchDesign,
  BenchEntry,
  BenchEntryImagePage,
  BenchEntryImagesQuery,
  BenchEventMap,
  BenchImageOptions,
  BenchImagePage,
  BenchImagesQuery,
  BenchLoop,
  BenchPrediction,
  BenchmarkReport,
  BenchmarkRequest,
  BenchResult,
  Box,
  CreateBench,
  DesignBench,
  StartBenchEntry,
  StartBenchLoop,
  UpdateBench,
  CliCommand,
  CloudConnection,
  CloudInstance,
  CloudOfferList,
  CloudProvider,
  ConnectCloud,
  CreateDataset,
  CreateLabelJob,
  CreateSyntheticDataset,
  Dataset,
  DatasetImages,
  DatasetImagesQuery,
  DatasetIssues,
  DatasetIssuesQuery,
  ExportFormat,
  ExportResult,
  Health,
  LabelAgent,
  LabelDatasetRequest,
  LabelEventMap,
  LabelImage,
  LabelImagePage,
  LabelImagesAdded,
  LabelImagesQuery,
  LabelJob,
  LogLine,
  Model,
  ModelEval,
  OnnxSwitches,
  PredictInput,
  PredictOptions,
  Prediction,
  QueueRequest,
  Run,
  RunEventMap,
  RunLogs,
  RunMetrics,
  RunEstimate,
  RunSamples,
  StartLabeling,
  SystemInfo,
  ThroughputTable,
  TrainConfig,
  UpdateLabelJob,
} from './types';
import { isArch } from '$lib/arch';
import type { AccessWatch } from './access';
import type { Reachability } from './reach';
import { BENCH_EVENTS, LABEL_EVENTS, openEventStream, RUN_EVENTS, type EventHandlers } from './sse';
import { uploadInChunks, type UploadProgress } from './upload';

export class ApiError extends Error {
  /** `request`: what was asked, `GET /api/runs` (no query), when known: ErrorState names what could not be loaded. */
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly hint?: string | null,
    readonly request?: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export type RunEventHandlers = EventHandlers<RunEventMap>;
export type LabelEventHandlers = EventHandlers<LabelEventMap>;
export type BenchEventHandlers = EventHandlers<BenchEventMap>;

export interface Api {
  readonly mock: boolean;
  health(signal?: AbortSignal): Promise<Health>;
  system(signal?: AbortSignal): Promise<SystemInfo>;
  /** `GET /api/system/throughput` (servers whose `system().features` has `throughput`): measured and reference rows. */
  throughput(signal?: AbortSignal): Promise<ThroughputTable>;
  /** `POST /api/system/benchmark` (feature `benchmark`): a 1–10 s probe; `409 run_active` while a run is queued or running. */
  benchmark(body?: BenchmarkRequest): Promise<BenchmarkReport>;

  datasets(signal?: AbortSignal): Promise<Dataset[]>;
  dataset(id: string, signal?: AbortSignal): Promise<Dataset>;
  createDataset(body: CreateDataset): Promise<Dataset>;
  createSyntheticDataset(body: CreateSyntheticDataset): Promise<Dataset>;
  datasetImages(id: string, q: DatasetImagesQuery, signal?: AbortSignal): Promise<DatasetImages>;
  /** `GET /api/datasets/{id}/issues`: what `folo dataset check` finds, in dataset order. */
  datasetIssues(id: string, q: DatasetIssuesQuery, signal?: AbortSignal): Promise<DatasetIssues>;
  /** URL of an image (thumbnail when `thumb` is set). The mock renders images on demand, hence async. */
  imageUrl(datasetId: string, imageId: string, thumb?: number): Promise<string>;
  /** Unregisters a dataset; `files` also deletes the folder of one the server wrote into its data dir (docs/API.md). */
  deleteDataset(id: string, opts?: { files?: boolean }): Promise<void>;

  models(signal?: AbortSignal): Promise<Model[]>;
  model(id: string, signal?: AbortSignal): Promise<Model>;
  /** PR curves, confusion matrix and per-class metrics on the model's validation split. */
  modelEval(id: string, signal?: AbortSignal): Promise<ModelEval>;
  /** `switches` apply to `onnx` only (and only on servers whose `system().features` has `export_onnx_switches`). */
  exportModel(id: string, format: ExportFormat, switches?: OnnxSwitches): Promise<ExportResult>;
  /**
   * The model's ONNX graph as a file (GET /api/models/{id}/weights?format=onnx with the switches as query parameters),
   * under the name the server gives it (`<name>.onnx`, `<name>-f16.onnx` with `half`). Servers list `weights_onnx`.
   */
  onnxFile(id: string, switches: OnnxSwitches, signal?: AbortSignal): Promise<File>;
  /**
   * The model's FOLO f16 export as a file (GET /api/models/{id}/weights?format=f16), under the name the server gives it
   * (`<name>-f16.folo`). Nothing is written on the server but its cache.
   */
  f16File(id: string, signal?: AbortSignal): Promise<File>;

  runs(signal?: AbortSignal): Promise<Run[]>;
  run(id: string, signal?: AbortSignal): Promise<Run>;
  createRun(config: TrainConfig): Promise<Run>;
  /** `POST /api/runs/estimate` (feature `run_estimate`): the run's time and peak GPU memory; nothing is created. */
  estimateRun(config: TrainConfig, signal?: AbortSignal): Promise<RunEstimate>;
  stopRun(id: string): Promise<Run>;
  /**
   * `POST /api/runs` with `resume_run_id`: continues a `stopped` or `failed` run from its last checkpoint (200, same
   * id). Only `device` may differ from the run's config. Errors: `run_not_resumable` (completed), `run_active`.
   */
  resumeRun(id: string, device?: string): Promise<Run>;
  deleteRun(id: string): Promise<void>;
  /**
   * `GET /api/runs/{id}/cli` (feature `run_cli`): the run's `folo train` command on the server's machine, quoted for
   * PowerShell (`cli`), and its exact arguments (`argv`).
   */
  runCli(id: string, signal?: AbortSignal): Promise<CliCommand>;
  /**
   * `POST /api/runs/{id}/queue`: moves a run that waits in the training queue (servers list `queue_control`). `start`
   * puts it at the front; FOLO never interrupts the active run, so it starts once the slot is free. A run the worker
   * already took is `409 run_not_queued`.
   */
  moveRun(id: string, req: QueueRequest): Promise<Run>;
  /** Full history; `every` keeps every N-th step (plus the latest) to thin long runs. */
  runMetrics(id: string, signal?: AbortSignal, every?: number): Promise<RunMetrics>;
  /** The most recent `limit` log lines, oldest first. */
  runLogs(id: string, limit?: number, signal?: AbortSignal): Promise<RunLogs>;
  runSamples(id: string, epoch?: number, signal?: AbortSignal): Promise<RunSamples>;
  /** Subscribes to a run's SSE stream. Returns an unsubscribe function. */
  runEvents(id: string, handlers: RunEventHandlers): () => void;

  predict(input: PredictInput, opts: PredictOptions, signal?: AbortSignal): Promise<Prediction>;
  /**
   * A model's `.folo` file for running it in the browser: the f16 export (GET /api/models/{id}/weights?format=f16,
   * a fraction of a checkpoint's size). The browser's HTTP cache revalidates it by ETag.
   */
  modelWeights(id: string, signal?: AbortSignal): Promise<ArrayBuffer>;
  /** The weights' current ETag (a HEAD request): it changes when the model's file does. Null when unknown. */
  modelWeightsTag(id: string, signal?: AbortSignal): Promise<string | null>;

  // Cloud compute (docs/API.md, Cloud compute): servers whose `system().features` has `cloud`.
  /** `GET /api/cloud/providers`: the kinds of compute, with the fields a connection to each takes. */
  cloudProviders(signal?: AbortSignal): Promise<CloudProvider[]>;
  /** `GET /api/cloud/connections`, oldest first; status, account and balance are from the last check. */
  cloudConnections(signal?: AbortSignal): Promise<CloudConnection[]>;
  /**
   * `POST /api/cloud/connections`: checks the values with the provider, then stores them. `400 invalid_body` names a
   * missing field, `422 cloud_auth_failed` is a rejected credential, `502 cloud_unavailable` a provider that does not
   * answer. The values are credentials: never log, store or show them.
   */
  connectCloud(body: ConnectCloud): Promise<CloudConnection>;
  /** `POST /api/cloud/connections/{id}/check`: checks the stored credentials again; a failed check is `status: error`. */
  checkCloudConnection(id: string): Promise<CloudConnection>;
  /** `DELETE /api/cloud/connections/{id}`: forgets the credentials; `409 connection_in_use` while it has live instances or active runs. */
  disconnectCloud(id: string): Promise<void>;
  /** `GET /api/cloud/connections/{id}/offers`, cheapest first, cached for a minute by the server unless `refresh`. */
  cloudOffers(id: string, signal?: AbortSignal, refresh?: boolean): Promise<CloudOfferList>;
  /** `GET /api/cloud/instances`: the live ones first, then those that ended in the last 7 days, newest first. */
  cloudInstances(signal?: AbortSignal): Promise<CloudInstance[]>;
  /** `DELETE /api/cloud/instances/{id}`: terminates it now (`terminating`, or `terminated` once confirmed); its run fails. */
  terminateCloudInstance(id: string): Promise<CloudInstance>;

  // Agent labeling (docs/API.md; servers list `agent_labeling`).
  /** Registered labeling agents (Claude Code, Codex, Antigravity), each with whether its CLI is installed on the server's machine. */
  labelAgents(signal?: AbortSignal): Promise<LabelAgent[]>;
  /** Newest first. */
  labelJobs(signal?: AbortSignal): Promise<LabelJob[]>;
  labelJob(id: string, signal?: AbortSignal): Promise<LabelJob>;
  createLabelJob(body: CreateLabelJob): Promise<LabelJob>;
  /** `409 label_job_active` while a run is active; once the job has boxes, classes may only be renamed or appended. */
  updateLabelJob(id: string, body: UpdateLabelJob): Promise<LabelJob>;
  /** Removes the job, its uploaded images and labels; never folder images or datasets made from it. */
  deleteLabelJob(id: string): Promise<void>;
  labelImages(jobId: string, q?: LabelImagesQuery, signal?: AbortSignal): Promise<LabelImagePage>;
  /**
   * Uploads images (multipart, one `image` part each) in requests of at most ~48 MiB / 50 files (the server takes
   * 64 MiB a request), calling `onprogress` after each. The answers are merged: every file added or skipped, and the
   * job after the last. A request that fails after others added images throws `UploadIncomplete` with those.
   */
  addLabelImages(jobId: string, files: File[], onprogress?: (p: UploadProgress) => void): Promise<LabelImagesAdded>;
  /** Adds the images of a folder (or one image file) on the server's machine, read in place. */
  importLabelImages(jobId: string, path: string): Promise<LabelImagesAdded>;
  /**
   * Labels drawn by hand: they replace the image's boxes and it becomes `labeled` by `manual` (none: a background image).
   * `expect`: the revision (lib/label `boxRevision`) of the boxes the edit was made on; `409 label_boxes_changed` once
   * the image has others (an older server ignores it).
   */
  updateLabelImage(jobId: string, imageId: string, boxes: Box[], expect?: string): Promise<LabelImage>;
  /** Marks a labeled image's labels as checked by a person (or not), leaving its boxes as they are; `expect` as above. */
  reviewLabelImage(jobId: string, imageId: string, reviewed: boolean, expect?: string): Promise<LabelImage>;
  deleteLabelImage(jobId: string, imageId: string): Promise<void>;
  startLabeling(jobId: string, body: StartLabeling): Promise<LabelJob>;
  stopLabeling(jobId: string): Promise<LabelJob>;
  /** Subscribes to a job's SSE stream (it replays nothing: load the images, then apply events). */
  labelEvents(jobId: string, handlers: LabelEventHandlers): () => void;
  /** The job's log history (its `log` events), oldest first. */
  labelLogs(jobId: string, limit?: number, signal?: AbortSignal): Promise<{ lines: LogLine[] }>;
  /** Writes the job's labeled images as a YOLO dataset and registers it. */
  createLabelDataset(jobId: string, body: LabelDatasetRequest): Promise<Dataset>;
  /** URL of a job's image (thumbnail when `thumb` is set), as `imageUrl`. */
  labelImageUrl(jobId: string, imageId: string, thumb?: number): Promise<string>;

  // Vision benchmarks (docs/API.md; servers list `vision_bench`, $lib/bench/rules BENCH_FEATURE). docs/BENCH.md
  // defines every number. `folo serve --mock` answers these `501 not_implemented`.
  /** `GET /api/bench/catalog`: the conditions a plan can name, the default plan and the axes of the score. */
  benchCatalog(signal?: AbortSignal): Promise<BenchCatalog>;
  /** Newest first. */
  benches(signal?: AbortSignal): Promise<Bench[]>;
  bench(id: string, signal?: AbortSignal): Promise<Bench>;
  /**
   * Copies the source's labeled images and true boxes in (a seeded sample past `max_images`); the plan is the
   * catalog's default when it names none. `400 invalid_dataset` when the source has no labeled image.
   */
  createBench(body: CreateBench): Promise<Bench>;
  /** Absent fields stay. `409 bench_frozen` for a `plan` once the benchmark has entries. */
  updateBench(id: string, body: UpdateBench): Promise<Bench>;
  /** The benchmark, its images, entries and loops (never a model, run or dataset a loop made). `409 bench_active` while one is active. */
  deleteBench(id: string): Promise<void>;
  /**
   * One request to a registered labeling agent (no tools; up to two minutes) for a plan drawn from the catalog: a proposal,
   * not saved (`updateBench` adopts it). `501 not_supported` without the agent's CLI, `502 agent_failed` (message, hint).
   */
  designBench(id: string, body: DesignBench, signal?: AbortSignal): Promise<BenchDesign>;
  /** In the benchmark's order (`limit` at most 1000); with `condition` and `severity`, the truth under it (geometry moves boxes). */
  benchImages(id: string, q?: BenchImagesQuery, signal?: AbortSignal): Promise<BenchImagePage>;
  /**
   * URL of a benchmark image: the original, or with `condition` and `severity` the JPEG every contestant is graded on
   * (any condition of the catalog, in the plan or not); `thumb` as for dataset images. The mock renders on demand.
   */
  benchImageUrl(id: string, imageId: string, opts?: BenchImageOptions): Promise<string>;
  /** The leaderboard: finished entries by score (highest first), then the others (newest first). */
  benchEntries(id: string, signal?: AbortSignal): Promise<BenchEntry[]>;
  /**
   * Runs a contestant: a model (`400 bench_class_mismatch` when no class matches; queued behind other model entries),
   * an agent (`501 not_supported` without its CLI), or external predictions, scored in the request (`400
   * invalid_body` names the first wrong record).
   */
  startBenchEntry(id: string, body: StartBenchEntry): Promise<BenchEntry>;
  benchEntry(id: string, entryId: string, signal?: AbortSignal): Promise<BenchEntry>;
  /** Everything measured. `409 bench_entry_not_done` until the entry is `done`. */
  benchResult(id: string, entryId: string, signal?: AbortSignal): Promise<BenchResult>;
  /** Per image the truth and the detections at or above best-F1 confidence; `worst` sorts by missed + wrong. `409 bench_entry_not_done` before. */
  benchEntryImages(id: string, entryId: string, q?: BenchEntryImagesQuery, signal?: AbortSignal): Promise<BenchEntryImagePage>;
  /** `409 bench_entry_not_active` when it is not queued or running. Stopping an open external entry discards it. */
  stopBenchEntry(id: string, entryId: string): Promise<BenchEntry>;
  /**
   * A part of an open external entry (`contestant.open: true`), up to 64 MiB, all or nothing: `400 invalid_body` names
   * the first bad record and adds nothing; `409 bench_entry_not_open` for an entry that is not an open external one.
   * Answers the entry with its new `progress.done`.
   */
  sendBenchPredictions(id: string, entryId: string, predictions: BenchPrediction[]): Promise<BenchEntry>;
  /** Scores an open external entry: `done`; `400 invalid_body` (naming what is missing) leaves it open. */
  finishBenchEntry(id: string, entryId: string): Promise<BenchEntry>;
  /** `409 bench_entry_active` while it is active, `409 bench_entry_in_loop` while a running loop uses it. */
  deleteBenchEntry(id: string, entryId: string): Promise<void>;
  /** Newest first. */
  benchLoops(id: string, signal?: AbortSignal): Promise<BenchLoop[]>;
  /**
   * Starts a self-improvement loop. `400 invalid_body` when the teacher's classes are not the benchmark's (same names,
   * same order), `400 invalid_dataset` when the pool has no image outside the benchmark.
   */
  startBenchLoop(id: string, body: StartBenchLoop): Promise<BenchLoop>;
  benchLoop(id: string, loopId: string, signal?: AbortSignal): Promise<BenchLoop>;
  /** Stops its entry or run. `409 bench_loop_not_active` once it has ended. */
  stopBenchLoop(id: string, loopId: string): Promise<BenchLoop>;
  /** Its datasets, runs, models and entries stay. `409 bench_loop_active` while it runs. */
  deleteBenchLoop(id: string, loopId: string): Promise<void>;
  /** Subscribes to a benchmark's SSE stream (`bench` first; it replays nothing else: load the lists, then apply events). */
  benchEvents(id: string, handlers: BenchEventHandlers): () => void;
}

/** The file name of a `Content-Disposition: attachment; filename="…"` header, or null. */
export function attachmentName(header: string | null): string | null {
  const m = header ? /filename="?([^";]+)"?/i.exec(header) : null;
  return m?.[1]?.trim() || null;
}

/**
 * The name a model's ONNX download gets on the server (docs/API.md): `<name>.onnx`, `-decoded.onnx` with `decode`,
 * `-f16.onnx` with `half`, `-decoded-f16.onnx` with both, characters outside `A-Z a-z 0-9 . _ -` as `_`. Studio names
 * the file before it asks for it.
 */
export function onnxFileName(modelName: string, half: boolean, decode = false): string {
  return `${modelName.replace(/[^A-Za-z0-9._-]/g, '_')}${decode ? '-decoded' : ''}${half ? '-f16' : ''}.onnx`;
}

/** The name a model's FOLO f16 download gets on the server (docs/API.md): `<name>-f16.folo`, by the same rule. */
export function f16FileName(modelName: string): string {
  return `${modelName.replace(/[^A-Za-z0-9._-]/g, '_')}-f16.folo`;
}

/** Builds a query string from defined values only. */
export function qs(params: Record<string, string | number | undefined | null>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') u.set(k, String(v));
  const s = u.toString();
  return s ? `?${s}` : '';
}

export class HttpApi implements Api {
  readonly mock = false;

  /**
   * `base` is the server origin ('' for same origin, e.g. behind the Vite proxy or `folo serve`). `reach` hears whether
   * each request and event stream reached the server (the hub's: `hub.offline`); a one-off check passes none. `access`
   * hears of each 401 or 403 (the hub's: whether the server still lets Studio in, `hub.refused`).
   */
  constructor(
    readonly base: string = '',
    readonly reach?: Reachability,
    readonly access?: AccessWatch,
    readonly onRecover?: (fn: () => void) => () => void,
  ) {}

  /** A 401 or 403 to an `/api/` request may be the server no longer letting Studio in (a proxy that stopped sending the
   * token): `access` checks. `seen`: its `checks` when the request went out. */
  #answered(res: Response, seen: number): void {
    if (res.status === 401 || res.status === 403) this.access?.refused(seen);
  }

  /** An event stream whose opening, dropping, and 401/403 refusals also update `reach` and `access`. */
  #stream<M>(url: string, events: readonly (keyof M & string)[], handlers: EventHandlers<M>): () => void {
    const watch = this.reach || this.access || this.onRecover ? { reach: this.reach, access: this.access, onRecover: this.onRecover } : undefined;
    return openEventStream(url, events, handlers, undefined, undefined, watch);
  }

  async #req<T>(method: string, path: string, body?: unknown, signal?: AbortSignal, onResponse?: (res: Response) => void): Promise<T> {
    const init: RequestInit = { method, signal, headers: { accept: 'application/json' } };
    if (body instanceof FormData) init.body = body;
    else if (body !== undefined) {
      init.body = JSON.stringify(body);
      init.headers = { ...init.headers, 'content-type': 'application/json' };
    }
    const request = `${method} /api${path.split('?')[0]}`;
    let res: Response;
    const seen = this.access?.checks ?? 0;
    try {
      res = await fetch(`${this.base}/api${path}`, init);
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      this.reach?.failed();
      throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.', request);
    }
    this.reach?.ok();
    this.#answered(res, seen);
    if (res.status === 204) return undefined as T;
    const text = await res.text();
    let json: unknown;
    try {
      json = text ? JSON.parse(text) : undefined;
    } catch {
      throw new ApiError(res.status, 'bad_response', `The server sent a response that is not JSON (HTTP ${res.status}).`, null, request);
    }
    if (!res.ok) {
      const err = (json as { error?: { code?: string; message?: string; hint?: string } } | undefined)?.error;
      throw new ApiError(res.status, err?.code ?? `http_${res.status}`, err?.message ?? `Request failed (HTTP ${res.status}).`, err?.hint, request);
    }
    onResponse?.(res);
    return json as T;
  }

  health(signal?: AbortSignal) {
    return this.#req<Health>('GET', '/health', undefined, signal);
  }
  system(signal?: AbortSignal) {
    return this.#req<SystemInfo>('GET', '/system', undefined, signal);
  }
  throughput(signal?: AbortSignal) {
    return this.#req<ThroughputTable>('GET', '/system/throughput', undefined, signal);
  }
  benchmark(body: BenchmarkRequest = {}) {
    return this.#req<BenchmarkReport>('POST', '/system/benchmark', body);
  }

  async datasets(signal?: AbortSignal) {
    return this.#list<Dataset>('/datasets', 'datasets', signal);
  }
  dataset(id: string, signal?: AbortSignal) {
    return this.#req<Dataset>('GET', `/datasets/${encodeURIComponent(id)}`, undefined, signal);
  }
  createDataset(body: CreateDataset) {
    return this.#req<Dataset>('POST', '/datasets', body);
  }
  createSyntheticDataset(body: CreateSyntheticDataset) {
    return this.#req<Dataset>('POST', '/datasets/synthetic', body);
  }
  datasetImages(id: string, q: DatasetImagesQuery, signal?: AbortSignal) {
    return this.#envelope<DatasetImages>(
      `/datasets/${encodeURIComponent(id)}/images${qs({ split: q.split, offset: q.offset, limit: q.limit, class: q.class })}`,
      'images',
      true,
      signal,
    );
  }
  datasetIssues(id: string, q: DatasetIssuesQuery, signal?: AbortSignal) {
    return this.#req<DatasetIssues>(
      'GET',
      `/datasets/${encodeURIComponent(id)}/issues${qs({ kind: q.kind, severity: q.severity, offset: q.offset, limit: q.limit })}`,
      undefined,
      signal,
    );
  }
  async imageUrl(datasetId: string, imageId: string, thumb?: number) {
    return `${this.base}/api/datasets/${encodeURIComponent(datasetId)}/images/${encodeURIComponent(imageId)}${qs({ thumb })}`;
  }
  deleteDataset(id: string, opts: { files?: boolean } = {}) {
    return this.#req<void>('DELETE', `/datasets/${encodeURIComponent(id)}${opts.files ? '?files=true' : ''}`);
  }

  async models(signal?: AbortSignal) {
    return this.#list<Model>('/models', 'models', signal);
  }
  async model(id: string, signal?: AbortSignal) {
    let headerArch: string | null = null;
    const m = await this.#req<Model>('GET', `/models/${encodeURIComponent(id)}`, undefined, signal, (res) => {
      headerArch = res.headers.get('x-folo-arch')?.trim() ?? null;
    });
    // folo-server emits the exact `folo2-*` architecture in `x-folo-arch` alongside the wire `folo_api::Arch` field.
    return isArch(headerArch) && headerArch !== m.arch ? { ...m, arch: headerArch } : m;
  }
  modelEval(id: string, signal?: AbortSignal) {
    return this.#req<ModelEval>('GET', `/models/${encodeURIComponent(id)}/eval`, undefined, signal);
  }
  exportModel(id: string, format: ExportFormat, switches?: OnnxSwitches) {
    // With `folo-f16` any switch is a 400: send them for ONNX only.
    const body = format === 'onnx' && switches ? { format, ...switches } : { format };
    return this.#req<ExportResult>('POST', `/models/${encodeURIComponent(id)}/export`, body);
  }

  async runs(signal?: AbortSignal) {
    return this.#list<Run>('/runs', 'runs', signal);
  }
  run(id: string, signal?: AbortSignal) {
    return this.#req<Run>('GET', `/runs/${encodeURIComponent(id)}`, undefined, signal);
  }
  createRun(config: TrainConfig) {
    return this.#req<Run>('POST', '/runs', config);
  }
  estimateRun(config: TrainConfig, signal?: AbortSignal) {
    return this.#req<RunEstimate>('POST', '/runs/estimate', config, signal);
  }
  stopRun(id: string) {
    return this.#req<Run>('POST', `/runs/${encodeURIComponent(id)}/stop`);
  }
  resumeRun(id: string, device?: string) {
    return this.#req<Run>('POST', '/runs', device ? { resume_run_id: id, device } : { resume_run_id: id });
  }
  deleteRun(id: string) {
    return this.#req<void>('DELETE', `/runs/${encodeURIComponent(id)}`);
  }
  moveRun(id: string, req: QueueRequest) {
    return this.#req<Run>('POST', `/runs/${encodeURIComponent(id)}/queue`, req);
  }
  runCli(id: string, signal?: AbortSignal) {
    return this.#req<CliCommand>('GET', `/runs/${encodeURIComponent(id)}/cli`, undefined, signal);
  }
  runMetrics(id: string, signal?: AbortSignal, every?: number) {
    return this.#req<RunMetrics>('GET', `/runs/${encodeURIComponent(id)}/metrics${qs({ every })}`, undefined, signal);
  }
  runLogs(id: string, limit?: number, signal?: AbortSignal) {
    return this.#req<RunLogs>('GET', `/runs/${encodeURIComponent(id)}/logs${qs({ limit })}`, undefined, signal);
  }
  runSamples(id: string, epoch?: number, signal?: AbortSignal) {
    return this.#req<RunSamples>('GET', `/runs/${encodeURIComponent(id)}/samples${qs({ epoch })}`, undefined, signal);
  }
  runEvents(id: string, handlers: RunEventHandlers) {
    return this.#stream(`${this.base}/api/runs/${encodeURIComponent(id)}/events`, RUN_EVENTS, handlers);
  }

  async labelAgents(signal?: AbortSignal) {
    return this.#list<LabelAgent>('/label/agents', 'agents', signal);
  }
  async labelJobs(signal?: AbortSignal) {
    return this.#list<LabelJob>('/label/jobs', 'jobs', signal);
  }
  labelJob(id: string, signal?: AbortSignal) {
    return this.#req<LabelJob>('GET', `/label/jobs/${encodeURIComponent(id)}`, undefined, signal);
  }
  createLabelJob(body: CreateLabelJob) {
    return this.#req<LabelJob>('POST', '/label/jobs', body);
  }
  updateLabelJob(id: string, body: UpdateLabelJob) {
    return this.#req<LabelJob>('PUT', `/label/jobs/${encodeURIComponent(id)}`, body);
  }
  deleteLabelJob(id: string) {
    return this.#req<void>('DELETE', `/label/jobs/${encodeURIComponent(id)}`);
  }
  labelImages(jobId: string, q: LabelImagesQuery = {}, signal?: AbortSignal) {
    // A page without its images, or without the `total` that bounds the paging, would stop LiveLabelJob's paging
    // early and show a truncated job as whole (review round 2).
    return this.#envelope<LabelImagePage>(`/label/jobs/${encodeURIComponent(jobId)}/images${qs({ status: q.status, ids: q.ids?.join(','), offset: q.offset, limit: q.limit })}`, 'images', true, signal);
  }
  async addLabelImages(jobId: string, files: File[], onprogress?: (p: UploadProgress) => void) {
    const send = (chunk: File[]) => {
      const fd = new FormData();
      for (const f of chunk) fd.append('image', f, f.name);
      return this.#req<LabelImagesAdded>('POST', `/label/jobs/${encodeURIComponent(jobId)}/images`, fd);
    };
    return (await uploadInChunks(files, send, onprogress)) ?? { added: [], skipped: [], job: await this.labelJob(jobId) };
  }
  importLabelImages(jobId: string, path: string) {
    return this.#req<LabelImagesAdded>('POST', `/label/jobs/${encodeURIComponent(jobId)}/images`, { path });
  }
  updateLabelImage(jobId: string, imageId: string, boxes: Box[], expect?: string) {
    return this.#req<LabelImage>('PUT', `/label/jobs/${encodeURIComponent(jobId)}/images/${encodeURIComponent(imageId)}`, { boxes, expect });
  }
  reviewLabelImage(jobId: string, imageId: string, reviewed: boolean, expect?: string) {
    return this.#req<LabelImage>('PUT', `/label/jobs/${encodeURIComponent(jobId)}/images/${encodeURIComponent(imageId)}`, { reviewed, expect });
  }
  deleteLabelImage(jobId: string, imageId: string) {
    return this.#req<void>('DELETE', `/label/jobs/${encodeURIComponent(jobId)}/images/${encodeURIComponent(imageId)}`);
  }
  startLabeling(jobId: string, body: StartLabeling) {
    return this.#req<LabelJob>('POST', `/label/jobs/${encodeURIComponent(jobId)}/start`, body);
  }
  stopLabeling(jobId: string) {
    return this.#req<LabelJob>('POST', `/label/jobs/${encodeURIComponent(jobId)}/stop`);
  }
  labelEvents(jobId: string, handlers: LabelEventHandlers) {
    return this.#stream(`${this.base}/api/label/jobs/${encodeURIComponent(jobId)}/events`, LABEL_EVENTS, handlers);
  }
  labelLogs(jobId: string, limit?: number, signal?: AbortSignal) {
    return this.#req<{ lines: LogLine[] }>('GET', `/label/jobs/${encodeURIComponent(jobId)}/logs${qs({ limit })}`, undefined, signal);
  }
  createLabelDataset(jobId: string, body: LabelDatasetRequest) {
    return this.#req<Dataset>('POST', `/label/jobs/${encodeURIComponent(jobId)}/dataset`, body);
  }
  async labelImageUrl(jobId: string, imageId: string, thumb?: number) {
    return `${this.base}/api/label/jobs/${encodeURIComponent(jobId)}/images/${encodeURIComponent(imageId)}${qs({ thumb })}`;
  }

  benchCatalog(signal?: AbortSignal) {
    return this.#req<BenchCatalog>('GET', '/bench/catalog', undefined, signal);
  }
  async benches(signal?: AbortSignal) {
    return this.#list<Bench>('/bench/benchmarks', 'benchmarks', signal);
  }
  bench(id: string, signal?: AbortSignal) {
    return this.#req<Bench>('GET', `/bench/benchmarks/${encodeURIComponent(id)}`, undefined, signal);
  }
  createBench(body: CreateBench) {
    return this.#req<Bench>('POST', '/bench/benchmarks', body);
  }
  updateBench(id: string, body: UpdateBench) {
    return this.#req<Bench>('PUT', `/bench/benchmarks/${encodeURIComponent(id)}`, body);
  }
  deleteBench(id: string) {
    return this.#req<void>('DELETE', `/bench/benchmarks/${encodeURIComponent(id)}`);
  }
  designBench(id: string, body: DesignBench, signal?: AbortSignal) {
    return this.#req<BenchDesign>('POST', `/bench/benchmarks/${encodeURIComponent(id)}/design`, body, signal);
  }
  benchImages(id: string, q: BenchImagesQuery = {}, signal?: AbortSignal) {
    return this.#envelope<BenchImagePage>(
      `/bench/benchmarks/${encodeURIComponent(id)}/images${qs({ offset: q.offset, limit: q.limit, condition: q.condition, severity: q.severity })}`,
      'images',
      true,
      signal,
    );
  }
  async benchImageUrl(id: string, imageId: string, opts: BenchImageOptions = {}) {
    return `${this.base}/api/bench/benchmarks/${encodeURIComponent(id)}/images/${encodeURIComponent(imageId)}${qs({ condition: opts.condition, severity: opts.severity, thumb: opts.thumb })}`;
  }
  async benchEntries(id: string, signal?: AbortSignal) {
    return this.#list<BenchEntry>(`/bench/benchmarks/${encodeURIComponent(id)}/entries`, 'entries', signal);
  }
  startBenchEntry(id: string, body: StartBenchEntry) {
    return this.#req<BenchEntry>('POST', `/bench/benchmarks/${encodeURIComponent(id)}/entries`, body);
  }
  benchEntry(id: string, entryId: string, signal?: AbortSignal) {
    return this.#req<BenchEntry>('GET', `/bench/benchmarks/${encodeURIComponent(id)}/entries/${encodeURIComponent(entryId)}`, undefined, signal);
  }
  benchResult(id: string, entryId: string, signal?: AbortSignal) {
    return this.#req<BenchResult>('GET', `/bench/benchmarks/${encodeURIComponent(id)}/entries/${encodeURIComponent(entryId)}/result`, undefined, signal);
  }
  benchEntryImages(id: string, entryId: string, q: BenchEntryImagesQuery = {}, signal?: AbortSignal) {
    return this.#envelope<BenchEntryImagePage>(
      `/bench/benchmarks/${encodeURIComponent(id)}/entries/${encodeURIComponent(entryId)}/images${qs({ sort: q.sort, offset: q.offset, limit: q.limit })}`,
      'images',
      true,
      signal,
    );
  }
  stopBenchEntry(id: string, entryId: string) {
    return this.#req<BenchEntry>('POST', `/bench/benchmarks/${encodeURIComponent(id)}/entries/${encodeURIComponent(entryId)}/stop`);
  }
  sendBenchPredictions(id: string, entryId: string, predictions: BenchPrediction[]) {
    return this.#req<BenchEntry>('POST', `/bench/benchmarks/${encodeURIComponent(id)}/entries/${encodeURIComponent(entryId)}/predictions`, { predictions });
  }
  finishBenchEntry(id: string, entryId: string) {
    return this.#req<BenchEntry>('POST', `/bench/benchmarks/${encodeURIComponent(id)}/entries/${encodeURIComponent(entryId)}/finish`);
  }
  deleteBenchEntry(id: string, entryId: string) {
    return this.#req<void>('DELETE', `/bench/benchmarks/${encodeURIComponent(id)}/entries/${encodeURIComponent(entryId)}`);
  }
  async benchLoops(id: string, signal?: AbortSignal) {
    return this.#list<BenchLoop>(`/bench/benchmarks/${encodeURIComponent(id)}/loops`, 'loops', signal);
  }
  startBenchLoop(id: string, body: StartBenchLoop) {
    return this.#req<BenchLoop>('POST', `/bench/benchmarks/${encodeURIComponent(id)}/loops`, body);
  }
  benchLoop(id: string, loopId: string, signal?: AbortSignal) {
    return this.#req<BenchLoop>('GET', `/bench/benchmarks/${encodeURIComponent(id)}/loops/${encodeURIComponent(loopId)}`, undefined, signal);
  }
  stopBenchLoop(id: string, loopId: string) {
    return this.#req<BenchLoop>('POST', `/bench/benchmarks/${encodeURIComponent(id)}/loops/${encodeURIComponent(loopId)}/stop`);
  }
  deleteBenchLoop(id: string, loopId: string) {
    return this.#req<void>('DELETE', `/bench/benchmarks/${encodeURIComponent(id)}/loops/${encodeURIComponent(loopId)}`);
  }
  benchEvents(id: string, handlers: BenchEventHandlers) {
    return this.#stream(`${this.base}/api/bench/benchmarks/${encodeURIComponent(id)}/events`, BENCH_EVENTS, handlers);
  }

  async modelWeights(id: string, signal?: AbortSignal) {
    let res: Response;
    const seen = this.access?.checks ?? 0;
    try {
      res = await fetch(`${this.base}/api/models/${encodeURIComponent(id)}/weights?format=f16`, { signal });
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      this.reach?.failed();
      throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
    }
    this.reach?.ok();
    this.#answered(res, seen);
    if (res.ok) {
      try {
        return await res.arrayBuffer();
      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') throw e;
        this.reach?.failed();
        throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
      }
    }
    const err = ((await res.json().catch(() => undefined)) as { error?: { code?: string; message?: string; hint?: string } } | undefined)?.error;
    throw new ApiError(res.status, err?.code ?? `http_${res.status}`, err?.message ?? `Could not download the model (HTTP ${res.status}).`, err?.hint);
  }

  onnxFile(id: string, switches: OnnxSwitches, signal?: AbortSignal) {
    const on = (k: keyof OnnxSwitches) => (switches[k] ? 'true' : undefined);
    const query = qs({ format: 'onnx', decode: on('decode'), dynamic: on('dynamic'), half: on('half') });
    return this.#weightsFile(id, query, onnxFileName('model', !!switches.half, !!switches.decode), 'the ONNX file', signal);
  }

  f16File(id: string, signal?: AbortSignal) {
    return this.#weightsFile(id, '?format=f16', f16FileName('model'), 'the model', signal);
  }

  /** A `/weights` export as a file, under the name the server gives it (`fallback` when it gives none). */
  async #weightsFile(id: string, query: string, fallback: string, what: string, signal?: AbortSignal) {
    let res: Response;
    const seen = this.access?.checks ?? 0;
    try {
      res = await fetch(`${this.base}/api/models/${encodeURIComponent(id)}/weights${query}`, { signal });
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      this.reach?.failed();
      throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
    }
    this.reach?.ok();
    this.#answered(res, seen);
    if (!res.ok) {
      const err = ((await res.json().catch(() => undefined)) as { error?: { code?: string; message?: string; hint?: string } } | undefined)?.error;
      throw new ApiError(res.status, err?.code ?? `http_${res.status}`, err?.message ?? `Could not download ${what} (HTTP ${res.status}).`, err?.hint);
    }
    const name = attachmentName(res.headers.get('content-disposition')) ?? fallback;
    try {
      return new File([await res.blob()], name, { type: 'application/octet-stream' });
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      this.reach?.failed();
      throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
    }
  }

  async modelWeightsTag(id: string, signal?: AbortSignal) {
    const seen = this.access?.checks ?? 0;
    try {
      const res = await fetch(`${this.base}/api/models/${encodeURIComponent(id)}/weights?format=f16`, { method: 'HEAD', signal });
      this.reach?.ok();
      this.#answered(res, seen);
      return res.ok ? res.headers.get('etag') : null;
    } catch {
      return null;
    }
  }

  async cloudProviders(signal?: AbortSignal) {
    return this.#list<CloudProvider>('/cloud/providers', 'providers', signal);
  }
  async cloudConnections(signal?: AbortSignal) {
    return this.#list<CloudConnection>('/cloud/connections', 'connections', signal);
  }
  connectCloud(body: ConnectCloud) {
    return this.#req<CloudConnection>('POST', '/cloud/connections', body);
  }
  checkCloudConnection(id: string) {
    return this.#req<CloudConnection>('POST', `/cloud/connections/${encodeURIComponent(id)}/check`);
  }
  disconnectCloud(id: string) {
    return this.#req<void>('DELETE', `/cloud/connections/${encodeURIComponent(id)}`);
  }
  cloudOffers(id: string, signal?: AbortSignal, refresh = false) {
    return this.#envelope<CloudOfferList>(`/cloud/connections/${encodeURIComponent(id)}/offers${qs({ refresh: refresh ? 'true' : undefined })}`, 'offers', false, signal);
  }
  async cloudInstances(signal?: AbortSignal) {
    return this.#list<CloudInstance>('/cloud/instances', 'instances', signal);
  }
  terminateCloudInstance(id: string) {
    return this.#req<CloudInstance>('DELETE', `/cloud/instances/${encodeURIComponent(id)}`);
  }

  predict(input: PredictInput, opts: PredictOptions, signal?: AbortSignal) {
    if (input.kind === 'file') {
      const fd = new FormData();
      fd.set('image', input.file);
      fd.set('model_id', opts.model_id);
      if (opts.conf !== undefined) fd.set('conf', String(opts.conf));
      if (opts.iou !== undefined) fd.set('iou', String(opts.iou));
      if (opts.max_det !== undefined) fd.set('max_det', String(opts.max_det));
      if (opts.tile !== undefined) fd.set('tile', JSON.stringify(opts.tile));
      if (opts.tta !== undefined) fd.set('tta', JSON.stringify(opts.tta));
      return this.#req<Prediction>('POST', '/predict', fd, signal);
    }
    return this.#req<Prediction>(
      'POST',
      '/predict',
      {
        model_id: opts.model_id,
        dataset_id: input.dataset_id,
        image_id: input.image_id,
        conf: opts.conf,
        iou: opts.iou,
        max_det: opts.max_det,
        ...(opts.tile !== undefined ? { tile: opts.tile } : {}),
        ...(opts.tta !== undefined ? { tta: opts.tta } : {}),
      },
      signal,
    );
  }

  /**
   * A collection: `GET path` answers `{ <key>: [...] }`. Anything else (the key missing, null, an object) is the server's
   * mistake and an error, not an empty list: a missing list showed "No runs yet" as if that were true, and a list of the
   * wrong type made screens throw and wait forever (s6-shell-3).
   */
  async #list<T>(path: string, key: string, signal?: AbortSignal): Promise<T[]> {
    return (await this.#envelope<Record<string, T[]>>(path, key, false, signal))[key]!;
  }

  /**
   * An answer that holds a collection under `key` (and with `paged`, the `total` of a page): checked as `#list` checks
   * a list, and resolved whole. Offers of `null` left "Loading offers…" forever; a page without its images showed an
   * empty grid, or threw in the screen that read `.length` (s6-shell-3).
   */
  async #envelope<T extends object>(path: string, key: string, paged: boolean, signal?: AbortSignal): Promise<T> {
    const json = await this.#req<Record<string, unknown> | undefined>('GET', path, undefined, signal);
    const ok = json !== null && typeof json === 'object' && !Array.isArray(json) && Array.isArray(json[key]) && (!paged || typeof json.total === 'number');
    if (!ok) {
      const what = paged ? `no page of ${key}` : `no list of ${key}`;
      throw new ApiError(200, 'bad_response', `The server sent an answer of an unexpected shape (${what}).`, 'Is the server a FOLO server of the same version as Studio?', `GET /api${path.split('?')[0]}`);
    }
    return json as T;
  }
}
