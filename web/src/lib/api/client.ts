// The Studio API surface (docs/API.md) and its HTTP implementation. The in-browser mock (./mock) implements the
// same interface, so screens never know which one they talk to.

import type {
  BenchmarkReport,
  BenchmarkRequest,
  Box,
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
import { LABEL_EVENTS, openEventStream, RUN_EVENTS, type EventHandlers } from './sse';
import { uploadInChunks, type UploadProgress } from './upload';

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly hint?: string | null,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export type RunEventHandlers = EventHandlers<RunEventMap>;
export type LabelEventHandlers = EventHandlers<LabelEventMap>;

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
  deleteDataset(id: string): Promise<void>;

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
  /** Claude Code, then Codex, each with whether its CLI is installed on the server's machine. */
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
  /** Labels drawn by hand: they replace the image's boxes and it becomes `labeled` by `manual` (none: a background image). */
  updateLabelImage(jobId: string, imageId: string, boxes: Box[]): Promise<LabelImage>;
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
}

/** The file name of a `Content-Disposition: attachment; filename="…"` header, or null. */
export function attachmentName(header: string | null): string | null {
  const m = header ? /filename="?([^";]+)"?/i.exec(header) : null;
  return m?.[1]?.trim() || null;
}

/**
 * The name a model's ONNX download gets on the server (docs/API.md): `<name>.onnx`, `<name>-f16.onnx` with `half`,
 * characters outside `A-Z a-z 0-9 . _ -` as `_`. Studio names the file before it asks for it.
 */
export function onnxFileName(modelName: string, half: boolean): string {
  return `${modelName.replace(/[^A-Za-z0-9._-]/g, '_')}${half ? '-f16' : ''}.onnx`;
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

  /** `base` is the server origin ('' for same origin, e.g. behind the Vite proxy or `folo serve`). */
  constructor(readonly base: string = '') {}

  async #req<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
    const init: RequestInit = { method, signal, headers: { accept: 'application/json' } };
    if (body instanceof FormData) init.body = body;
    else if (body !== undefined) {
      init.body = JSON.stringify(body);
      init.headers = { ...init.headers, 'content-type': 'application/json' };
    }
    let res: Response;
    try {
      res = await fetch(`${this.base}/api${path}`, init);
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
    }
    if (res.status === 204) return undefined as T;
    const text = await res.text();
    let json: unknown;
    try {
      json = text ? JSON.parse(text) : undefined;
    } catch {
      throw new ApiError(res.status, 'bad_response', `The server sent a response that is not JSON (HTTP ${res.status}).`);
    }
    if (!res.ok) {
      const err = (json as { error?: { code?: string; message?: string; hint?: string } } | undefined)?.error;
      throw new ApiError(res.status, err?.code ?? `http_${res.status}`, err?.message ?? `Request failed (HTTP ${res.status}).`, err?.hint);
    }
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
    return (await this.#req<{ datasets: Dataset[] }>('GET', '/datasets', undefined, signal)).datasets;
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
    return this.#req<DatasetImages>(
      'GET',
      `/datasets/${encodeURIComponent(id)}/images${qs({ split: q.split, offset: q.offset, limit: q.limit, class: q.class })}`,
      undefined,
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
  deleteDataset(id: string) {
    return this.#req<void>('DELETE', `/datasets/${encodeURIComponent(id)}`);
  }

  async models(signal?: AbortSignal) {
    return (await this.#req<{ models: Model[] }>('GET', '/models', undefined, signal)).models;
  }
  model(id: string, signal?: AbortSignal) {
    return this.#req<Model>('GET', `/models/${encodeURIComponent(id)}`, undefined, signal);
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
    return (await this.#req<{ runs: Run[] }>('GET', '/runs', undefined, signal)).runs;
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
    return openEventStream(`${this.base}/api/runs/${encodeURIComponent(id)}/events`, RUN_EVENTS, handlers);
  }

  async labelAgents(signal?: AbortSignal) {
    return (await this.#req<{ agents: LabelAgent[] }>('GET', '/label/agents', undefined, signal)).agents;
  }
  async labelJobs(signal?: AbortSignal) {
    return (await this.#req<{ jobs: LabelJob[] }>('GET', '/label/jobs', undefined, signal)).jobs;
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
    return this.#req<LabelImagePage>(
      'GET',
      `/label/jobs/${encodeURIComponent(jobId)}/images${qs({ status: q.status, offset: q.offset, limit: q.limit })}`,
      undefined,
      signal,
    );
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
  updateLabelImage(jobId: string, imageId: string, boxes: Box[]) {
    return this.#req<LabelImage>('PUT', `/label/jobs/${encodeURIComponent(jobId)}/images/${encodeURIComponent(imageId)}`, { boxes });
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
    return openEventStream(`${this.base}/api/label/jobs/${encodeURIComponent(jobId)}/events`, LABEL_EVENTS, handlers);
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

  async modelWeights(id: string, signal?: AbortSignal) {
    let res: Response;
    try {
      res = await fetch(`${this.base}/api/models/${encodeURIComponent(id)}/weights?format=f16`, { signal });
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
    }
    if (res.ok) return res.arrayBuffer();
    const err = ((await res.json().catch(() => undefined)) as { error?: { code?: string; message?: string; hint?: string } } | undefined)?.error;
    throw new ApiError(res.status, err?.code ?? `http_${res.status}`, err?.message ?? `Could not download the model (HTTP ${res.status}).`, err?.hint);
  }

  async onnxFile(id: string, switches: OnnxSwitches, signal?: AbortSignal) {
    const on = (k: keyof OnnxSwitches) => (switches[k] ? 'true' : undefined);
    const url = `${this.base}/api/models/${encodeURIComponent(id)}/weights${qs({ format: 'onnx', decode: on('decode'), dynamic: on('dynamic'), half: on('half') })}`;
    let res: Response;
    try {
      res = await fetch(url, { signal });
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') throw e;
      throw new ApiError(0, 'network', 'Cannot reach the FOLO server.', 'Is folo serve running? Check the server URL in Settings.');
    }
    if (!res.ok) {
      const err = ((await res.json().catch(() => undefined)) as { error?: { code?: string; message?: string; hint?: string } } | undefined)?.error;
      throw new ApiError(res.status, err?.code ?? `http_${res.status}`, err?.message ?? `Could not download the ONNX file (HTTP ${res.status}).`, err?.hint);
    }
    const name = attachmentName(res.headers.get('content-disposition')) ?? `model${switches.half ? '-f16' : ''}.onnx`;
    return new File([await res.blob()], name, { type: 'application/octet-stream' });
  }

  /** The f16 `.folo` file's URL (GET /api/models/{id}/weights?format=f16), for a browser download link. */
  weightsUrl(id: string): string {
    return `${this.base}/api/models/${encodeURIComponent(id)}/weights?format=f16`;
  }

  async modelWeightsTag(id: string, signal?: AbortSignal) {
    try {
      const res = await fetch(`${this.base}/api/models/${encodeURIComponent(id)}/weights?format=f16`, { method: 'HEAD', signal });
      return res.ok ? res.headers.get('etag') : null;
    } catch {
      return null;
    }
  }

  async cloudProviders(signal?: AbortSignal) {
    return (await this.#req<{ providers: CloudProvider[] }>('GET', '/cloud/providers', undefined, signal)).providers;
  }
  async cloudConnections(signal?: AbortSignal) {
    return (await this.#req<{ connections: CloudConnection[] }>('GET', '/cloud/connections', undefined, signal)).connections;
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
    return this.#req<CloudOfferList>('GET', `/cloud/connections/${encodeURIComponent(id)}/offers${qs({ refresh: refresh ? 'true' : undefined })}`, undefined, signal);
  }
  async cloudInstances(signal?: AbortSignal) {
    return (await this.#req<{ instances: CloudInstance[] }>('GET', '/cloud/instances', undefined, signal)).instances;
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
      return this.#req<Prediction>('POST', '/predict', fd, signal);
    }
    return this.#req<Prediction>(
      'POST',
      '/predict',
      { model_id: opts.model_id, dataset_id: input.dataset_id, image_id: input.image_id, conf: opts.conf, iou: opts.iou, max_det: opts.max_det },
      signal,
    );
  }
}
