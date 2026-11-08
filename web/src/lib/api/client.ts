// The Studio API surface (docs/API.md) and its HTTP implementation. The in-browser mock (./mock) implements the
// same interface, so screens never know which one they talk to.

import type {
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
  Model,
  ModelEval,
  PredictInput,
  PredictOptions,
  Prediction,
  Run,
  RunEventMap,
  RunLogs,
  RunMetrics,
  RunSamples,
  SystemInfo,
  TrainConfig,
} from './types';
import { openEventStream } from './sse';

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

export type RunEventHandlers = {
  [K in keyof RunEventMap]?: (data: RunEventMap[K]) => void;
} & {
  /** Called after the stream reconnects; callers should refetch history to fill the gap. */
  reconnect?: () => void;
  /** Connection state changes, for "reconnecting…" indicators. */
  connection?: (state: 'open' | 'reconnecting') => void;
};

export interface Api {
  readonly mock: boolean;
  health(signal?: AbortSignal): Promise<Health>;
  system(signal?: AbortSignal): Promise<SystemInfo>;

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
  exportModel(id: string, format: ExportFormat): Promise<ExportResult>;

  runs(signal?: AbortSignal): Promise<Run[]>;
  run(id: string, signal?: AbortSignal): Promise<Run>;
  createRun(config: TrainConfig): Promise<Run>;
  stopRun(id: string): Promise<Run>;
  /**
   * `POST /api/runs` with `resume_run_id`: continues a `stopped` or `failed` run from its last checkpoint (200, same
   * id). Only `device` may differ from the run's config. Errors: `run_not_resumable` (completed), `run_active`.
   */
  resumeRun(id: string, device?: string): Promise<Run>;
  deleteRun(id: string): Promise<void>;
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
  exportModel(id: string, format: ExportFormat) {
    return this.#req<ExportResult>('POST', `/models/${encodeURIComponent(id)}/export`, { format });
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
  stopRun(id: string) {
    return this.#req<Run>('POST', `/runs/${encodeURIComponent(id)}/stop`);
  }
  resumeRun(id: string, device?: string) {
    return this.#req<Run>('POST', '/runs', device ? { resume_run_id: id, device } : { resume_run_id: id });
  }
  deleteRun(id: string) {
    return this.#req<void>('DELETE', `/runs/${encodeURIComponent(id)}`);
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
    return openEventStream(`${this.base}/api/runs/${encodeURIComponent(id)}/events`, handlers);
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
