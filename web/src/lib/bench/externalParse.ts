// Runs external benchmark prediction JSON reading, parsing, and validation in a dedicated Web Worker
// so large files never block the UI thread, keeping validated predictions inside the worker and streaming
// bounded upload parts on demand, with a clean main-thread fallback and AbortSignal cancellation.
import type { BenchPrediction } from '../api/types';
import {
  estimateBytes,
  PART_BYTES,
  parseAndCheckPredictions,
  partsOf,
  type ExternalPartSource,
  type PredictionContext,
  type PredictionSummary,
} from './external';
import type { ExternalWorkerRequest, ExternalWorkerResponse } from './externalWorker';

/** Minimal subset of the standard Worker interface used by `parseExternalPredictionsFile`. */
export interface ExternalParseWorker {
  postMessage(message: ExternalWorkerRequest): void;
  terminate(): void;
  addEventListener(type: 'message', listener: (ev: MessageEvent<ExternalWorkerResponse>) => void): void;
  addEventListener(type: 'error', listener: (ev: Event) => void): void;
  addEventListener(type: 'messageerror', listener: (ev: Event) => void): void;
  removeEventListener(type: 'message', listener: (ev: MessageEvent<ExternalWorkerResponse>) => void): void;
  removeEventListener(type: 'error', listener: (ev: Event) => void): void;
  removeEventListener(type: 'messageerror', listener: (ev: Event) => void): void;
}

/** Validated external predictions session backed by either a live Web Worker or main-thread memory. */
export interface ExternalPredictionSession extends ExternalPartSource {
  readonly summary: PredictionSummary;
  /** Releases worker or memory resources; safe to call multiple times. */
  dispose(): void;
}

export type ExternalParseResult =
  | { ok: true; session: ExternalPredictionSession }
  | { ok: false; error: string };

export interface ParseExternalOptions {
  /** Aborting terminates an in-flight worker immediately and rejects with an AbortError. */
  signal?: AbortSignal;
  /** Optional worker factory (defaults to spawning `./externalWorker.ts` when `Worker` exists). */
  createWorker?: () => ExternalParseWorker | null;
  /** When falling back to the main thread, yield a paint frame before parsing. Default true. */
  yieldOnFallback?: boolean;
  /** Byte cap per upload part (defaults to `PART_BYTES`). */
  maxPartBytes?: number;
}

/** True when `err` represents a cancelled parse (DOMException/Error with name `AbortError`). */
export function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === 'AbortError';
}

function abortError(signal: AbortSignal): Error {
  if (signal.reason instanceof Error) return signal.reason;
  return new DOMException('Aborted', 'AbortError');
}

/** Races `p` against `signal` without leaving an unhandled rejection if `p` settles after abort. */
export function raceAbort<T>(p: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return p;
  if (signal.aborted) {
    p.catch(() => {});
    return Promise.reject(abortError(signal));
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError(signal));
    signal.addEventListener('abort', onAbort, { once: true });
    p.then(
      (v) => {
        signal.removeEventListener('abort', onAbort);
        resolve(v);
      },
      (e) => {
        signal.removeEventListener('abort', onAbort);
        reject(e);
      },
    );
  });
}

function yieldPaint(signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.reject(abortError(signal));
  return new Promise<void>((resolve, reject) => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    let raf: number | undefined;
    const done = () => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    };
    const onAbort = () => {
      if (raf !== undefined && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(raf);
      if (timer !== undefined) clearTimeout(timer);
      reject(abortError(signal!));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
    if (typeof requestAnimationFrame === 'function') {
      raf = requestAnimationFrame(() => {
        raf = undefined;
        timer = setTimeout(done, 0);
      });
    } else {
      timer = setTimeout(done, 0);
    }
  });
}

function defaultCreateWorker(): ExternalParseWorker | null {
  if (typeof Worker === 'undefined') return null;
  return new Worker(new URL('./externalWorker.ts', import.meta.url), { type: 'module' }) as unknown as ExternalParseWorker;
}

/** Sends a single request to `worker` and resolves with the matching response, or rejects on error/abort. */
function sendWorkerRequest(
  worker: ExternalParseWorker,
  req: ExternalWorkerRequest,
  signal?: AbortSignal,
  terminateOnAbort = false,
): Promise<ExternalWorkerResponse> {
  if (signal?.aborted) {
    if (terminateOnAbort) worker.terminate();
    return Promise.reject(abortError(signal));
  }
  return new Promise<ExternalWorkerResponse>((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
      worker.removeEventListener('messageerror', onError);
      signal?.removeEventListener('abort', onAbort);
    };
    const onAbort = () => {
      if (settled) return;
      settled = true;
      cleanup();
      if (terminateOnAbort) worker.terminate();
      reject(abortError(signal!));
    };
    const onError = (ev: Event) => {
      ev.preventDefault?.();
      if (signal?.aborted) return onAbort();
      if (settled) return;
      settled = true;
      cleanup();
      resolve({ type: 'error', id: req.id, kind: 'transport' });
    };
    const onMessage = (ev: MessageEvent<ExternalWorkerResponse>) => {
      const res = ev.data;
      if (!res || typeof res !== 'object') {
        if (settled) return;
        settled = true;
        cleanup();
        resolve({ type: 'error', id: req.id, kind: 'transport' });
        return;
      }
      if (res.id !== req.id) return;
      if (settled) return;
      settled = true;
      cleanup();
      resolve(res);
    };

    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);
    worker.addEventListener('messageerror', onError);
    signal?.addEventListener('abort', onAbort, { once: true });

    try {
      worker.postMessage(req);
    } catch {
      if (!settled) {
        settled = true;
        cleanup();
        resolve({ type: 'error', id: req.id, kind: 'transport' });
      }
    }
  });
}

function createWorkerSession(
  worker: ExternalParseWorker,
  summary: PredictionSummary,
  total: number,
  partsCount: number,
): ExternalPredictionSession {
  let disposed = false;
  let nextReqId = 2;
  const disposeCtl = new AbortController();

  const dispose = () => {
    if (disposed) return;
    disposed = true;
    disposeCtl.abort(new Error('The predictions file session was closed.'));
    worker.terminate();
  };

  const fetchList = async (req: ExternalWorkerRequest, signal?: AbortSignal): Promise<BenchPrediction[]> => {
    if (disposed) throw new Error('The predictions file session was closed.');
    const combined = signal ? AbortSignal.any([signal, disposeCtl.signal]) : disposeCtl.signal;
    const res = await sendWorkerRequest(worker, req, combined);
    if (res.type === 'predictions') return res.predictions;
    throw new Error(res.type === 'error' && res.message ? res.message : 'Could not read predictions from worker.');
  };

  return {
    summary,
    total,
    partsCount,
    getPredictions: (signal) => fetchList({ type: 'get-all', id: nextReqId++ }, signal),
    getPart: (index, signal) => fetchList({ type: 'get-part', id: nextReqId++, index }, signal),
    dispose,
  };
}

function createMemorySession(
  predictions: BenchPrediction[],
  summary: PredictionSummary,
  ctx: PredictionContext,
  maxPartBytes: number,
): ExternalPredictionSession {
  let small: BenchPrediction[] | null = estimateBytes(predictions) > maxPartBytes ? null : predictions;
  let parts: BenchPrediction[][] | null = small ? null : partsOf(predictions, ctx.plan, maxPartBytes);
  const total = predictions.length;
  const partsCount = parts ? parts.length : 0;

  return {
    summary,
    total,
    partsCount,
    getPredictions: async (signal) => {
      if (signal?.aborted) throw abortError(signal);
      if (!small) throw new Error('The predictions file session was closed.');
      return small;
    },
    getPart: async (index, signal) => {
      if (signal?.aborted) throw abortError(signal);
      const part = parts?.[index];
      if (!part) throw new Error('The predictions file session was closed.');
      return part;
    },
    dispose: () => {
      small = null;
      parts = null;
    },
  };
}

type WorkerParseOutcome = { fallback: true } | { fallback: false; result: ExternalParseResult };

async function runParseInWorker(
  file: File,
  ctx: PredictionContext,
  worker: ExternalParseWorker,
  maxPartBytes: number,
  signal?: AbortSignal,
): Promise<WorkerParseOutcome> {
  if (signal?.aborted) {
    worker.terminate();
    throw abortError(signal);
  }
  let res: ExternalWorkerResponse;
  try {
    res = await sendWorkerRequest(worker, { type: 'parse', id: 1, file, ctx, maxPartBytes }, signal, true);
  } catch (err) {
    worker.terminate();
    throw err;
  }

  if (res.type === 'parsed') {
    if (!res.ok) {
      worker.terminate();
      return { fallback: false, result: { ok: false, error: res.error } };
    }
    return {
      fallback: false,
      result: {
        ok: true,
        session: createWorkerSession(worker, res.summary, res.totalRecords, res.partsCount),
      },
    };
  }

  worker.terminate();
  if (res.type === 'error') {
    if (res.kind === 'read') return { fallback: false, result: { ok: false, error: 'The file could not be read.' } };
    if (res.kind === 'parse') {
      return {
        fallback: false,
        result: {
          ok: false,
          error: res.message ? `The file could not be parsed: ${res.message}.` : 'The file could not be read.',
        },
      };
    }
  }
  return { fallback: true };
}

async function parseOnMainThread(
  file: File,
  ctx: PredictionContext,
  maxPartBytes: number,
  signal?: AbortSignal,
  yieldFirst = true,
): Promise<ExternalParseResult> {
  if (signal?.aborted) throw abortError(signal);
  if (yieldFirst) await yieldPaint(signal);
  if (signal?.aborted) throw abortError(signal);
  let text: string;
  try {
    text = await raceAbort(file.text(), signal);
  } catch {
    if (signal?.aborted) throw abortError(signal);
    return { ok: false, error: 'The file could not be read.' };
  }
  if (signal?.aborted) throw abortError(signal);
  try {
    const check = parseAndCheckPredictions(text, ctx);
    if (!check.ok) return { ok: false, error: check.error };
    return {
      ok: true,
      session: createMemorySession(check.predictions, check.summary, ctx, maxPartBytes),
    };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { ok: false, error: msg ? `The file could not be parsed: ${msg}.` : 'The file could not be read.' };
  }
}

/**
 * Reads, parses, and validates an external predictions file against `ctx`.
 * Uses a dedicated Web Worker when available so large files never block the UI thread; keeps the
 * validated records inside the worker and exposes a lightweight `ExternalPredictionSession` that serves
 * bounded upload parts on demand. Terminates the worker immediately on invalid files, cancellation, or
 * `session.dispose()`, and falls back to the main thread when Workers are unavailable or fail transport.
 */
export async function parseExternalPredictionsFile(
  file: File,
  ctx: PredictionContext,
  options: ParseExternalOptions = {},
): Promise<ExternalParseResult> {
  const { signal, createWorker = defaultCreateWorker, yieldOnFallback = true, maxPartBytes = PART_BYTES } = options;
  if (signal?.aborted) throw abortError(signal);

  let worker: ExternalParseWorker | null = null;
  try {
    worker = createWorker();
  } catch {
    worker = null;
  }

  if (worker) {
    const outcome = await runParseInWorker(file, ctx, worker, maxPartBytes, signal);
    if (!outcome.fallback) return outcome.result;
    if (signal?.aborted) throw abortError(signal);
  }

  return parseOnMainThread(file, ctx, maxPartBytes, signal, yieldOnFallback);
}
