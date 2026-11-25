// Runs external benchmark prediction JSON reading, parsing, and validation in a dedicated Web Worker
// so large files never block the UI thread, with a clean main-thread fallback and AbortSignal cancellation.
import { parseAndCheckPredictions, type PredictionCheck, type PredictionContext } from './external';
import type { ExternalParseRequest, ExternalParseResponse } from './externalWorker';

/** Minimal subset of the standard Worker interface used by `parseExternalPredictionsFile`. */
export interface ExternalParseWorker {
  postMessage(message: ExternalParseRequest): void;
  terminate(): void;
  addEventListener(type: 'message', listener: (ev: MessageEvent<ExternalParseResponse>) => void): void;
  addEventListener(type: 'error', listener: (ev: Event) => void): void;
  addEventListener(type: 'messageerror', listener: (ev: Event) => void): void;
  removeEventListener(type: 'message', listener: (ev: MessageEvent<ExternalParseResponse>) => void): void;
  removeEventListener(type: 'error', listener: (ev: Event) => void): void;
  removeEventListener(type: 'messageerror', listener: (ev: Event) => void): void;
}

export interface ParseExternalOptions {
  /** Aborting terminates an in-flight worker immediately and rejects with an AbortError. */
  signal?: AbortSignal;
  /** Optional worker factory (defaults to spawning `./externalWorker.ts` when `Worker` exists). */
  createWorker?: () => ExternalParseWorker | null;
  /** When falling back to the main thread, yield a paint frame before parsing. Default true. */
  yieldOnFallback?: boolean;
}

/** True when `err` represents a cancelled parse (DOMException/Error with name `AbortError`). */
export function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === 'AbortError';
}

function abortError(signal: AbortSignal): Error {
  if (signal.reason instanceof Error) return signal.reason;
  const err = new DOMException('Aborted', 'AbortError');
  return err;
}

/** Races `p` against `signal` without leaving an unhandled rejection if `p` settles after abort. */
function raceAbort<T>(p: Promise<T>, signal?: AbortSignal): Promise<T> {
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

type WorkerOutcome = { fallback: true } | { fallback: false; check: PredictionCheck };

function runInWorker(
  file: File,
  ctx: PredictionContext,
  worker: ExternalParseWorker,
  signal?: AbortSignal,
): Promise<WorkerOutcome> {
  if (signal?.aborted) {
    worker.terminate();
    return Promise.reject(abortError(signal));
  }
  return new Promise<WorkerOutcome>((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      worker.removeEventListener('message', onMessage);
      worker.removeEventListener('error', onError);
      worker.removeEventListener('messageerror', onError);
      signal?.removeEventListener('abort', onAbort);
      worker.terminate();
    };
    const finish = (outcome: WorkerOutcome) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(outcome);
    };
    const onAbort = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(abortError(signal!));
    };
    const onError = () => {
      if (signal?.aborted) return onAbort();
      finish({ fallback: true });
    };
    const onMessage = (ev: MessageEvent<ExternalParseResponse>) => {
      const res = ev.data;
      if (!res || typeof res !== 'object') {
        finish({ fallback: true });
        return;
      }
      if (res.ok) {
        finish({ fallback: false, check: res.check });
      } else if (res.kind === 'read') {
        finish({ fallback: false, check: { ok: false, error: 'The file could not be read.' } });
      } else {
        finish({
          fallback: false,
          check: { ok: false, error: res.message ? `The file could not be parsed: ${res.message}.` : 'The file could not be read.' },
        });
      }
    };

    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);
    worker.addEventListener('messageerror', onError);
    signal?.addEventListener('abort', onAbort, { once: true });

    try {
      worker.postMessage({ file, ctx });
    } catch {
      finish({ fallback: true });
    }
  });
}

async function parseOnMainThread(
  file: File,
  ctx: PredictionContext,
  signal?: AbortSignal,
  yieldFirst = true,
): Promise<PredictionCheck> {
  if (signal?.aborted) throw abortError(signal);
  if (yieldFirst) await yieldPaint(signal);
  if (signal?.aborted) throw abortError(signal);
  let text: string;
  try {
    text = await raceAbort(file.text(), signal);
  } catch (err) {
    if (signal?.aborted || isAbortError(err)) throw signal?.aborted ? abortError(signal) : err;
    return { ok: false, error: 'The file could not be read.' };
  }
  if (signal?.aborted) throw abortError(signal);
  return parseAndCheckPredictions(text, ctx);
}

/**
 * Reads, parses, and validates an external predictions file against `ctx`.
 * Uses a dedicated Web Worker when available so large files never block the UI thread; terminates the worker
 * immediately on completion or when `signal` aborts, and falls back to the main thread when Workers are
 * unavailable or fail to initialize/communicate.
 */
export async function parseExternalPredictionsFile(
  file: File,
  ctx: PredictionContext,
  options: ParseExternalOptions = {},
): Promise<PredictionCheck> {
  const { signal, createWorker = defaultCreateWorker, yieldOnFallback = true } = options;
  if (signal?.aborted) throw abortError(signal);

  let worker: ExternalParseWorker | null = null;
  try {
    worker = createWorker();
  } catch {
    worker = null;
  }

  if (worker) {
    const outcome = await runInWorker(file, ctx, worker, signal);
    if (!outcome.fallback) return outcome.check;
    if (signal?.aborted) throw abortError(signal);
  }

  return parseOnMainThread(file, ctx, signal, yieldOnFallback);
}
