// Web Worker that reads, parses, and validates an external benchmark predictions JSON file off the UI thread,
// keeping parsed predictions inside the worker and serving bounded upload parts on demand so large files never
// block the UI thread during structured cloning.
import type { BenchPrediction } from '../api/types';
import {
  estimateBytes,
  PART_BYTES,
  parseAndCheckPredictions,
  partsOf,
  type PredictionContext,
  type PredictionSummary,
} from './external';

/** Request sent from the main thread to the external predictions parser worker. */
export type ExternalWorkerRequest =
  | { type: 'parse'; id: number; file: File; ctx: PredictionContext; maxPartBytes?: number }
  | { type: 'get-all'; id: number }
  | { type: 'get-part'; id: number; index: number };

/** Response sent back from the external predictions parser worker. */
export type ExternalWorkerResponse =
  | { type: 'parsed'; id: number; ok: true; summary: PredictionSummary; totalRecords: number; partsCount: number }
  | { type: 'parsed'; id: number; ok: false; error: string }
  | { type: 'predictions'; id: number; predictions: BenchPrediction[] }
  | { type: 'error'; id: number; kind: 'read' | 'parse' | 'transport' | 'state'; message?: string };

/** Minimal worker global scope interface used to send responses back to the main thread. */
export interface ExternalWorkerScope {
  addEventListener(type: 'message', listener: (ev: MessageEvent<ExternalWorkerRequest>) => void): void;
  postMessage(message: ExternalWorkerResponse): void;
}

/**
 * Creates a stateful message handler for one worker instance.
 * Validated predictions stay in the worker (grouped into bounded parts when large) so only the tiny
 * summary crosses threads on parse, and bounded parts are transferred on demand when submitting.
 */
export function createExternalWorkerHandler(): (req: ExternalWorkerRequest) => Promise<ExternalWorkerResponse> {
  let storedSmall: BenchPrediction[] | null = null;
  let storedParts: BenchPrediction[][] | null = null;

  return async (req: ExternalWorkerRequest): Promise<ExternalWorkerResponse> => {
    if (req.type === 'parse') {
      storedSmall = null;
      storedParts = null;
      let text: string;
      try {
        text = await req.file.text();
      } catch {
        return { type: 'error', id: req.id, kind: 'read' };
      }
      try {
        const check = parseAndCheckPredictions(text, req.ctx);
        if (!check.ok) return { type: 'parsed', id: req.id, ok: false, error: check.error };
        const maxPartBytes = req.maxPartBytes ?? PART_BYTES;
        if (estimateBytes(check.predictions) > maxPartBytes) {
          storedParts = partsOf(check.predictions, req.ctx.plan, maxPartBytes);
          return {
            type: 'parsed',
            id: req.id,
            ok: true,
            summary: check.summary,
            totalRecords: check.predictions.length,
            partsCount: storedParts.length,
          };
        }
        storedSmall = check.predictions;
        return {
          type: 'parsed',
          id: req.id,
          ok: true,
          summary: check.summary,
          totalRecords: check.predictions.length,
          partsCount: 0,
        };
      } catch (err) {
        return {
          type: 'error',
          id: req.id,
          kind: 'parse',
          message: err instanceof Error ? err.message : String(err),
        };
      }
    }

    if (req.type === 'get-all') {
      if (!storedSmall) {
        return { type: 'error', id: req.id, kind: 'state', message: 'Validated predictions are not available.' };
      }
      return { type: 'predictions', id: req.id, predictions: storedSmall };
    }

    if (req.type === 'get-part') {
      const part = storedParts?.[req.index];
      if (!part) {
        return { type: 'error', id: req.id, kind: 'state', message: `Prediction part ${req.index} is not available.` };
      }
      return { type: 'predictions', id: req.id, predictions: part };
    }

    return { type: 'error', id: (req as { id?: number }).id ?? 0, kind: 'state', message: 'Unknown worker request.' };
  };
}

/**
 * Runs one worker request and posts the reply safely: if `scope.postMessage` throws (e.g. structured-clone
 * RangeError/DataCloneError), catches it and posts a small transport-error reply so the caller never hangs.
 */
export async function dispatchExternalWorkerMessage(
  scope: Pick<ExternalWorkerScope, 'postMessage'>,
  handler: (req: ExternalWorkerRequest) => Promise<ExternalWorkerResponse>,
  req: ExternalWorkerRequest,
): Promise<void> {
  let res: ExternalWorkerResponse;
  try {
    res = await handler(req);
  } catch (err) {
    res = {
      type: 'error',
      id: req.id,
      kind: 'parse',
      message: err instanceof Error ? err.message : String(err),
    };
  }
  try {
    scope.postMessage(res);
  } catch (err) {
    try {
      scope.postMessage({
        type: 'error',
        id: req.id,
        kind: 'transport',
        message: err instanceof Error ? err.message : String(err),
      });
    } catch {
      // Worker channel is already broken or closed.
    }
  }
}

// Register the message listener only when executing inside a Web Worker global scope.
if (typeof document === 'undefined' && typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  const scope = self as unknown as ExternalWorkerScope;
  const handler = createExternalWorkerHandler();
  scope.addEventListener('message', (ev) => {
    void dispatchExternalWorkerMessage(scope, handler, ev.data);
  });
}
