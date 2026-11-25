// Web Worker that reads, parses, and validates an external benchmark predictions JSON file off the UI thread.
import { parseAndCheckPredictions, type PredictionCheck, type PredictionContext } from './external';

/** Request sent from the main thread to the external predictions parser worker. */
export interface ExternalParseRequest {
  file: File;
  ctx: PredictionContext;
}

/** Response sent back from the external predictions parser worker. */
export type ExternalParseResponse =
  | { ok: true; check: PredictionCheck }
  | { ok: false; kind: 'read' | 'error'; message?: string };

/** Reads a predictions File, parses its JSON, and validates all records against the benchmark context. */
export async function runExternalParseRequest(req: ExternalParseRequest): Promise<ExternalParseResponse> {
  let text: string;
  try {
    text = await req.file.text();
  } catch {
    return { ok: false, kind: 'read' };
  }
  try {
    return { ok: true, check: parseAndCheckPredictions(text, req.ctx) };
  } catch (err) {
    return { ok: false, kind: 'error', message: err instanceof Error ? err.message : String(err) };
  }
}

interface WorkerScope {
  addEventListener(type: 'message', listener: (ev: MessageEvent<ExternalParseRequest>) => void): void;
  postMessage(message: ExternalParseResponse): void;
}

// Register the message listener only when executing inside a Web Worker global scope.
if (typeof document === 'undefined' && typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  const scope = self as unknown as WorkerScope;
  scope.addEventListener('message', (ev) => {
    void runExternalParseRequest(ev.data).then((res) => scope.postMessage(res));
  });
}
