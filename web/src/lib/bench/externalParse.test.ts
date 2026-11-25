// Unit tests for worker-offloaded external benchmark predictions parsing, bounded part streaming, fallback, and cancellation.
import { describe, expect, it, vi } from 'vitest';
import type { BenchEntry, BenchPlan } from '../api/types';
import { sendExternal, type ExternalSender, type PredictionContext } from './external';
import { isAbortError, parseExternalPredictionsFile, type ExternalParseWorker } from './externalParse';
import {
  createExternalWorkerHandler,
  dispatchExternalWorkerMessage,
  type ExternalWorkerRequest,
  type ExternalWorkerResponse,
} from './externalWorker';

const plan: BenchPlan = {
  conditions: [
    { condition: 'haze', severities: [1, 3], weight: 1 },
    { condition: 'jpeg', severities: [5], weight: 1 },
  ],
};

const ctx: PredictionContext = {
  imageIds: ['1', '2'],
  classes: 3,
  plan,
  catalog: ['haze', 'jpeg', 'low_light'],
  names: { haze: 'Haze', jpeg: 'JPEG compression', low_light: 'Low light' },
};

const validRecords = [
  { image: '1', boxes: [{ class: 0, score: 0.9, x1: 1, y1: 2, x2: 30, y2: 40 }] },
  { image: '2', boxes: [] },
];
const validJson = JSON.stringify(validRecords);

const invalidJson = JSON.stringify({
  predictions: [{ image: '1', boxes: [{ class: 9, score: 0.5, x1: 0, y1: 0, x2: 10, y2: 10 }] }],
});

/** Minimal in-process fake Worker that lets tests control message/error delivery and inspect termination. */
class FakeWorker implements ExternalParseWorker {
  readonly posted: ExternalWorkerRequest[] = [];
  readonly responses: ExternalWorkerResponse[] = [];
  terminated = false;
  readonly listeners = {
    message: new Set<(ev: MessageEvent<ExternalWorkerResponse>) => void>(),
    error: new Set<(ev: Event) => void>(),
    messageerror: new Set<(ev: Event) => void>(),
  };
  onPost?: (req: ExternalWorkerRequest, worker: FakeWorker) => void;

  postMessage(message: ExternalWorkerRequest) {
    this.posted.push(message);
    this.onPost?.(message, this);
  }

  terminate() {
    this.terminated = true;
  }

  addEventListener(type: 'message', listener: (ev: MessageEvent<ExternalWorkerResponse>) => void): void;
  addEventListener(type: 'error' | 'messageerror', listener: (ev: Event) => void): void;
  addEventListener(type: 'message' | 'error' | 'messageerror', listener: unknown): void {
    (this.listeners[type] as Set<unknown>).add(listener);
  }

  removeEventListener(type: 'message', listener: (ev: MessageEvent<ExternalWorkerResponse>) => void): void;
  removeEventListener(type: 'error' | 'messageerror', listener: (ev: Event) => void): void;
  removeEventListener(type: 'message' | 'error' | 'messageerror', listener: unknown): void {
    (this.listeners[type] as Set<unknown>).delete(listener);
  }

  emitMessage(data: ExternalWorkerResponse) {
    this.responses.push(data);
    const ev = { data } as MessageEvent<ExternalWorkerResponse>;
    for (const fn of [...this.listeners.message]) fn(ev);
  }

  emitError(type: 'error' | 'messageerror' = 'error') {
    const ev = new Event(type, { cancelable: true });
    for (const fn of [...this.listeners[type]]) fn(ev);
    return ev;
  }
}

/** Wires a FakeWorker directly to the real worker handler and dispatch logic. */
function attachRealHandler(worker: FakeWorker) {
  const handler = createExternalWorkerHandler();
  worker.onPost = (req, w) => {
    void dispatchExternalWorkerMessage(
      { postMessage: (res) => w.emitMessage(res) },
      handler,
      req,
    );
  };
  return worker;
}

describe('createExternalWorkerHandler and dispatchExternalWorkerMessage', () => {
  it('returns only summary metadata on parse, serves bounded parts on demand, and reports read errors', async () => {
    const handler = createExternalWorkerHandler();
    const okRes = await handler({ type: 'parse', id: 1, file: new File([validJson], 'ok.json'), ctx, maxPartBytes: 200 });
    expect(okRes).toEqual({
      type: 'parsed',
      id: 1,
      ok: true,
      summary: { images: 2, conditions: [], missing: ['haze', 'jpeg'], boxes: 1, scores: true },
      totalRecords: 2,
      partsCount: 2,
    });
    expect('predictions' in okRes).toBe(false);

    const p0 = await handler({ type: 'get-part', id: 2, index: 0 });
    const p1 = await handler({ type: 'get-part', id: 3, index: 1 });
    expect(p0.type === 'predictions' && p0.predictions).toHaveLength(1);
    expect(p1.type === 'predictions' && p1.predictions).toHaveLength(1);

    const badRes = await handler({ type: 'parse', id: 4, file: new File([invalidJson], 'bad.json'), ctx });
    expect(badRes.type === 'parsed' && !badRes.ok && badRes.error).toMatch(/^Record 1: box 1/);

    const broken = new File(['{}'], 'broken.json');
    vi.spyOn(broken, 'text').mockRejectedValue(new Error('disk error'));
    expect(await handler({ type: 'parse', id: 5, file: broken, ctx })).toEqual({ type: 'error', id: 5, kind: 'read' });
  });

  it('catches throwing scope.postMessage and sends a small transport error instead of hanging', async () => {
    const handler = createExternalWorkerHandler();
    const replies: ExternalWorkerResponse[] = [];
    let first = true;
    await dispatchExternalWorkerMessage(
      {
        postMessage(res) {
          if (first) {
            first = false;
            throw new RangeError('Maximum call stack size exceeded');
          }
          replies.push(res);
        },
      },
      handler,
      { type: 'parse', id: 7, file: new File([validJson], 'ok.json'), ctx },
    );

    expect(replies).toEqual([
      { type: 'error', id: 7, kind: 'transport', message: 'Maximum call stack size exceeded' },
    ]);
  });
});

describe('parseExternalPredictionsFile', () => {
  it('keeps validated predictions in the worker, streams bounded parts via sendExternal, and terminates on dispose', async () => {
    const worker = attachRealHandler(new FakeWorker());
    const file = new File([validJson], 'preds.json', { type: 'application/json' });

    const res = await parseExternalPredictionsFile(file, ctx, {
      createWorker: () => worker,
      maxPartBytes: 200,
    });

    expect(res.ok).toBe(true);
    if (!res.ok) return;
    expect(res.session.summary).toEqual({ images: 2, conditions: [], missing: ['haze', 'jpeg'], boxes: 1, scores: true });
    expect(res.session.total).toBe(2);
    expect(res.session.partsCount).toBe(2);
    // Parse response only carried summary metadata, not the full predictions graph.
    expect(worker.responses[0]?.type).toBe('parsed');
    expect('predictions' in (worker.responses[0] ?? {})).toBe(false);
    expect(worker.terminated).toBe(false);

    const sentParts: number[] = [];
    const entry = { id: 'be_1', status: 'done' } as BenchEntry;
    const sender: ExternalSender = {
      start: async () => ({ id: 'be_1', status: 'running' }) as BenchEntry,
      send: async (_id, ps) => {
        sentParts.push(ps.length);
        return { id: 'be_1', status: 'running' } as BenchEntry;
      },
      finish: async () => entry,
      get: async () => entry,
      stop: async () => entry,
      remove: async () => {},
    };

    const done = await sendExternal(sender, { name: 'ext', scored: true, source: res.session, plan });
    expect(done.status).toBe('done');
    expect(sentParts).toEqual([1, 1]);

    res.session.dispose();
    expect(worker.terminated).toBe(true);
    await expect(res.session.getPart(0)).rejects.toThrow(/closed/);
  });

  it('terminates the worker immediately when validation fails', async () => {
    const worker = attachRealHandler(new FakeWorker());
    const res = await parseExternalPredictionsFile(new File([invalidJson], 'bad.json'), ctx, {
      createWorker: () => worker,
    });
    expect(res.ok).toBe(false);
    expect(worker.terminated).toBe(true);
  });

  it('terminates the worker immediately when AbortSignal fires and rejects with AbortError', async () => {
    const worker = new FakeWorker();
    const ctl = new AbortController();
    const file = new File([validJson], 'huge.json');

    const promise = parseExternalPredictionsFile(file, ctx, {
      signal: ctl.signal,
      createWorker: () => worker,
    });
    expect(worker.posted).toHaveLength(1);
    expect(worker.terminated).toBe(false);

    ctl.abort();
    await expect(promise).rejects.toSatisfy(isAbortError);
    expect(worker.terminated).toBe(true);
    expect(worker.listeners.message.size).toBe(0);

    worker.emitMessage({ type: 'error', id: 1, kind: 'read' });
  });

  it('rejects immediately without spawning a worker when signal is already aborted', async () => {
    const createWorker = vi.fn(() => new FakeWorker());
    const ctl = new AbortController();
    ctl.abort();

    await expect(
      parseExternalPredictionsFile(new File([validJson], 'preds.json'), ctx, {
        signal: ctl.signal,
        createWorker,
      }),
    ).rejects.toSatisfy(isAbortError);
    expect(createWorker).not.toHaveBeenCalled();
  });

  it('falls back to main thread when Workers are unavailable, throw, emit error, or reply with transport error', async () => {
    const file = new File([validJson], 'preds.json');

    const noWorker = await parseExternalPredictionsFile(file, ctx, {
      createWorker: () => null,
      yieldOnFallback: false,
    });
    expect(noWorker.ok).toBe(true);

    const ctorThrows = await parseExternalPredictionsFile(file, ctx, {
      createWorker: () => {
        throw new Error('CSP blocked worker');
      },
      yieldOnFallback: false,
    });
    expect(ctorThrows.ok).toBe(true);

    let errorEv: Event | undefined;
    const errorWorker = new FakeWorker();
    errorWorker.onPost = (_req, w) => queueMicrotask(() => (errorEv = w.emitError('error')));
    const res1 = await parseExternalPredictionsFile(file, ctx, {
      createWorker: () => errorWorker,
      yieldOnFallback: false,
    });
    expect(res1.ok).toBe(true);
    expect(errorWorker.terminated).toBe(true);
    expect(errorEv?.defaultPrevented).toBe(true);

    const transportWorker = new FakeWorker();
    transportWorker.onPost = (req, w) =>
      queueMicrotask(() => w.emitMessage({ type: 'error', id: req.id, kind: 'transport', message: 'clone failed' }));
    const res2 = await parseExternalPredictionsFile(file, ctx, {
      createWorker: () => transportWorker,
      yieldOnFallback: false,
    });
    expect(res2.ok).toBe(true);
    expect(transportWorker.terminated).toBe(true);
  });

  it('returns a read error message without main-thread fallback when the worker reports a read failure', async () => {
    const worker = new FakeWorker();
    worker.onPost = (req, w) => queueMicrotask(() => w.emitMessage({ type: 'error', id: req.id, kind: 'read' }));
    const file = new File([validJson], 'unreadable.json');
    const textSpy = vi.spyOn(file, 'text');

    const res = await parseExternalPredictionsFile(file, ctx, { createWorker: () => worker });
    expect(res).toEqual({ ok: false, error: 'The file could not be read.' });
    expect(textSpy).not.toHaveBeenCalled();
    expect(worker.terminated).toBe(true);
  });

  it('treats a genuine File.text() AbortError with a non-aborted signal as a read error rather than cancellation', async () => {
    const file = new File(['{}'], 'aborted-io.json');
    vi.spyOn(file, 'text').mockRejectedValue(new DOMException('read aborted', 'AbortError'));
    const ctl = new AbortController();

    const res = await parseExternalPredictionsFile(file, ctx, {
      signal: ctl.signal,
      createWorker: () => null,
      yieldOnFallback: false,
    });
    expect(res).toEqual({ ok: false, error: 'The file could not be read.' });
  });

  it('cancels a pending main-thread fallback read cleanly without unhandled rejections', async () => {
    let rejectRead!: (err: unknown) => void;
    const file = new File(['{}'], 'slow.json');
    vi.spyOn(file, 'text').mockImplementation(
      () =>
        new Promise<string>((_res, rej) => {
          rejectRead = rej;
        }),
    );

    const ctl = new AbortController();
    const p = parseExternalPredictionsFile(file, ctx, {
      signal: ctl.signal,
      createWorker: () => null,
      yieldOnFallback: false,
    });

    ctl.abort();
    await expect(p).rejects.toSatisfy(isAbortError);

    // Late rejection from the abandoned File.text() must not become an unhandled rejection.
    rejectRead(new Error('late disk failure'));
    await new Promise((r) => setTimeout(r, 0));
  });
});
