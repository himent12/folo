// Unit tests for worker-offloaded external benchmark predictions parsing, fallback behavior, and cancellation.
import { describe, expect, it, vi } from 'vitest';
import type { BenchPlan } from '../api/types';
import type { PredictionContext } from './external';
import { isAbortError, parseExternalPredictionsFile, type ExternalParseWorker } from './externalParse';
import { runExternalParseRequest, type ExternalParseRequest, type ExternalParseResponse } from './externalWorker';

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

const validJson = JSON.stringify([
  { image: '1', boxes: [{ class: 0, score: 0.9, x1: 1, y1: 2, x2: 30, y2: 40 }] },
  { image: '2', boxes: [] },
]);

const invalidJson = JSON.stringify({
  predictions: [{ image: '1', boxes: [{ class: 9, score: 0.5, x1: 0, y1: 0, x2: 10, y2: 10 }] }],
});

/** Minimal in-process fake Worker that lets tests control message/error delivery and inspect termination. */
class FakeWorker implements ExternalParseWorker {
  readonly posted: ExternalParseRequest[] = [];
  terminated = false;
  readonly listeners = {
    message: new Set<(ev: MessageEvent<ExternalParseResponse>) => void>(),
    error: new Set<(ev: Event) => void>(),
    messageerror: new Set<(ev: Event) => void>(),
  };
  onPost?: (req: ExternalParseRequest, worker: FakeWorker) => void;

  postMessage(message: ExternalParseRequest) {
    this.posted.push(message);
    this.onPost?.(message, this);
  }

  terminate() {
    this.terminated = true;
  }

  addEventListener(type: 'message', listener: (ev: MessageEvent<ExternalParseResponse>) => void): void;
  addEventListener(type: 'error' | 'messageerror', listener: (ev: Event) => void): void;
  addEventListener(type: 'message' | 'error' | 'messageerror', listener: unknown): void {
    (this.listeners[type] as Set<unknown>).add(listener);
  }

  removeEventListener(type: 'message', listener: (ev: MessageEvent<ExternalParseResponse>) => void): void;
  removeEventListener(type: 'error' | 'messageerror', listener: (ev: Event) => void): void;
  removeEventListener(type: 'message' | 'error' | 'messageerror', listener: unknown): void {
    (this.listeners[type] as Set<unknown>).delete(listener);
  }

  emitMessage(data: ExternalParseResponse) {
    const ev = { data } as MessageEvent<ExternalParseResponse>;
    for (const fn of [...this.listeners.message]) fn(ev);
  }

  emitError(type: 'error' | 'messageerror' = 'error') {
    const ev = new Event(type);
    for (const fn of [...this.listeners[type]]) fn(ev);
  }
}

describe('runExternalParseRequest', () => {
  it('reads, parses, and validates valid and invalid files cleanly', async () => {
    const okRes = await runExternalParseRequest({ file: new File([validJson], 'ok.json'), ctx });
    expect(okRes.ok).toBe(true);
    if (okRes.ok) {
      expect(okRes.check.ok).toBe(true);
      if (okRes.check.ok) {
        expect(okRes.check.summary).toEqual({ images: 2, conditions: [], missing: ['haze', 'jpeg'], boxes: 1, scores: true });
      }
    }

    const badRes = await runExternalParseRequest({ file: new File([invalidJson], 'bad.json'), ctx });
    expect(badRes.ok).toBe(true);
    if (badRes.ok) {
      expect(badRes.check.ok).toBe(false);
      if (!badRes.check.ok) expect(badRes.check.error).toMatch(/^Record 1: box 1/);
    }
  });

  it('reports file read failures as kind="read"', async () => {
    const broken = new File(['{}'], 'broken.json');
    vi.spyOn(broken, 'text').mockRejectedValue(new Error('disk error'));
    expect(await runExternalParseRequest({ file: broken, ctx })).toEqual({ ok: false, kind: 'read' });
  });
});

describe('parseExternalPredictionsFile', () => {
  it('offloads parsing to the worker and terminates it once done', async () => {
    const worker = new FakeWorker();
    worker.onPost = (req, w) => {
      void runExternalParseRequest(req).then((res) => w.emitMessage(res));
    };

    const file = new File([validJson], 'preds.json', { type: 'application/json' });
    const result = await parseExternalPredictionsFile(file, ctx, { createWorker: () => worker });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.summary.images).toBe(2);
      expect(result.summary.boxes).toBe(1);
    }
    expect(worker.posted).toHaveLength(1);
    expect(worker.posted[0]!.file).toBe(file);
    expect(worker.terminated).toBe(true);
    expect(worker.listeners.message.size).toBe(0);
    expect(worker.listeners.error.size).toBe(0);
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

    // Late messages after cancellation are ignored safely.
    worker.emitMessage({ ok: false, kind: 'read' });
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

  it('falls back to main thread when Workers are unavailable or fail to start', async () => {
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
  });

  it('falls back to main thread if worker emits an error or postMessage throws', async () => {
    const file = new File([validJson], 'preds.json');

    const errorWorker = new FakeWorker();
    errorWorker.onPost = (_req, w) => queueMicrotask(() => w.emitError('error'));
    const res1 = await parseExternalPredictionsFile(file, ctx, {
      createWorker: () => errorWorker,
      yieldOnFallback: false,
    });
    expect(res1.ok).toBe(true);
    expect(errorWorker.terminated).toBe(true);

    const cloneErrorWorker = new FakeWorker();
    cloneErrorWorker.onPost = () => {
      throw new DOMException('Cannot clone', 'DataCloneError');
    };
    const res2 = await parseExternalPredictionsFile(file, ctx, {
      createWorker: () => cloneErrorWorker,
      yieldOnFallback: false,
    });
    expect(res2.ok).toBe(true);
    expect(cloneErrorWorker.terminated).toBe(true);
  });

  it('returns a read error message without main-thread fallback when the worker reports a read failure', async () => {
    const worker = new FakeWorker();
    worker.onPost = (_req, w) => queueMicrotask(() => w.emitMessage({ ok: false, kind: 'read' }));
    const file = new File([validJson], 'unreadable.json');
    const textSpy = vi.spyOn(file, 'text');

    const res = await parseExternalPredictionsFile(file, ctx, { createWorker: () => worker });
    expect(res).toEqual({ ok: false, error: 'The file could not be read.' });
    expect(textSpy).not.toHaveBeenCalled();
    expect(worker.terminated).toBe(true);
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
