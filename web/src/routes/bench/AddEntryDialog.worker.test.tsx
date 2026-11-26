// @vitest-environment happy-dom
// Verifies AddEntryDialog offloads external predictions JSON parsing to a Web Worker, coordinates in-flight
// image ID loading, ignores stale worker replies, and terminates workers on re-pick, submit, close, and unmount.
import { createRoot, flush as flushSync, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Bench, BenchImagePage } from '$lib/api/types';
import { router } from '$lib/router';
import {
  createExternalWorkerHandler,
  dispatchExternalWorkerMessage,
  type ExternalWorkerRequest,
  type ExternalWorkerResponse,
} from '$lib/bench/externalWorker';
import { AgentChoice } from '../label/agentChoice';
import AddEntryDialog from './AddEntryDialog';

interface RecordedWorker {
  posted: ExternalWorkerRequest[];
  terminated: boolean;
  /** Manually delivers an arbitrary message (even after termination, to verify stale-reply guards). */
  deliver: (res: ExternalWorkerResponse) => void;
  /** Executes the real worker handler for the last posted request and delivers its response. */
  runAndDeliver: () => Promise<void>;
  /** Automatically answers every subsequent request using the real worker handler. */
  enableAutoAnswer: () => void;
}

let target: HTMLElement;
let c: Rendered | undefined;
let stopRoot: (() => void) | undefined;
let originalWorker: typeof globalThis.Worker | undefined;
let spawned: RecordedWorker[];

function installMockWorker() {
  spawned = [];
  originalWorker = globalThis.Worker;
  class StubWorker {
    readonly listeners = new Map<string, Set<(ev: unknown) => void>>();
    readonly handler = createExternalWorkerHandler();
    readonly rec: RecordedWorker;
    autoAnswer = false;

    constructor() {
      const rec: RecordedWorker = {
        posted: [],
        terminated: false,
        deliver: (res) => {
          for (const fn of [...(this.listeners.get('message') ?? [])]) fn({ data: res });
        },
        runAndDeliver: async () => {
          const req = rec.posted.at(-1);
          if (!req) return;
          await dispatchExternalWorkerMessage(
            { postMessage: (res) => rec.deliver(res) },
            this.handler,
            req,
          );
        },
        enableAutoAnswer: () => {
          this.autoAnswer = true;
        },
      };
      this.rec = rec;
      spawned.push(rec);
    }

    postMessage(req: ExternalWorkerRequest) {
      this.rec.posted.push(req);
      if (this.autoAnswer) {
        void dispatchExternalWorkerMessage(
          { postMessage: (res) => this.rec.deliver(res) },
          this.handler,
          req,
        );
      }
    }

    terminate() {
      this.rec.terminated = true;
    }

    addEventListener(type: string, fn: (ev: unknown) => void) {
      let set = this.listeners.get(type);
      if (!set) {
        set = new Set();
        this.listeners.set(type, set);
      }
      set.add(fn);
    }

    removeEventListener(type: string, fn: (ev: unknown) => void) {
      this.listeners.get(type)?.delete(fn);
    }
  }
  vi.stubGlobal('Worker', StubWorker);
}

/** Polls via microtasks/immediates until `cond()` becomes truthy (no fixed timer-turn counts). */
async function until<T>(cond: () => T | null | undefined | false, label: string): Promise<T> {
  const deadline = performance.now() + 3000;
  for (;;) {
    flushSync();
    const v = cond();
    if (v) return v;
    if (performance.now() > deadline) throw new Error(`Timed out waiting for ${label}`);
    await new Promise((r) => setImmediate(r));
  }
}

function selectFile(f: File) {
  const input = document.querySelector<HTMLInputElement>('input[type="file"]')!;
  Object.defineProperty(input, 'files', { configurable: true, value: [f] });
  input.dispatchEvent(new Event('change', { bubbles: true }));
  flushSync();
}

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
  installMockWorker();
});

afterEach(() => {
  if (c) c.unmount();
  c = undefined;
  stopRoot?.();
  stopRoot = undefined;
  target.remove();
  vi.unstubAllGlobals();
  if (originalWorker !== undefined) globalThis.Worker = originalWorker;
  vi.restoreAllMocks();
});

async function mountDialog(opts: { interceptBenchImages?: (id: string) => Promise<BenchImagePage> } = {}) {
  const bench: Bench = await hub.api.bench('bn_9f03');
  const models = await hub.api.models();
  const agents = await hub.api.labelAgents();
  const entries = await hub.api.benchEntries('bn_9f03');
  const catalog = await hub.api.benchCatalog();
  const imagesPage = await hub.api.benchImages('bn_9f03', { limit: 1000 });
  const validJson = JSON.stringify(imagesPage.images.map((i) => ({ image: i.id, boxes: i.boxes.map((b) => ({ ...b, score: 0.8 })) })));

  if (opts.interceptBenchImages) {
    vi.spyOn(hub.api, 'benchImages').mockImplementation((id) => opts.interceptBenchImages!(id));
  }

  let choice!: AgentChoice;
  stopRoot = createRoot((dispose) => {
    choice = new AgentChoice(() => agents);
    return dispose;
  });

  const isOpen = signal(true);
  const kind = signal<'model' | 'agent' | 'external'>('external');
  const onstarted = vi.fn();
  c = render(
    () => (
      <AddEntryDialog
        bind:open={isOpen}
        bind:kind={kind}
        bench={bench}
        catalog={catalog}
        entries={entries}
        models={models}
        modelsError={undefined}
        agents={agents}
        agentsError={undefined}
        retryAgents={() => {}}
        choice={choice}
        onstarted={onstarted}
      />
    ),
    target,
  );
  flushSync();
  return { isOpen, validJson, imagesPage, onstarted };
}

describe('AddEntryDialog worker offloading', () => {
  it('parses in a Worker, submits through the worker session, and terminates the worker when the dialog closes', async () => {
    const { validJson, onstarted } = await mountDialog();
    selectFile(new File([validJson], 'worker-model-predictions.json', { type: 'application/json' }));

    const worker = await until(() => spawned[0], 'first worker spawned');
    expect(worker.terminated).toBe(false);
    expect(document.querySelector('.f-state.busy')).not.toBeNull();

    await worker.runAndDeliver();
    await until(() => document.querySelector('.f-state.ok'), 'validation summary shown');
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('worker-model');
    // Session stays alive while dialog is open so submit can stream predictions from the worker.
    expect(worker.terminated).toBe(false);

    worker.enableAutoAnswer();
    const form = document.querySelector<HTMLFormElement>('dialog[open] form')!;
    form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await until(() => onstarted.mock.calls.length > 0, 'entry submitted');
    expect(onstarted).toHaveBeenCalledWith(expect.objectContaining({ status: 'done', name: 'worker-model' }));
    expect(worker.terminated).toBe(true);
  });

  it('terminates the previous worker on re-upload, ignores its late reply, and terminates on dialog close', async () => {
    const { isOpen, validJson } = await mountDialog();

    selectFile(new File([validJson], 'first-predictions.json', { type: 'application/json' }));
    const w1 = await until(() => spawned[0], 'worker #1 spawned');
    expect(w1.terminated).toBe(false);

    // Picking a second file cancels and terminates worker #1 immediately.
    const invalidJson = JSON.stringify({
      predictions: [{ image: '1', boxes: [{ class: 99, x1: 0, y1: 0, x2: 4, y2: 4 }] }],
    });
    selectFile(new File([invalidJson], 'second-predictions.json', { type: 'application/json' }));
    const w2 = await until(() => spawned[1], 'worker #2 spawned');
    expect(w1.terminated).toBe(true);
    expect(w2.terminated).toBe(false);

    // Even if worker #1 delivers a late valid reply after worker #2 started, it must not overwrite state.
    w1.deliver({
      type: 'parsed',
      id: 1,
      ok: true,
      summary: { images: 2, conditions: [], missing: [], boxes: 1, scores: true },
      totalRecords: 2,
      partsCount: 0,
    });
    flushSync();
    expect(document.querySelector('.f-state.ok')).toBeNull();
    expect(document.querySelector('.f-state.busy')).not.toBeNull();

    // Closing the dialog terminates worker #2 immediately.
    isOpen.set(false);
    flushSync();
    expect(w2.terminated).toBe(true);
  });

  it('terminates an in-flight worker and an active validated session when the dialog unmounts', async () => {
    const { validJson } = await mountDialog();

    selectFile(new File([validJson], 'unmount.json', { type: 'application/json' }));
    const worker = await until(() => spawned[0], 'worker spawned');
    await worker.runAndDeliver();
    await until(() => document.querySelector('.f-state.ok'), 'validated state');
    expect(worker.terminated).toBe(false);

    c!.unmount();
    c = undefined;
    expect(worker.terminated).toBe(true);
  });

  it('waits for an in-flight benchImages request before validating, and still validates if benchImages fails', async () => {
    let resolveImages!: (page: BenchImagePage) => void;
    const deferred = new Promise<BenchImagePage>((res) => (resolveImages = res));
    const { imagesPage } = await mountDialog({ interceptBenchImages: () => deferred });

    // Pick a file with an unknown image ID while benchImages is still in flight.
    const unknownImageJson = JSON.stringify([{ image: 'unknown-img-id', boxes: [] }]);
    selectFile(new File([unknownImageJson], 'early.json', { type: 'application/json' }));
    await until(() => document.querySelector('.f-state.busy'), 'reading state');
    // Worker has not started validating with null imageIds yet.
    expect(spawned).toHaveLength(0);

    // Resolve benchImages: worker now starts with the full imageIds list and rejects the unknown image ID.
    resolveImages(imagesPage);
    const worker = await until(() => spawned[0], 'worker spawned after benchImages resolved');
    const req = worker.posted[0]!;
    expect(req.type === 'parse' && req.ctx.imageIds).toEqual(imagesPage.images.map((i) => i.id));
    await worker.runAndDeliver();
    const bad = await until(() => document.querySelector('.f-state.bad'), 'unknown image rejected');
    expect(bad.textContent).toMatch(/has no image "unknown-img-id"/);

    // Now verify that when benchImages fails, partial validation (imageIds: null) still proceeds.
    c!.unmount();
    c = undefined;
    stopRoot?.();
    stopRoot = undefined;
    spawned = [];

    const { validJson } = await mountDialog({
      interceptBenchImages: () => Promise.reject(new Error('offline')),
    });
    selectFile(new File([validJson], 'partial.json', { type: 'application/json' }));
    const wPartial = await until(() => spawned[0], 'worker spawned with null imageIds');
    const partialReq = wPartial.posted[0]!;
    expect(partialReq.type === 'parse' && partialReq.ctx.imageIds).toBeNull();
    await wPartial.runAndDeliver();
    await until(() => document.querySelector('.f-state.ok'), 'partial validation succeeded');
  });

  it('guards route navigation while an external upload is pending, shows Stop sending?, and proceeds after Stop and discard', async () => {
    router.go('/benchmarks/bn_9f03?add=external', { replace: true });
    const { isOpen, validJson } = await mountDialog();
    selectFile(new File([validJson], 'guarded.json', { type: 'application/json' }));
    const worker = await until(() => spawned[0], 'worker spawned');
    await worker.runAndDeliver();
    await until(() => document.querySelector('.f-state.ok'), 'validated state');

    // Submit without auto-answering the worker read so the send stays active.
    const form = document.querySelector<HTMLFormElement>('dialog[open] form')!;
    form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    flushSync();

    // Navigating away (e.g. Back to the entry page) is held and displays the stop confirmation.
    router.go('/benchmarks/bn_9f03/entries/be_1');
    flushSync();
    expect(router.route.path).toBe('/benchmarks/bn_9f03');
    expect(isOpen.get()).toBe(true);
    expect(document.querySelector('dialog[open]')?.textContent).toContain('Stop sending?');

    // Confirming "Stop and discard" aborts the send, closes the dialog, and completes the held navigation.
    const stopBtn = [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => b.textContent?.trim() === 'Stop and discard')!;
    stopBtn.click();
    await until(() => !isOpen.get() && router.route.path === '/benchmarks/bn_9f03/entries/be_1', 'held navigation proceeded after stop');
    expect(worker.terminated).toBe(true);
  });
});
