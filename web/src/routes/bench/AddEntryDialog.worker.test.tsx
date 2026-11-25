// @vitest-environment happy-dom
// Verifies AddEntryDialog offloads external predictions JSON parsing to a Web Worker and terminates it on re-pick, close, and unmount.
import { createRoot, flush as flushSync, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Bench } from '$lib/api/types';
import { runExternalParseRequest, type ExternalParseRequest, type ExternalParseResponse } from '$lib/bench/externalWorker';
import { AgentChoice } from '../label/agentChoice';
import AddEntryDialog from './AddEntryDialog';

interface RecordedWorker {
  posted: ExternalParseRequest[];
  terminated: boolean;
  deliver: (res: ExternalParseResponse) => void;
  runAndDeliver: () => Promise<void>;
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
    readonly rec: RecordedWorker;

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
          const res = await runExternalParseRequest(req);
          rec.deliver(res);
        },
      };
      this.rec = rec;
      spawned.push(rec);
    }

    postMessage(req: ExternalParseRequest) {
      this.rec.posted.push(req);
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

async function settle(rounds = 8) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
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

async function mountDialog() {
  const bench: Bench = await hub.api.bench('bn_9f03');
  const models = await hub.api.models();
  const agents = await hub.api.labelAgents();
  const entries = await hub.api.benchEntries('bn_9f03');
  const catalog = await hub.api.benchCatalog();
  const images = (await hub.api.benchImages('bn_9f03', { limit: 1000 })).images;
  const validJson = JSON.stringify(images.map((i) => ({ image: i.id, boxes: i.boxes.map((b) => ({ ...b, score: 0.8 })) })));

  let choice!: AgentChoice;
  stopRoot = createRoot((dispose) => {
    choice = new AgentChoice(() => agents);
    return dispose;
  });

  const isOpen = signal(true);
  const kind = signal<'model' | 'agent' | 'external'>('external');
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
        onstarted={() => {}}
      />
    ),
    target,
  );
  flushSync();
  await settle();
  return { isOpen, validJson };
}

describe('AddEntryDialog worker offloading', () => {
  it('parses an external file in a Worker, shows the summary, and terminates the worker on completion', async () => {
    const { validJson } = await mountDialog();
    selectFile(new File([validJson], 'worker-model-predictions.json', { type: 'application/json' }));
    await settle();

    expect(spawned).toHaveLength(1);
    expect(spawned[0]!.terminated).toBe(false);
    expect(document.querySelector('.f-state.busy')).not.toBeNull();

    await spawned[0]!.runAndDeliver();
    await settle();

    expect(spawned[0]!.terminated).toBe(true);
    expect(document.querySelector('.f-state.ok')).not.toBeNull();
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('worker-model');
  });

  it('terminates the previous worker immediately on re-upload and on dialog close', async () => {
    const { isOpen, validJson } = await mountDialog();

    selectFile(new File([validJson], 'first.json', { type: 'application/json' }));
    await settle();
    expect(spawned).toHaveLength(1);
    expect(spawned[0]!.terminated).toBe(false);

    // Picking a second file cancels and terminates worker #1 immediately.
    selectFile(new File([validJson], 'second.json', { type: 'application/json' }));
    await settle();
    expect(spawned).toHaveLength(2);
    expect(spawned[0]!.terminated).toBe(true);
    expect(spawned[1]!.terminated).toBe(false);

    // Closing the dialog terminates worker #2 immediately.
    isOpen.set(false);
    flushSync();
    expect(spawned[1]!.terminated).toBe(true);
  });

  it('terminates an in-flight worker when the dialog unmounts', async () => {
    const { validJson } = await mountDialog();

    selectFile(new File([validJson], 'unmount.json', { type: 'application/json' }));
    await settle();
    expect(spawned).toHaveLength(1);
    expect(spawned[0]!.terminated).toBe(false);

    c!.unmount();
    c = undefined;
    expect(spawned[0]!.terminated).toBe(true);
  });
});
