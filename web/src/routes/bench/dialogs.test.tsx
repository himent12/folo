// @vitest-environment happy-dom
// Bench dialogs: in-flight submit guards, onbeforeclose while busy, and lazy NewLoopDialog benchResult fetching.
import { flush as flushSync, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Bench, BenchEntry, BenchLoop, BenchResult } from '$lib/api/types';
import { router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import { render, type Rendered } from '../../../framework/testing';
import EditBenchDialog from './EditBenchDialog';
import NewBenchDialog from './NewBenchDialog';
import NewLoopDialog from './NewLoopDialog';

let target: HTMLElement;
let c: Rendered | undefined;

beforeEach(() => {
  hub.mode = 'mock';
  toast.items = [];
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  c?.unmount();
  c = undefined;
  target.remove();
  vi.restoreAllMocks();
});

const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, ' ').trim() ?? '';
const button = (name: string) => [...target.ownerDocument.querySelectorAll<HTMLButtonElement>('dialog button')].find((b) => text(b) === name);

describe('Bench dialogs async & busy guards', () => {
  it('NewBenchDialog blocks Cancel/Escape while submitting and does not hijack navigation after unmount', async () => {
    const api = new MockApi('normal', 0);
    let resolveCreate!: (b: Bench) => void;
    vi.spyOn(api, 'createBench').mockImplementation(
      () =>
        new Promise<Bench>((res) => {
          resolveCreate = res;
        }),
    );
    hub.api = api;
    const goSpy = vi.spyOn(router, 'go').mockImplementation(() => {});
    const open = signal(true);
    c = render(() => <NewBenchDialog bind:open={open} />, target);
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('dialog select option')) throw new Error('waiting for datasets');
    });
    const nameInput = target.querySelector<HTMLInputElement>('dialog input')!;
    nameInput.value = 'Road signs';
    nameInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();

    target.querySelector<HTMLFormElement>('dialog form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    flushSync();
    expect(button('Cancel')!.disabled).toBe(true);

    // Escape cancel event is prevented while busy.
    const cancelEv = new Event('cancel', { cancelable: true });
    target.querySelector('dialog')!.dispatchEvent(cancelEv);
    expect(cancelEv.defaultPrevented).toBe(true);

    // Unmount before resolution: must not call router.go when the request settles.
    c.unmount();
    c = undefined;
    resolveCreate({ ...(await api.benches())[0]!, id: 'bn_new', name: 'Road signs', images: 10 });
    await new Promise((r) => setTimeout(r, 10));
    expect(goSpy).not.toHaveBeenCalled();
  });

  it('NewBenchDialog does not navigate away if closed and reopened on the same page before a stale submit settles', async () => {
    const api = new MockApi('normal', 0);
    let resolveCreate!: (b: Bench) => void;
    vi.spyOn(api, 'createBench').mockImplementation(
      () =>
        new Promise<Bench>((res) => {
          resolveCreate = res;
        }),
    );
    hub.api = api;
    const goSpy = vi.spyOn(router, 'go').mockImplementation(() => {});
    const open = signal(true);
    c = render(() => <NewBenchDialog bind:open={open} />, target);
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('dialog select option')) throw new Error('waiting for datasets');
    });
    const nameInput = target.querySelector<HTMLInputElement>('dialog input')!;
    nameInput.value = 'Draft one';
    nameInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();

    target.querySelector<HTMLFormElement>('dialog form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    flushSync();

    // Caller closes and reopens a fresh draft on the same route while the first request is in flight.
    open.set(false);
    flushSync();
    open.set(true);
    flushSync();

    resolveCreate({ ...(await api.benches())[0]!, id: 'bn_old', name: 'Draft one', images: 10 });
    await new Promise((r) => setTimeout(r, 10));
    flushSync();
    expect(open.get()).toBe(true);
    expect(goSpy).not.toHaveBeenCalled();
  });

  it('EditBenchDialog blocks dismissal while busy and ignores late rejection after unmount', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    const bench = (await api.benches())[0]!;
    let rejectSave!: (err: unknown) => void;
    const onsave = vi.fn(
      () =>
        new Promise<void>((_res, rej) => {
          rejectSave = rej;
        }),
    );
    const open = signal(true);
    c = render(() => <EditBenchDialog bind:open={open} bench={bench} onsave={onsave} />, target);
    flushSync();

    const nameInput = target.querySelector<HTMLInputElement>('dialog input')!;
    nameInput.value = `${bench.name} v2`;
    nameInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();

    target.querySelector<HTMLFormElement>('dialog form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    flushSync();
    expect(button('Cancel')!.disabled).toBe(true);
    const cancelEv = new Event('cancel', { cancelable: true });
    target.querySelector('dialog')!.dispatchEvent(cancelEv);
    expect(cancelEv.defaultPrevented).toBe(true);

    c.unmount();
    c = undefined;
    rejectSave(new Error('late error'));
    await new Promise((r) => setTimeout(r, 10));
    expect(open.get()).toBe(true);
  });

  it('NewLoopDialog does not fetch benchResult while closed and fetches once opened', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    const benches = await api.benches();
    let bench = benches[0]!;
    let entries: BenchEntry[] = [];
    let teacher: string | null = null;
    for (const b of benches) {
      const es = await api.benchEntries(b.id);
      const hit = es.find((e) => e.contestant.kind === 'model' && e.status === 'done' && e.fingerprint === b.fingerprint);
      if (hit && hit.contestant.kind === 'model') {
        bench = b;
        entries = es;
        teacher = hit.contestant.model_id;
        break;
      }
    }
    expect(teacher).not.toBeNull();
    const models = await api.models();
    const datasets = await api.datasets();
    const jobs = await api.labelJobs();
    const resultSpy = vi.spyOn(api, 'benchResult');

    const open = signal(false);
    c = render(
      () => <NewLoopDialog bind:open={open} bench={bench} models={models} datasets={datasets} jobs={jobs} entries={entries} teacher={teacher} onstarted={() => {}} />,
      target,
    );
    flushSync();
    await new Promise((r) => setTimeout(r, 20));
    expect(resultSpy).not.toHaveBeenCalled();

    open.set(true);
    flushSync();
    await vi.waitFor(() => {
      expect(resultSpy).toHaveBeenCalledTimes(1);
    });
  });

  it('NewLoopDialog holds Start while reading a complete teacher entry threshold, allows explicit confidence, and blocks dismissal while busy', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    const models = await api.models();
    const datasets = await api.datasets();
    const jobs = await api.labelJobs();
    const modelIds = new Set(models.map((m) => m.id));
    const benches = await api.benches();
    let bench = benches[0]!;
    let entries: BenchEntry[] = [];
    let teacher: string | null = null;
    let entryId = '';
    for (const b of benches) {
      const es = await api.benchEntries(b.id);
      const hit = es.find(
        (e) => e.contestant.kind === 'model' && modelIds.has(e.contestant.model_id) && e.status === 'done' && !e.score?.partial && e.fingerprint === b.fingerprint,
      );
      if (hit && hit.contestant.kind === 'model') {
        bench = b;
        entries = es;
        teacher = hit.contestant.model_id;
        entryId = hit.id;
        break;
      }
    }
    expect(teacher).not.toBeNull();
    const realRes = await api.benchResult(bench.id, entryId);
    let resolveResult!: (r: BenchResult) => void;
    vi.spyOn(api, 'benchResult').mockImplementation(
      () =>
        new Promise<BenchResult>((res) => {
          resolveResult = res;
        }),
    );
    let resolveStart!: (l: BenchLoop) => void;
    const startSpy = vi.spyOn(api, 'startBenchLoop').mockImplementation(
      () =>
        new Promise<BenchLoop>((res) => {
          resolveStart = res;
        }),
    );

    const open = signal(true);
    const onstarted = vi.fn();
    c = render(
      () => <NewLoopDialog bind:open={open} bench={bench} models={models} datasets={datasets} jobs={jobs} entries={entries} teacher={teacher} onstarted={onstarted} />,
      target,
    );
    flushSync();

    // While benchResult is still pending and ownThreshold is on, Start must stay disabled so a threshold-less teacher cannot submit.
    const startBtn = () => [...target.ownerDocument.querySelectorAll<HTMLButtonElement>('dialog button')].find((b) => text(b).startsWith('Start'))!;
    expect(startBtn().disabled).toBe(true);
    expect(target.ownerDocument.querySelector('dialog [data-gate-why]')?.textContent).toContain('Reading');

    // Resolve with no threshold found: Start stays disabled until an explicit confidence is provided.
    resolveResult({ ...realRes, self_labeling: { ...realRes.self_labeling, threshold: null } });
    await vi.waitFor(() => {
      flushSync();
      if (!target.ownerDocument.querySelector('dialog input[placeholder="e.g. 0.30"]')) throw new Error('waiting for conf input');
    });
    expect(startBtn().disabled).toBe(true);

    // Supply explicit confidence and submit: Cancel and Escape must be blocked while busy.
    const confInput = target.ownerDocument.querySelector<HTMLInputElement>('dialog input[placeholder="e.g. 0.30"]')!;
    confInput.value = '0.35';
    confInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(startBtn().disabled).toBe(false);

    target.querySelector<HTMLFormElement>('dialog form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    flushSync();
    expect(startSpy).toHaveBeenCalledTimes(1);
    expect(button('Cancel')!.disabled).toBe(true);
    const cancelEv = new Event('cancel', { cancelable: true });
    target.querySelector('dialog')!.dispatchEvent(cancelEv);
    expect(cancelEv.defaultPrevented).toBe(true);

    // Unmount while in flight: settling must not invoke onstarted.
    c.unmount();
    c = undefined;
    resolveStart({} as BenchLoop);
    await new Promise((r) => setTimeout(r, 10));
    expect(onstarted).not.toHaveBeenCalled();
  });

  it('keeps Start disabled and offers Retry or explicit confidence when reading the teacher entry fails', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    const benches = await api.benches();
    const models = await api.models();
    const datasets = await api.datasets();
    const jobs = await api.labelJobs();
    const modelIds = new Set(models.map((m) => m.id));
    let bench = benches[0]!;
    let entries = await api.benchEntries(bench.id);
    let teacher: string | null = null;
    let entryId = '';
    for (const b of benches) {
      const es = await api.benchEntries(b.id);
      const hit = es.find(
        (e) => e.contestant.kind === 'model' && modelIds.has(e.contestant.model_id) && e.status === 'done' && !e.score?.partial && e.fingerprint === b.fingerprint,
      );
      if (hit && hit.contestant.kind === 'model') {
        bench = b;
        entries = es;
        teacher = hit.contestant.model_id;
        entryId = hit.id;
        break;
      }
    }
    expect(teacher).not.toBeNull();
    const realRes = await api.benchResult(bench.id, entryId);
    const resSpy = vi
      .spyOn(api, 'benchResult')
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce({ ...realRes, self_labeling: { ...realRes.self_labeling, threshold: 0.4 } });

    const open = signal(true);
    c = render(
      () => <NewLoopDialog bind:open={open} bench={bench} models={models} datasets={datasets} jobs={jobs} entries={entries} teacher={teacher} onstarted={vi.fn()} />,
      target,
    );
    flushSync();

    const startBtn = () => [...target.ownerDocument.querySelectorAll<HTMLButtonElement>('dialog button')].find((b) => text(b).startsWith('Start'))!;
    await vi.waitFor(() => {
      flushSync();
      if (!button('Retry')) throw new Error('waiting for Retry button');
    });
    // Failed read must not claim it is still reading and must keep Start disabled.
    expect(target.ownerDocument.querySelector('dialog')?.textContent).not.toContain('reading its entry…');
    expect(startBtn().disabled).toBe(true);
    expect(target.ownerDocument.querySelector('dialog [data-gate-why]')?.textContent).toContain('Could not read');

    // Turning off own threshold and supplying an explicit confidence unblocks Start even without a retry.
    const sw = target.ownerDocument.querySelector<HTMLInputElement>('dialog .conf input[type="checkbox"]')!;
    sw.click();
    flushSync();
    const confInput = target.ownerDocument.querySelector<HTMLInputElement>('dialog input[placeholder="e.g. 0.30"]')!;
    confInput.value = '0.35';
    confInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(startBtn().disabled).toBe(false);

    // Turning own threshold back on re-shows Retry; clicking Retry re-fetches and unblocks Start once threshold resolves.
    sw.click();
    flushSync();
    expect(startBtn().disabled).toBe(true);
    button('Retry')!.click();
    await vi.waitFor(() => {
      flushSync();
      if (startBtn().disabled) throw new Error('waiting for Start after retry');
    });
    expect(resSpy).toHaveBeenCalledTimes(2);
  });

  it('moves keyboard focus to the confidence input when Retry resolves with a null threshold', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    const benches = await api.benches();
    const models = await api.models();
    const datasets = await api.datasets();
    const jobs = await api.labelJobs();
    const modelIds = new Set(models.map((m) => m.id));
    let bench = benches[0]!;
    let entries = await api.benchEntries(bench.id);
    let teacher: string | null = null;
    let entryId = '';
    for (const b of benches) {
      const es = await api.benchEntries(b.id);
      const hit = es.find(
        (e) => e.contestant.kind === 'model' && modelIds.has(e.contestant.model_id) && e.status === 'done' && !e.score?.partial && e.fingerprint === b.fingerprint,
      );
      if (hit && hit.contestant.kind === 'model') {
        bench = b;
        entries = es;
        teacher = hit.contestant.model_id;
        entryId = hit.id;
        break;
      }
    }
    expect(teacher).not.toBeNull();
    const realRes = await api.benchResult(bench.id, entryId);
    let resolveRetry!: (r: BenchResult) => void;
    vi.spyOn(api, 'benchResult')
      .mockRejectedValueOnce(new Error('offline'))
      .mockImplementationOnce(
        () =>
          new Promise<BenchResult>((res) => {
            resolveRetry = res;
          }),
      );

    const open = signal(true);
    c = render(
      () => <NewLoopDialog bind:open={open} bench={bench} models={models} datasets={datasets} jobs={jobs} entries={entries} teacher={teacher} onstarted={vi.fn()} />,
      target,
    );
    flushSync();

    const retry = await vi.waitFor(() => {
      flushSync();
      const b = button('Retry');
      if (!b) throw new Error('waiting for Retry');
      return b;
    });
    retry.focus();
    retry.click();
    await vi.waitFor(() => {
      flushSync();
      expect(target.ownerDocument.activeElement).toBe(target.ownerDocument.querySelector('dialog .conf input[type="checkbox"]'));
    });

    // Retry resolves with threshold: null -> switch becomes disabled and focus must move to the required confidence input.
    resolveRetry({ ...realRes, self_labeling: { ...realRes.self_labeling, threshold: null } });
    await vi.waitFor(() => {
      flushSync();
      const confInput = target.ownerDocument.querySelector<HTMLInputElement>('dialog input[placeholder="e.g. 0.30"]');
      if (!confInput || target.ownerDocument.activeElement !== confInput) throw new Error('waiting for focus on confidence input');
    });
    expect(target.ownerDocument.querySelector<HTMLInputElement>('dialog .conf input[type="checkbox"]')!.disabled).toBe(true);
  });
});
