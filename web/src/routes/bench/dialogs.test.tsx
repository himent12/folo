// @vitest-environment happy-dom
// Bench dialogs: in-flight submit guards, onbeforeclose while busy, and lazy NewLoopDialog benchResult fetching.
import { flush as flushSync, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Bench, BenchEntry } from '$lib/api/types';
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
});
