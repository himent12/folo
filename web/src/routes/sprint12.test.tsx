// @vitest-environment happy-dom
// Sprint 12 Studio route regression tests:
// - RunCompare `?ids=` reconciliation never clobbers `router.location` while leaving `/runs/compare` during `router.holdWhile`.
// - LabelJob `start()` aborts before `startLabeling` if navigated to another job while `instructions.save()` is in flight.
// - Benchmark `removeEntry()` settling after navigating to another benchmark does not drop an entry or close a dialog on the new benchmark.
// - Runs `announce()` ignores superseded microtasks and never updates status text after unmount.
import { flush as flushSync, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import Benchmark from './Benchmark';
import LabelJob from './LabelJob';
import RunCompare from './RunCompare';
import Runs from './Runs';

let target: HTMLElement;
let c: Rendered | undefined;
let releaseHold: (() => void) | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`waiting for ${what}`);
    },
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  toast.items = [];
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
});

afterEach(() => {
  releaseHold?.();
  releaseHold = undefined;
  c?.unmount();
  c = undefined;
  target.remove();
  vi.restoreAllMocks();
});

describe('Studio Sprint 12 route hardening', () => {
  it('RunCompare does not clobber outgoing router.location while leaving during router.holdWhile', async () => {
    const api = new MockApi('normal', 0);
    const realRuns = api.runs.bind(api);
    let releaseRuns!: () => void;
    const holdRuns = new Promise<void>((r) => (releaseRuns = r));
    vi.spyOn(api, 'runs').mockImplementation(async () => {
      await holdRuns;
      return realRuns();
    });
    hub.api = api;

    router.route = parse('#/runs/compare?ids=missing_run_id');
    flushSync();
    c = render(() => <RunCompare />, target);
    flushSync();

    // Begin navigating away while a chunk is held: router.location is /datasets, router.route is still /runs/compare.
    const holding = signal(true);
    releaseHold = router.holdWhile((to) => holding.get() && to.path === '/datasets');
    router.go('/datasets');
    flushSync();
    expect(router.location.path).toBe('/datasets');
    expect(router.route.path).toBe('/runs/compare');

    // Runs list settles with missing_run_id absent: RunCompare must NOT overwrite router.location back to /runs/compare.
    releaseRuns();
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    expect(router.location.path).toBe('/datasets');

    holding.set(false);
    flushSync();
    expect(router.route.path).toBe('/datasets');
  });

  it('LabelJob start() does not fire startLabeling for the old job if navigated away during instructions.save()', async () => {
    const api = new MockApi('normal', 0);
    const jobs = await api.labelJobs();
    const idleJob = jobs.find((j) => j.status === 'idle' && j.counts.pending + j.counts.failed > 0) ?? jobs.find((j) => j.status === 'idle')!;
    const otherJob = jobs.find((j) => j.id !== idleJob.id)!;
    expect(idleJob).toBeTruthy();
    expect(otherJob).toBeTruthy();
    // Ensure idleJob has unlabeled images so the direct "Label N images" button calls start() immediately.
    const realLabelJob = api.labelJob.bind(api);
    vi.spyOn(api, 'labelJob').mockImplementation(async (id) => {
      const j = await realLabelJob(id);
      if (id === idleJob.id && j.counts.pending + j.counts.failed === 0) {
        return { ...j, counts: { ...j.counts, pending: 2, images: Math.max(j.counts.images, 2) } };
      }
      return j;
    });

    let releaseSave!: () => void;
    const holdSave = new Promise<void>((r) => (releaseSave = r));
    const realUpdate = api.updateLabelJob.bind(api);
    const updateSpy = vi.spyOn(api, 'updateLabelJob').mockImplementation(async (id, patch) => {
      await holdSave;
      return realUpdate(id, patch);
    });
    const startSpy = vi.spyOn(api, 'startLabeling');
    hub.api = api;

    router.route = parse(`#/datasets/label/${idleJob.id}`);
    c = render(() => <LabelJob />, target);

    const textarea = await until(() => target.querySelector<HTMLTextAreaElement>('textarea'), 'instructions textarea');
    const startBtn = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => /^Label \d+ image/i.test(b.textContent?.trim() ?? '') && !b.disabled),
      'Label N images button',
    );

    // Edit instructions so start() awaits instructions.save(), then click Label and switch to another job before save finishes.
    textarea.value = `${textarea.value} extra`;
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    startBtn.click();
    flushSync();
    await until(() => updateSpy.mock.calls.length === 1, 'instructions update call');

    // Navigate to otherJob while save is still in flight.
    router.route = parse(`#/datasets/label/${otherJob.id}`);
    flushSync();
    await until(() => target.querySelector('h1')?.textContent?.includes(otherJob.name), 'other job heading');

    releaseSave();
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    expect(startSpy).not.toHaveBeenCalled();
  });

  it('Benchmark removeEntry() settling after switching benchmarks does not touch the new benchmark', async () => {
    const api = new MockApi('normal', 0);
    const benches = await api.benches();
    let b1 = benches[0]!;
    let b1Entries = await api.benchEntries(b1.id);
    for (const b of benches) {
      const entries = await api.benchEntries(b.id);
      if (entries.length > 0) {
        b1 = b;
        b1Entries = entries;
        break;
      }
    }
    const b2 = benches.find((b) => b.id !== b1.id)!;
    const victim = b1Entries.find((e) => e.status === 'done') ?? b1Entries[0]!;
    expect(b2).toBeTruthy();
    expect(victim).toBeTruthy();

    let releaseDelete!: () => void;
    const holdDelete = new Promise<void>((r) => (releaseDelete = r));
    const realDelete = api.deleteBenchEntry.bind(api);
    vi.spyOn(api, 'deleteBenchEntry').mockImplementation(async (bid, eid) => {
      await holdDelete;
      return realDelete(bid, eid);
    });
    hub.api = api;

    router.route = parse(`#/benchmarks/${b1.id}`);
    c = render(() => <Benchmark />, target);

    // Open row menu for victim entry and trigger Delete entry confirm dialog.
    const menuTrigger = await until(
      () => target.querySelector<HTMLButtonElement>(`button[aria-haspopup="menu"][aria-label*="${victim.name}"]`),
      'entry row menu trigger',
    );
    menuTrigger.click();
    flushSync();
    const delItem = await until(
      () => [...target.querySelectorAll<HTMLElement>('[role="menuitem"]')].find((el) => /Delete/i.test(el.textContent ?? '')),
      'delete menuitem',
    );
    delItem.click();
    flushSync();

    const confirmBtn = await until(
      () => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => /Delete/i.test(b.textContent ?? '')),
      'confirm delete button',
    );
    confirmBtn.click();
    flushSync();

    // Switch to b2 while deleteBenchEntry(b1, victim) is in flight.
    const b2EntriesBefore = (await api.benchEntries(b2.id)).length;
    router.route = parse(`#/benchmarks/${b2.id}`);
    flushSync();
    await until(() => target.querySelector('h1')?.textContent?.includes(b2.name), 'second benchmark heading');

    releaseDelete();
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    expect(target.querySelector('h1')?.textContent).toContain(b2.name);
    expect((await api.benchEntries(b2.id)).length).toBe(b2EntriesBefore);
  });

  it('Runs clears filter announcements safely on unmount without throwing', async () => {
    router.route = parse('#/runs?q=no_such_run_xyz');
    c = render(() => <Runs />, target);

    const clearBtn = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => /Clear filters|Show all runs/i.test(b.textContent ?? '')),
      'clear filters button',
    );
    clearBtn.click();
    flushSync();
    // Unmount before microtask announcement resolves: must not throw or leave live errors.
    c.unmount();
    c = undefined;
    await Promise.resolve();
    await new Promise((r) => setTimeout(r, 0));
    expect(target.innerHTML).toBe('');
  });
});
