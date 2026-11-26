// @vitest-environment happy-dom
// Sprint 12 Studio route regression tests:
// - RunCompare `?ids=` reconciliation never clobbers `router.location` while leaving `/runs/compare` during `router.holdWhile`.
// - LabelJob `start()` aborts before `startLabeling` if navigated away (including while `router.holdWhile` keeps the old view mounted).
// - Benchmark `removeEntry()` settling after switching benchmarks does not drop destination leaderboard entries or close a destination dialog.
// - Runs `announce()` ignores superseded/duplicate microtasks safely on unmount.
import { flush as flushSync, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { router } from '$lib/router';
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

const flushMicrotasks = async () => {
  for (let i = 0; i < 6; i++) await Promise.resolve();
  flushSync();
};

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
    let runsDone!: () => void;
    const runsSettled = new Promise<void>((r) => (runsDone = r));
    vi.spyOn(api, 'runs').mockImplementation(async () => {
      await holdRuns;
      const out = await realRuns();
      runsDone();
      return out;
    });
    hub.api = api;

    router.go('/runs/compare?ids=missing_run_id', { replace: true });
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
    await runsSettled;
    await flushMicrotasks();
    expect(router.location.path).toBe('/datasets');

    holding.set(false);
    flushSync();
    expect(router.route.path).toBe('/datasets');
  });

  it('LabelJob start() does not fire startLabeling if user navigated away while router.holdWhile kept the old view mounted', async () => {
    const api = new MockApi('normal', 0);
    const jobs = await api.labelJobs();
    const idleJob = jobs.find((j) => j.status === 'idle' && j.counts.pending + j.counts.failed > 0) ?? jobs.find((j) => j.status === 'idle')!;
    expect(idleJob).toBeTruthy();
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
    let saveDone!: () => void;
    const saveSettled = new Promise<void>((r) => (saveDone = r));
    const realUpdate = api.updateLabelJob.bind(api);
    const updateSpy = vi.spyOn(api, 'updateLabelJob').mockImplementation(async (id, patch) => {
      await holdSave;
      const out = await realUpdate(id, patch);
      saveDone();
      return out;
    });
    const startSpy = vi.spyOn(api, 'startLabeling');
    hub.api = api;

    router.go(`/datasets/label/${idleJob.id}`, { replace: true });
    flushSync();
    c = render(() => <LabelJob />, target);

    const textarea = await until(() => target.querySelector<HTMLTextAreaElement>('textarea'), 'instructions textarea');
    const startBtn = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => /^Label \d+ image/i.test(b.textContent?.trim() ?? '') && !b.disabled),
      'Label N images button',
    );

    textarea.value = `${textarea.value} extra`;
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    startBtn.click();
    flushSync();
    await until(() => updateSpy.mock.calls.length === 1, 'instructions update call');

    // Navigate to /models while router.holdWhile keeps LabelJob mounted and displaying idleJob.
    const holding = signal(true);
    releaseHold = router.holdWhile((to) => holding.get() && to.path === '/models');
    router.go('/models');
    flushSync();
    expect(router.location.path).toBe('/models');
    expect(router.route.path).toBe(`/datasets/label/${idleJob.id}`);

    // Save finishes while old view is still held: start() must see router.location moved and NOT dispatch startLabeling.
    releaseSave();
    await saveSettled;
    await flushMicrotasks();
    expect(startSpy).not.toHaveBeenCalled();

    holding.set(false);
    flushSync();
  });

  it('Benchmark removeEntry() settling after switching benchmarks does not drop destination leaderboard rows or close its dialog', async () => {
    const api = new MockApi('normal', 0);
    const benches = await api.benches();
    const withEntries = [];
    for (const b of benches) {
      const entries = await api.benchEntries(b.id);
      if (entries.length > 0) withEntries.push({ bench: b, entries });
    }
    expect(withEntries.length).toBeGreaterThanOrEqual(2);
    const b1 = withEntries[0]!.bench;
    const b2 = withEntries[1]!.bench;
    const victim = withEntries[0]!.entries[0]!;
    const b2Target = withEntries[1]!.entries[0]!;
    // Give b2's entry the same entry id as b1's victim to prove LiveBench.dropEntry + removing dialog guards block cross-benchmark mutation.
    const realBenchEntries = api.benchEntries.bind(api);
    vi.spyOn(api, 'benchEntries').mockImplementation(async (bid) => {
      const list = await realBenchEntries(bid);
      if (bid === b2.id) return list.map((e, i) => (i === 0 ? { ...e, id: victim.id } : e));
      return list;
    });

    let releaseDelete!: () => void;
    const holdDelete = new Promise<void>((r) => (releaseDelete = r));
    let deleteDone!: () => void;
    const deleteSettled = new Promise<void>((r) => (deleteDone = r));
    const realDelete = api.deleteBenchEntry.bind(api);
    vi.spyOn(api, 'deleteBenchEntry').mockImplementation(async (bid, eid) => {
      await holdDelete;
      const out = await realDelete(bid, eid);
      deleteDone();
      return out;
    });
    hub.api = api;

    router.go(`/benchmarks/${b1.id}`, { replace: true });
    flushSync();
    c = render(() => <Benchmark />, target);

    const openDeleteDialog = async (entryName: string) => {
      const trigger = await until(
        () => target.querySelector<HTMLButtonElement>(`button[aria-haspopup="menu"][aria-label*="${entryName}"]`),
        `menu trigger for ${entryName}`,
      );
      trigger.click();
      flushSync();
      const delItem = await until(
        () => [...target.querySelectorAll<HTMLElement>('[role="menuitem"]')].find((el) => /Delete/i.test(el.textContent ?? '')),
        'delete menuitem',
      );
      delItem.click();
      flushSync();
      return until(
        () => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => /Delete/i.test(b.textContent ?? '')),
        'confirm delete button',
      );
    };

    const confirmB1 = await openDeleteDialog(victim.name);
    confirmB1.click();
    flushSync();

    // Switch to b2 and open b2's delete confirmation dialog for the entry sharing `victim.id`.
    router.go(`/benchmarks/${b2.id}`, { replace: true });
    flushSync();
    await until(() => target.querySelector('h1')?.textContent?.includes(b2.name) && target.textContent?.includes(b2Target.name), 'b2 leaderboard');
    await openDeleteDialog(b2Target.name);
    expect(document.querySelector('dialog[open]')).not.toBeNull();

    // Old b1 delete settles: must NOT drop b2Target from b2's rendered leaderboard or close b2's open dialog.
    releaseDelete();
    await deleteSettled;
    await flushMicrotasks();
    expect(document.querySelector('dialog[open]')).not.toBeNull();
    expect(target.textContent).toContain(b2Target.name);
  });

  it('Runs ignores queued duplicate announce() microtasks after unmount', async () => {
    router.go('/runs?q=no_such_run_xyz', { replace: true });
    flushSync();
    c = render(() => <Runs />, target);

    const clearBtn = () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => /Clear filters|Show all runs/i.test(b.textContent ?? ''));
    // First clear sets liveNote = 'Cleared run filters.' synchronously.
    (await until(clearBtn, 'first clear filters button')).click();
    flushSync();
    // Re-apply an empty filter and click Clear filters a second time to hit the duplicate-message microtask branch of announce().
    router.go('/runs?q=still_no_such_run_xyz', { replace: true });
    flushSync();
    (await until(clearBtn, 'second clear filters button')).click();
    flushSync();

    c.unmount();
    c = undefined;
    await flushMicrotasks();
    expect(target.innerHTML).toBe('');
  });
});
