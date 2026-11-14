// @vitest-environment happy-dom
// The Datasets list's view of the datasets there are: one whose folder is gone (moved, renamed, deleted) is marked and
// its page keeps its name and can remove it from FOLO; and the list reads them again when they may have changed.
import { flush as flushSync, type Component } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Dataset from './Dataset';
import Datasets from './Datasets';
import { datasetsChanged, generationEnded, generationStarted, pendingGeneration } from './datasets/changes';

let target: HTMLElement;
let c: { unmount(): void } | undefined;

beforeEach(() => {
  const api = new MockApi('normal', 0);
  const read = api.dataset.bind(api);
  vi.spyOn(api, 'dataset').mockImplementation((id: string) =>
    id === 'ds_shapes' ? Promise.reject(new ApiError(404, 'path_not_found', "the folder of dataset 'shapes' is gone: 'D:/data/shapes' does not exist")) : read(id),
  );
  hub.api = api;
  hub.mode = 'mock';
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
});

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  vi.restoreAllMocks();
});

function open(page: Component<Record<string, never>>, path: string) {
  router.route = parse(`#${path}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  const P = page;
  c = render(() => <P />, target);
  flushSync();
}

const until = <T,>(f: () => T | null | undefined, what: string) =>
  vi.waitFor(() => {
    flushSync();
    const v = f();
    if (!v) throw new Error(`still waiting for ${what}`);
    return v;
  });

describe('a dataset whose folder is missing', () => {
  it('is marked in the list, not shown as if nothing were wrong', async () => {
    open(Datasets, '/datasets');
    const row = await until(() => target.querySelector('.card[data-name="shapes"]'), 'the shapes row');
    expect(row.textContent).toContain('Folder missing');
    expect(target.querySelector('.card[data-name="fasteners"]')!.textContent).not.toContain('Folder missing');
  });

  it('keeps its name on its page, says what is wrong, and can be removed from FOLO', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const del = vi.spyOn(hub.api, 'deleteDataset');
    open(Dataset, '/datasets/ds_shapes');
    await until(() => target.querySelector('h1')?.textContent === 'shapes', 'the name');
    expect(target.textContent).toContain('Folder missing');
    expect(target.textContent).toContain('Put it back and press Retry');
    const remove = [...target.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Remove from FOLO')!;
    remove.click();
    flushSync();
    const confirm = await until(
      () => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => b.textContent?.trim() === 'Remove from FOLO'),
      'the confirmation',
    );
    confirm.click();
    await vi.waitFor(() => expect(go).toHaveBeenCalledWith('/datasets', { replace: true }));
    expect(del).toHaveBeenCalledWith('ds_shapes', { files: false });
  });

  it('is not asked about again on every visit, only on Retry and when the tab comes back (v-data-runs N6)', async () => {
    const read = vi.mocked(hub.api.dataset);
    const issues = vi.spyOn(hub.api, 'datasetIssues');
    const asked = () => read.mock.calls.filter(([id]) => id === 'ds_shapes').length;
    open(Datasets, '/datasets');
    await until(() => target.querySelector('.card[data-name="shapes"]')?.textContent?.includes('Folder missing'), 'the mark');
    expect(asked()).toBe(1);
    // Its page, then the list again: both say so from what the list found.
    c!.unmount();
    open(Dataset, '/datasets/ds_shapes');
    await until(() => target.textContent?.includes('Put it back and press Retry'), 'the page');
    c!.unmount();
    open(Datasets, '/datasets');
    await until(() => target.querySelector('.card[data-name="shapes"]')?.textContent?.includes('Folder missing'), 'the mark again');
    expect(asked()).toBe(1);
    expect(issues.mock.calls.filter(([id]) => id === 'ds_shapes')).toHaveLength(0);
    // Back in the tab: the folder may be back.
    document.dispatchEvent(new Event('visibilitychange'));
    await vi.waitFor(() => expect(asked()).toBe(2));
    // Retry on its page asks.
    c!.unmount();
    open(Dataset, '/datasets/ds_shapes');
    const retry = await until(() => [...target.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Retry'), 'Retry');
    expect(asked()).toBe(2);
    retry.click();
    await vi.waitFor(() => expect(asked()).toBe(3));
  });
});

describe('the Datasets list', () => {
  it('reads the datasets again after a dataset action in this tab and when the tab comes back', async () => {
    const list = vi.spyOn(hub.api, 'datasets');
    open(Datasets, '/datasets');
    await until(() => target.querySelector('.card[data-name="fasteners"]'), 'the rows');
    const calls = list.mock.calls.length;
    datasetsChanged();
    await vi.waitFor(() => expect(list.mock.calls.length).toBe(calls + 1));
    document.dispatchEvent(new Event('visibilitychange'));
    await vi.waitFor(() => expect(list.mock.calls.length).toBe(calls + 2));
  });
});

describe('Add dataset from the URL', () => {
  it('offers a folder of labeled images first when there are no datasets yet (studio-data F4)', async () => {
    vi.spyOn(hub.api, 'datasets').mockResolvedValue([]);
    open(Datasets, '/datasets');
    const empty = await until(() => target.querySelector('.ui-empty, [class*="empty"]')?.closest('section') ?? null, 'the empty state');
    expect(empty.textContent).toContain('Register a folder of images you have labeled (YOLO or COCO)');
    [...empty.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Register folder')!.click();
    const dialog = await until(() => document.querySelector<HTMLDialogElement>('dialog[open]'), 'the dialog');
    expect(dialog.querySelector('input[placeholder="e.g. D:/data/my-dataset"]')).not.toBeNull();
  });

  it('opens on ?add= followed while the list is open, and closing it takes ?add= out of the URL (v-shell N8)', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation((path: string) => {
      router.route = parse(`#${path}`);
    });
    open(Datasets, '/datasets');
    await until(() => target.querySelector('.card[data-name="fasteners"]'), 'the rows');
    expect(document.querySelector('dialog[open]')).toBeNull();
    // The palette, Overview's onboarding or New run's empty state links here while the list is on display.
    router.route = parse('#/datasets?add=synthetic');
    const dialog = await until(() => document.querySelector<HTMLDialogElement>('dialog[open]'), 'the dialog');
    expect(dialog.textContent).toContain('Generate');
    [...dialog.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Cancel')!.click();
    await until(() => go.mock.calls.length > 0, 'the URL change');
    expect(go).toHaveBeenLastCalledWith('/datasets', { replace: true });
    flushSync();
    expect(document.querySelector('dialog[open]')).toBeNull();
  });
});

describe('a sample set generating across a reload', () => {
  it('has the list read itself again until the new dataset is there, then stops (datasets F10)', async () => {
    const before = await hub.api.datasets();
    const g = { server: hub.server, name: 'shapes-synth', since: Date.now(), ids: before.map((d) => d.id) };
    generationStarted(g);
    const list = vi.spyOn(hub.api, 'datasets');
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
    try {
      open(Datasets, '/datasets');
      await until(() => target.querySelector('.card[data-name="fasteners"]'), 'the rows');
      const calls = list.mock.calls.length;
      await vi.advanceTimersByTimeAsync(3000);
      await vi.waitFor(() => expect(list.mock.calls.length).toBe(calls + 1));
      // It lands (numbered, as the server names a taken name).
      await hub.api.createSyntheticDataset({ name: 'shapes-synth', images: 10, classes: ['circle'], imgsz: 64, seed: 0 });
      const shown = () => [...target.querySelectorAll('.card[data-name]')].some((r) => !before.some((d) => d.name === r.getAttribute('data-name')));
      for (let i = 0; i < 40 && !shown(); i++) {
        await vi.advanceTimersByTimeAsync(100);
        flushSync();
      }
      expect(shown()).toBe(true);
      expect(pendingGeneration(hub.server)).toBeNull();
      const after = list.mock.calls.length;
      await vi.advanceTimersByTimeAsync(9000);
      expect(list.mock.calls.length).toBe(after);
    } finally {
      vi.useRealTimers();
      generationEnded(g);
    }
  });

  it('says a folder still being read before a reload is on its way, until a dataset of its path is listed (studio-data F1)', async () => {
    const api = hub.api;
    const before = await api.datasets();
    // As typed (back slashes, a trailing one); the server lists it with forward slashes.
    const g = { server: hub.server, name: 'coco', since: Date.now(), ids: before.map((d) => d.id), path: 'D:\\data\\coco\\', what: 'Reading coco' };
    generationStarted(g);
    let there = false;
    const read = api.datasets.bind(api);
    vi.spyOn(api, 'datasets').mockImplementation(async (s?: AbortSignal) => {
      const list = await read(s);
      return there ? [...list, { ...list[0]!, id: 'ds_c0c0', name: 'COCO 2017', path: 'D:/data/coco' }] : list;
    });
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
    try {
      open(Datasets, '/datasets');
      await until(() => target.querySelector('.card[data-name="fasteners"]'), 'the rows');
      const note = await until(() => target.querySelector('.waiting'), 'the note');
      expect(note.textContent).toBe('Reading coco, started just now: it shows up here when it is done');
      there = true;
      for (let i = 0; i < 40 && !target.querySelector('.card[data-name="COCO 2017"]'); i++) {
        await vi.advanceTimersByTimeAsync(250);
        flushSync();
      }
      expect(target.querySelector('.card[data-name="COCO 2017"]')).not.toBeNull();
      expect(target.querySelector('.waiting')).toBeNull();
      expect(pendingGeneration(hub.server)).toBeNull();
    } finally {
      vi.useRealTimers();
      generationEnded(g);
    }
  });

  // review studio-2 R3: a read slower than the poll's interval is never cut short by the next one.
  it('waits for each read to settle before the next, and lands a set behind reads slower than the interval', async () => {
    const api = hub.api;
    const before = await api.datasets();
    const g = { server: hub.server, name: 'slow-synth', since: Date.now(), ids: before.map((d) => d.id) };
    generationStarted(g);
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
    let slow = false;
    let aborted = 0;
    const read = api.datasets.bind(api);
    const list = vi.spyOn(api, 'datasets').mockImplementation((s?: AbortSignal) =>
      !slow
        ? read(s)
        : new Promise((ok, no) => {
            const t = setTimeout(() => void read().then(ok, no), 4000);
            s?.addEventListener('abort', () => {
              aborted++;
              clearTimeout(t);
              no(new DOMException('aborted', 'AbortError'));
            });
          }),
    );
    try {
      open(Datasets, '/datasets');
      await until(() => target.querySelector('.card[data-name="fasteners"]'), 'the rows');
      slow = true;
      const calls = list.mock.calls.length;
      await read(undefined).then(() => api.createSyntheticDataset({ name: 'slow-synth', images: 10, classes: ['circle'], imgsz: 64, seed: 0 }));
      // 3 s to the first poll, 4 s for its read; the next goes out 3 s after that, not while it is out.
      await vi.advanceTimersByTimeAsync(3000 + 3900);
      expect([list.mock.calls.length - calls, aborted]).toEqual([1, 0]);
      const shown = () => !!target.querySelector('.card[data-name^="slow-synth"]');
      for (let i = 0; i < 40 && !shown(); i++) {
        await vi.advanceTimersByTimeAsync(250);
        flushSync();
      }
      expect([shown(), aborted, pendingGeneration(hub.server)]).toEqual([true, 0, null]);
    } finally {
      vi.useRealTimers();
      generationEnded(g);
    }
  });

  it('is not read while the tab is hidden', async () => {
    const before = await hub.api.datasets();
    const g = { server: hub.server, name: 'hidden-synth', since: Date.now(), ids: before.map((d) => d.id) };
    generationStarted(g);
    const list = vi.spyOn(hub.api, 'datasets');
    const shown = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden');
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
    try {
      open(Datasets, '/datasets');
      await until(() => target.querySelector('.card[data-name="fasteners"]'), 'the rows');
      const calls = list.mock.calls.length;
      await vi.advanceTimersByTimeAsync(9000);
      expect(list.mock.calls.length).toBe(calls);
      shown.mockReturnValue('visible');
      await vi.advanceTimersByTimeAsync(3000);
      expect(list.mock.calls.length).toBe(calls + 1);
    } finally {
      vi.useRealTimers();
      generationEnded(g);
    }
  });

  // review studio-2 R6: the marker belongs to the server it was asked of.
  it("is only waited for, and only forgotten, on the server it was asked of", async () => {
    const before = await hub.api.datasets();
    // Asked of another server; this one (the mock) already has a set by that name that is new to it.
    const g = { server: 'http://127.0.0.1:7915', name: 'shapes-synth', since: Date.now(), ids: before.map((d) => d.id) };
    generationStarted(g);
    await hub.api.createSyntheticDataset({ name: 'shapes-synth', images: 10, classes: ['circle'], imgsz: 64, seed: 0 });
    const list = vi.spyOn(hub.api, 'datasets');
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
    try {
      open(Datasets, '/datasets');
      await until(() => target.querySelector('.card[data-name="fasteners"]'), 'the rows');
      const calls = list.mock.calls.length;
      await vi.advanceTimersByTimeAsync(9000);
      expect(list.mock.calls.length).toBe(calls);
      expect(pendingGeneration(hub.server)).toBeNull();
      expect(pendingGeneration(g.server)).toEqual(g);
      // Over on that server, a newer generation asked for since is not forgotten by the older one's end.
      const newer = { ...g, since: g.since + 1 };
      generationStarted(newer);
      generationEnded(g);
      expect(pendingGeneration(g.server)).toEqual(newer);
      generationEnded(newer);
      expect(pendingGeneration(g.server)).toBeNull();
    } finally {
      vi.useRealTimers();
      generationEnded(g);
    }
  });
});
