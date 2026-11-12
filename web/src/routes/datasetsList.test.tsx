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
import { datasetsChanged } from './datasets/changes';

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
    const row = await until(() => target.querySelector('tr[aria-label="shapes"]'), 'the shapes row');
    expect(row.textContent).toContain('Folder missing');
    expect(target.querySelector('tr[aria-label="fasteners"]')!.textContent).not.toContain('Folder missing');
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
    await until(() => target.querySelector('tr[aria-label="shapes"]')?.textContent?.includes('Folder missing'), 'the mark');
    expect(asked()).toBe(1);
    // Its page, then the list again: both say so from what the list found.
    c!.unmount();
    open(Dataset, '/datasets/ds_shapes');
    await until(() => target.textContent?.includes('Put it back and press Retry'), 'the page');
    c!.unmount();
    open(Datasets, '/datasets');
    await until(() => target.querySelector('tr[aria-label="shapes"]')?.textContent?.includes('Folder missing'), 'the mark again');
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
    await until(() => target.querySelector('tr[aria-label="fasteners"]'), 'the rows');
    const calls = list.mock.calls.length;
    datasetsChanged();
    await vi.waitFor(() => expect(list.mock.calls.length).toBe(calls + 1));
    document.dispatchEvent(new Event('visibilitychange'));
    await vi.waitFor(() => expect(list.mock.calls.length).toBe(calls + 2));
  });
});

describe('Add dataset from the URL', () => {
  it('opens on ?add= followed while the list is open, and closing it takes ?add= out of the URL (v-shell N8)', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation((path: string) => {
      router.route = parse(`#${path}`);
    });
    open(Datasets, '/datasets');
    await until(() => target.querySelector('tr[aria-label="fasteners"]'), 'the rows');
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
