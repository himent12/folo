// @vitest-environment happy-dom
// A registered dataset whose folder is gone (moved, renamed, deleted): the list marks it, and its page keeps its
// name, says what is wrong and can remove it from FOLO.
import { flush as flushSync, type Component } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Dataset from './Dataset';
import Datasets from './Datasets';

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
    await vi.waitFor(() => expect(go).toHaveBeenCalledWith('/datasets'));
    expect(del).toHaveBeenCalledWith('ds_shapes');
  });
});
