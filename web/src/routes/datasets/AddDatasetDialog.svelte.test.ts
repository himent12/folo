// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import { router } from '$lib/router.svelte';
import { toast } from '$lib/ui/toast.svelte';
import AddDatasetDialog from './AddDatasetDialog.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  toast.items = [];
  target = document.createElement('div');
  document.body.appendChild(target);
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

async function register(path: string, known?: { id: string }[]) {
  const go = vi.spyOn(router, 'go').mockImplementation(() => {});
  c = mount(AddDatasetDialog, { target, props: { open: true, known } });
  flushSync();
  const input = document.querySelector<HTMLInputElement>('input[placeholder="e.g. D:/data/my-dataset"]')!;
  input.value = path;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
  input.form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
  await vi.waitFor(() => {
    flushSync();
    if (!toast.items.length && !document.querySelector('.err')) throw new Error('still waiting for an answer');
  });
  return go;
}

describe('AddDatasetDialog', () => {
  it('keeps Register disabled, saying why, until there is a folder path', () => {
    c = mount(AddDatasetDialog, { target, props: { open: true } });
    flushSync();
    const submit = document.querySelector<HTMLButtonElement>('button[type="submit"]')!;
    expect(submit.disabled).toBe(true);
    expect(document.getElementById(submit.getAttribute('aria-describedby') ?? '')?.textContent).toBe('Enter a folder path');
    const input = document.querySelector<HTMLInputElement>('input[placeholder="e.g. D:/data/my-dataset"]')!;
    input.value = 'D:/data/shapes';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(submit.disabled).toBe(false);
    expect(submit.hasAttribute('aria-describedby')).toBe(false);
  });

  it('gates Register while the name shows an error, and lifts it once the name is fixed', () => {
    c = mount(AddDatasetDialog, { target, props: { open: true } });
    flushSync();
    const type = (sel: string, v: string) => {
      const el = document.querySelector<HTMLInputElement>(sel)!;
      el.value = v;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      return el;
    };
    const path = type('input[placeholder="e.g. D:/data/my-dataset"]', 'D:/data/my dataset');
    type('input[id$="-name"]', 'my dataset!');
    path.form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    flushSync();
    const submit = document.querySelector<HTMLButtonElement>('button[type="submit"]')!;
    expect(submit.disabled).toBe(true);
    expect(document.getElementById(submit.getAttribute('aria-describedby') ?? '')?.textContent).toBe('Fix the name first');
    type('input[id$="-name"]', 'my-dataset');
    expect(submit.disabled).toBe(false);
  });

  it('says a folder is already registered, and opens it, instead of announcing a new dataset', async () => {

    const go = await register('D:\\data\\shapes\\', [{ id: 'ds_fasteners' }, { id: 'ds_shapes' }]);
    expect(toast.items[0]).toMatchObject({ tone: 'info', title: 'Already registered as shapes' });
    expect(go).toHaveBeenCalledWith('/datasets/ds_shapes');
  });

  it('reads the list itself when it was not given one', async () => {
    const go = await register('D:/data/shapes-hard');
    expect(toast.items[0]?.title).toBe('Already registered as shapes-hard');
    expect(go).toHaveBeenCalledWith('/datasets/ds_shapes_hard');
  });

  it('announces a dataset that was not in the list as new', async () => {
    const go = await register('D:/data/shapes', [{ id: 'ds_fasteners' }]);
    expect(toast.items[0]).toMatchObject({ tone: 'success', title: 'Dataset registered' });
    expect(go).toHaveBeenCalledWith('/datasets/ds_shapes');
  });
});
