// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import { undescribedFields } from '$lib/ui/fieldA11y';
import AddDatasetDialog from './AddDatasetDialog';

/** Port of `mount(C, { target, props })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = <P extends object>(C: (p: P) => JSX.Element, o: { target: HTMLElement; props?: P }) => render(() => C(o.props ?? ({} as P)), o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

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
  it('describes every control by the hint or error under it, in both modes', () => {
    c = mount(AddDatasetDialog, { target, props: { open: true } });
    flushSync();
    expect(document.querySelectorAll('.field').length).toBeGreaterThan(2);
    expect(undescribedFields(document)).toEqual([]);
    [...document.querySelectorAll<HTMLButtonElement>('[role="tab"]')].at(-1)!.click();
    flushSync();
    expect(undescribedFields(document)).toEqual([]);
  });

  it('keeps Register disabled, saying why, until there is a folder path and while the name shows an error', () => {
    c = mount(AddDatasetDialog, { target, props: { open: true } });
    flushSync();
    const type = (sel: string, v: string) => {
      const el = document.querySelector<HTMLInputElement>(sel)!;
      el.value = v;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      return el;
    };
    const submit = document.querySelector<HTMLButtonElement>('button[type="submit"]')!;
    const why = () => document.getElementById(submit.getAttribute('aria-describedby') ?? '')?.textContent;
    expect(submit.disabled).toBe(true);
    expect(why()).toBeTruthy();
    const path = type('input[placeholder="e.g. D:/data/my-dataset"]', 'D:/data/my dataset');
    expect(submit.disabled).toBe(false);
    expect(submit.hasAttribute('aria-describedby')).toBe(false);
    type('input[id$="-name"]', 'my dataset!');
    path.form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    flushSync();
    expect(submit.disabled).toBe(true);
    expect(why()).toBeTruthy();
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

  it('takes a path pasted in quotes (Windows "Copy as path"), and names it without them', async () => {
    c = mount(AddDatasetDialog, { target, props: { open: true } });
    flushSync();
    const input = document.querySelector<HTMLInputElement>('input[placeholder="e.g. D:/data/my-dataset"]')!;
    input.value = '"D:\\data\\shapes"';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.placeholder).toBe('shapes');
    unmount(c);
    const go = await register(' "D:\\data\\shapes" ', [{ id: 'ds_fasteners' }]);
    expect(go).toHaveBeenCalledWith('/datasets/ds_shapes');
  });

  it('asks for a full path in the form instead of sending a relative one', async () => {
    const create = vi.spyOn(hub.api, 'createDataset');
    c = mount(AddDatasetDialog, { target, props: { open: true } });
    flushSync();
    const input = document.querySelector<HTMLInputElement>('input[placeholder="e.g. D:/data/my-dataset"]')!;
    input.value = 'bccd';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    input.form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    flushSync();
    expect(document.querySelector('.msg.error')?.textContent).toContain('Use the full path, such as D:/data/my-dataset.');
    expect(create).not.toHaveBeenCalled();
  });

  it('announces a dataset that was not in the list as new', async () => {
    const go = await register('D:/data/shapes', [{ id: 'ds_fasteners' }]);
    expect(toast.items[0]).toMatchObject({ tone: 'success', title: 'Dataset registered' });
    expect(go).toHaveBeenCalledWith('/datasets/ds_shapes');
  });
});
