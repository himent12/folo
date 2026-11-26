// @vitest-environment happy-dom
import { flush as flushSync, signal } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import { undescribedFields } from '$lib/ui/fieldA11y';
import AddDatasetDialog from './AddDatasetDialog';
import { pendingGeneration } from './changes';

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

async function register(path: string, known?: { id: string; name: string }[]) {
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

    const go = await register('D:\\data\\shapes\\', [{ id: 'ds_fasteners', name: 'fasteners' }, { id: 'ds_shapes', name: 'shapes' }]);
    expect(toast.items[0]).toMatchObject({ tone: 'info', title: 'Already registered as shapes' });
    expect(go).toHaveBeenCalledWith('/datasets/ds_shapes');
  });

  it('keeps a folder it is reading where a reload finds it, and forgets it when the answer comes (studio-data F1)', async () => {
    let answer!: () => void;
    const read = hub.api.createDataset.bind(hub.api);
    vi.spyOn(hub.api, 'createDataset').mockImplementation((b) => new Promise((ok) => (answer = () => ok(read(b)))));
    c = mount(AddDatasetDialog, { target, props: { open: true, known: [] } });
    flushSync();
    const input = document.querySelector<HTMLInputElement>('input[placeholder="e.g. D:/data/my-dataset"]')!;
    input.value = 'D:\\data\\shapes';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    input.form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await vi.waitFor(() => expect(answer).toBeDefined());
    expect(pendingGeneration(hub.server)).toMatchObject({ name: 'shapes', path: 'D:\\data\\shapes', what: 'Reading shapes' });
    vi.spyOn(router, 'go').mockImplementation(() => {});
    answer();
    await vi.waitFor(() => expect(pendingGeneration(hub.server)).toBeNull());
  });

  it('still waits for a folder whose request a reload cut off: the server goes on with it (studio-data F1)', async () => {
    let fail!: (e: unknown) => void;
    vi.spyOn(hub.api, 'createDataset').mockImplementation(() => new Promise((_, no) => (fail = no)));
    c = mount(AddDatasetDialog, { target, props: { open: true, known: [] } });
    flushSync();
    const input = document.querySelector<HTMLInputElement>('input[placeholder="e.g. D:/data/my-dataset"]')!;
    input.value = 'D:/data/coco';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    input.form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await vi.waitFor(() => expect(fail).toBeDefined());
    try {
      // The reload: the page is about to go, and its request fails as the browser drops it (before `pagehide`).
      window.dispatchEvent(new Event('beforeunload'));
      fail(new ApiError(0, 'network', 'Cannot reach the FOLO server.'));
      await new Promise((r) => setTimeout(r));
      expect(pendingGeneration(hub.server)).toMatchObject({ path: 'D:/data/coco', what: 'Reading coco' });
      expect(document.querySelector('.err')).toBeNull();
    } finally {
      // The page stays after all (here: back from the cache): the failure is said, and the dataset no longer waited for.
      window.dispatchEvent(new Event('pageshow'));
    }
    flushSync();
    expect(document.querySelector('.err')?.textContent).toContain("Can't reach the FOLO server");
    expect(pendingGeneration(hub.server)).toBeNull();
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
    const go = await register(' "D:\\data\\shapes" ', [{ id: 'ds_fasteners', name: 'fasteners' }]);
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

  it('locks the form while it generates, and after a dismissal says where the result is instead of jumping to it', async () => {
    let finish!: () => void;
    const ds = await hub.api.dataset('ds_shapes');
    vi.spyOn(hub.api, 'createSyntheticDataset').mockImplementation(() => new Promise((res) => (finish = () => res(ds))));
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    c = mount(AddDatasetDialog, { target, props: { open: true, mode: 'synthetic' } });
    flushSync();
    document.querySelector<HTMLFormElement>('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    flushSync();
    expect(document.querySelector('fieldset')!.disabled).toBe(true);
    expect(document.querySelector('p[role="status"]')?.textContent).toContain('Generating 1,000 images…');
    const close = [...document.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Close')!;
    close.click();
    flushSync();
    expect(toast.items.at(-1)).toMatchObject({ tone: 'info', title: 'Still generating 1,000 images' });
    finish();
    await vi.waitFor(() => expect(toast.items.at(-1)?.title).toBe('Synthetic dataset ready'));
    expect(go).not.toHaveBeenCalled();
    toast.items.at(-1)!.action!.run();
    expect(go).toHaveBeenCalledWith('/datasets/ds_shapes');
  });

  it('never lets a dismissed request move the person, also when the dialog was opened again meanwhile', async () => {
    let finish!: () => void;
    const ds = await hub.api.dataset('ds_shapes');
    vi.spyOn(hub.api, 'createSyntheticDataset').mockImplementation(() => new Promise((res) => (finish = () => res(ds))));
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const open = signal(true);
    c = render(() => <AddDatasetDialog bind:open={open} mode="synthetic" />, target);
    flushSync();
    document.querySelector<HTMLFormElement>('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    flushSync();
    open.set(false);
    flushSync();
    open.set(true);
    flushSync();
    const before = toast.items.length;
    finish();
    // The new opening stays open where it is, and says it there (a toast would sit behind it).
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelector('dialog')!.textContent).toContain('Synthetic dataset ready');
    });
    expect(open.peek()).toBe(true);
    expect(go).not.toHaveBeenCalled();
    expect(toast.items.length).toBe(before);
  });

  it('answers with a toast once the dialog is gone (Back while it runs), a failure too', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const ds = await hub.api.dataset('ds_shapes');
    let finish!: (ok: boolean) => void;
    vi.spyOn(hub.api, 'createSyntheticDataset').mockImplementation(
      () => new Promise((res, rej) => (finish = (ok) => (ok ? res(ds) : rej(new ApiError(500, 'internal', 'disk full'))))),
    );
    const start = () => {
      c = mount(AddDatasetDialog, { target, props: { open: true, mode: 'synthetic' } });
      flushSync();
      document.querySelector<HTMLFormElement>('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      flushSync();
      unmount(c);
      c = undefined;
    };
    start();
    finish(true);
    await vi.waitFor(() => expect(toast.items.at(-1)?.title).toBe('Synthetic dataset ready'));
    expect(go).not.toHaveBeenCalled();
    start();
    finish(false);
    await vi.waitFor(() => expect(toast.items.at(-1)).toMatchObject({ tone: 'error' }));
  });

  it('says where the images went when none can train (a COCO test file registered alone)', async () => {
    const ds = await hub.api.dataset('ds_shapes');
    vi.spyOn(hub.api, 'createDataset').mockResolvedValue({ ...ds, id: 'ds_new', name: 'bccd', splits: { train: 0, val: 0, test: 72 } });
    await register('D:/data/bccd/annotations/instances_test.json', [{ id: 'ds_shapes', name: 'shapes' }]);
    expect(toast.items[0]).toMatchObject({ title: 'Dataset registered', detail: 'bccd · all 72 images are in the test split, none to train on' });
  });

  // Sweep 4 #330: the seed had no upper bound, so one past 2^53 − 1 went out as another seed (the browser's rounding).
  it('generates with any seed a browser holds exactly, and names the limit past it', async () => {
    const create = vi.spyOn(hub.api, 'createSyntheticDataset');
    c = mount(AddDatasetDialog, { target, props: { open: true, mode: 'synthetic' } });
    flushSync();
    const field = [...document.querySelectorAll('.field')].find((f) => f.querySelector('label')?.textContent === 'Seed')!;
    const input = field.querySelector<HTMLInputElement>('input')!;
    const submit = (v: string) => {
      input.value = v;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      document.querySelector<HTMLFormElement>('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      flushSync();
    };
    // Checked as typed (lib/seed, rule A): a number field rounded 9007199254740990.5 to …990 first (#354 review).
    for (const t of ['9007199254740993', '9007199254740990.5', '1e3']) {
      submit(t);
      expect(field.querySelector('.msg.error')?.textContent).toBe('A whole number from 0 to 9,007,199,254,740,991.');
    }
    expect(create).not.toHaveBeenCalled();
    submit('4294967296');
    await vi.waitFor(() => expect(create).toHaveBeenCalledOnce());
    expect(create.mock.calls[0]![0].seed).toBe(4294967296);
  });

  it("takes the server's image counts, whole numbers from 1 to 100,000, as Desktop does (issue #382)", async () => {
    const create = vi.spyOn(hub.api, 'createSyntheticDataset');
    c = mount(AddDatasetDialog, { target, props: { open: true, mode: 'synthetic' } });
    flushSync();
    const field = [...document.querySelectorAll('.field')].find((f) => f.querySelector('label')?.textContent === 'Images')!;
    const input = field.querySelector<HTMLInputElement>('input')!;
    expect([input.min, input.max, input.step]).toEqual(['1', '100000', '1']);
    const submit = (v: string) => {
      input.value = v;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      document.querySelector<HTMLFormElement>('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      flushSync();
    };
    // 10.5 was rounded to 11 and sent; 0 and 100,001 are the server's 400.
    for (const t of ['10.5', '0', '100001']) {
      submit(t);
      expect(field.querySelector('.msg.error')?.textContent).toBe('A whole number from 1 to 100,000.');
    }
    expect(create).not.toHaveBeenCalled();
    submit('1');
    await vi.waitFor(() => expect(create).toHaveBeenCalledOnce());
    expect(create.mock.calls[0]![0].images).toBe(1);
  });

  it('takes as many classes as the generator has shapes (8), and says so', () => {
    c = mount(AddDatasetDialog, { target, props: { open: true, mode: 'synthetic' } });
    flushSync();
    const input = document.querySelector<HTMLInputElement>('input[id$="-classes"]')!;
    input.value = 'star, ring, cross, hexagon, heart, car, dog';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    flushSync();
    expect(document.querySelectorAll('.chip')).toHaveLength(8);
    expect(document.getElementById(input.getAttribute('aria-describedby')!.split(' ')[0]!)?.textContent).toContain('8 of 8');
  });

  it('numbers the default name when it is taken, and points out a typed name that is', async () => {
    const create = vi.spyOn(hub.api, 'createSyntheticDataset');
    vi.spyOn(router, 'go').mockImplementation(() => {});
    const known = [{ id: 'ds_a', name: 'shapes-synth' }, { id: 'ds_b', name: 'bccd' }];
    c = mount(AddDatasetDialog, { target, props: { open: true, mode: 'synthetic', known } });
    flushSync();
    const field = document.querySelector<HTMLInputElement>('input[id$="-sname"]')!;
    expect(field.placeholder).toBe('shapes-synth-2');
    field.value = 'BCCD';
    field.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(document.getElementById(field.getAttribute('aria-describedby')!.split(' ')[0]!)?.textContent).toBe(
      'A dataset is already called BCCD. Another name tells them apart.',
    );
    field.value = '';
    field.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    field.form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await vi.waitFor(() => expect(create).toHaveBeenCalled());
    expect(create.mock.calls[0]![0].name).toBe('shapes-synth-2');
  });

  it('points a folder of images without labels to Label images', async () => {
    vi.spyOn(hub.api, 'createDataset').mockRejectedValue(new ApiError(400, 'unknown_format', "'D:/data/images-only' has 2 images but no labels", 'label them first'));
    await register('D:/data/images-only', []);
    const err = document.querySelector('.err')!;
    expect(err.textContent).toContain('Label them first: an agent can draw the boxes in Label images.');
    expect(err.querySelector('a')?.getAttribute('href')).toBe('#/datasets/label');
  });

  it('announces a dataset that was not in the list as new', async () => {
    const go = await register('D:/data/shapes', [{ id: 'ds_fasteners', name: 'fasteners' }]);
    expect(toast.items[0]).toMatchObject({ tone: 'success', title: 'Dataset registered' });
    expect(go).toHaveBeenCalledWith('/datasets/ds_shapes');
  });

  it('focuses the first invalid field when submitting Register folder or Generate sample set fails validation', () => {
    const type = (el: HTMLInputElement, v: string) => {
      el.value = v;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
    };

    // Folder mode: relative path invalid -> focuses Folder input with aria-invalid already applied.
    c = mount(AddDatasetDialog, { target, props: { open: true, mode: 'folder' } });
    flushSync();
    const pathEl = document.querySelector<HTMLInputElement>('input[id$="-path"]')!;
    const nameEl = document.querySelector<HTMLInputElement>('input[id$="-name"]')!;
    const submitBtn = document.querySelector<HTMLButtonElement>('button[type="submit"]')!;
    type(pathEl, 'relative/folder');
    submitBtn.focus();
    pathEl.form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    expect(submitBtn.disabled).toBe(true);
    expect(pathEl.getAttribute('aria-invalid')).toBe('true');
    expect(document.activeElement).toBe(pathEl);

    // Fix path, break name -> focuses Name input.
    type(pathEl, 'D:/data/valid');
    type(nameEl, 'bad name!');
    submitBtn.focus();
    pathEl.form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    expect(document.activeElement).toBe(nameEl);
    unmount(c);

    // Synthetic mode: invalid images (10.5) and seed ('abc') -> focuses Images first, then Seed once Images is fixed.
    c = mount(AddDatasetDialog, { target, props: { open: true, mode: 'synthetic' } });
    flushSync();
    const imagesEl = document.querySelector<HTMLInputElement>('input[id$="-images"]')!;
    const seedEl = document.querySelector<HTMLInputElement>('input[id$="-seed"]')!;
    const genBtn = document.querySelector<HTMLButtonElement>('button[type="submit"]')!;
    type(imagesEl, '10.5');
    type(seedEl, 'abc');
    genBtn.focus();
    imagesEl.form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    expect(document.activeElement).toBe(imagesEl);
    type(imagesEl, '10');
    genBtn.focus();
    imagesEl.form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    expect(document.activeElement).toBe(seedEl);
  });
});
