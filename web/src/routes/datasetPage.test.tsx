// @vitest-environment happy-dom
// The dataset page: what it offers for the dataset it shows.
import { flush as flushSync } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Dataset from './Dataset';

let target: HTMLElement;
let c: { unmount(): void } | undefined;
let api: MockApi;

beforeEach(() => {
  api = new MockApi('normal', 0);
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

function open(path: string) {
  history.replaceState(history.state, '', `#${path}`);
  router.route = parse(`#${path}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <Dataset />, target);
  flushSync();
}

const until = <T,>(f: () => T | null | undefined, what: string) =>
  vi.waitFor(() => {
    flushSync();
    const v = f();
    if (!v) throw new Error(`still waiting for ${what}`);
    return v;
  });

describe('Dataset page', () => {
  it('says a dataset with no training images cannot train, and turns Train off with the reason', async () => {
    const read = api.dataset.bind(api);
    vi.spyOn(api, 'dataset').mockImplementation(async (id: string) => ({ ...(await read(id)), splits: { train: 0, val: 0, test: 72 } }));
    open('/datasets/ds_shapes');
    const note = await until(() => target.querySelector('.no-train'), 'the notice');
    expect(note.textContent).toContain('No training images: all 72 images are in the test split.');
    const train = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Train on this dataset'))!;
    expect(train.disabled).toBe(true);
    expect(document.getElementById(train.getAttribute('aria-describedby')!)?.textContent).toBe('No training images in this dataset');
  });

  it('offers Train as a link for a dataset that can train', async () => {
    open('/datasets/ds_shapes');
    const train = await until(() => [...target.querySelectorAll('a')].find((a) => a.textContent?.includes('Train on this dataset')), 'Train');
    expect(train.getAttribute('href')).toBe('#/runs/new?dataset=ds_shapes');
    expect(target.querySelector('.no-train')).toBeNull();
  });
});

describe('Dataset page history', () => {
  const hash = () => decodeURIComponent(location.hash);

  it('opens an image as a history entry that Back (or Close) takes away, leaving the grid', async () => {
    open('/datasets/ds_shapes');
    const cell = await until(() => target.querySelector<HTMLElement>('[data-i="3"]'), 'the grid');
    const depth = history.length;
    cell.click();
    await until(() => hash().includes('image='), 'the image in the URL');
    expect(history.length).toBe(depth + 1);
    const back = vi.spyOn(history, 'back');
    document.querySelector('dialog.insp')!.dispatchEvent(new Event('close'));
    expect(back).toHaveBeenCalledOnce();
    await until(() => hash() === '#/datasets/ds_shapes', 'the grid again');
  });

  it('still knows where an image was opened from after Back and Forward, so Close goes back instead of doubling the grid', async () => {
    open('/datasets/ds_shapes');
    const cell = await until(() => target.querySelector<HTMLElement>('[data-i="2"]'), 'the grid');
    cell.click();
    await until(() => hash().includes('image='), 'the image in the URL');
    history.back();
    await until(() => hash() === '#/datasets/ds_shapes', 'the grid after Back');
    history.forward();
    await until(() => hash().includes('image='), 'the image after Forward');
    const back = vi.spyOn(history, 'back');
    const replace = vi.spyOn(history, 'replaceState');
    await until(() => document.querySelector('dialog.insp'), 'the inspector');
    document.querySelector('dialog.insp')!.dispatchEvent(new Event('close'));
    expect(back).toHaveBeenCalledOnce();
    expect(replace).not.toHaveBeenCalled();
    await until(() => hash() === '#/datasets/ds_shapes', 'the grid again');
  });

  it('replaces an image entry reached by a link when it closes (there is no grid entry behind it)', async () => {
    open('/datasets/ds_shapes?image=1');
    await until(() => document.querySelector('dialog.insp'), 'the inspector');
    const back = vi.spyOn(history, 'back');
    document.querySelector('dialog.insp')!.dispatchEvent(new Event('close'));
    expect(back).not.toHaveBeenCalled();
    await until(() => hash() === '#/datasets/ds_shapes', 'the grid in place of the image');
  });

  it('adds an entry for the first filter only, and Clear goes back to the unfiltered view', async () => {
    open('/datasets/ds_shapes');
    await until(() => target.querySelector('[data-i="0"]'), 'the grid');
    const depth = history.length;
    const pick = (label: string) => [...target.querySelectorAll<HTMLButtonElement>('.ds-toolbar button')].find((b) => b.textContent?.startsWith(label))!;
    pick('Val').click();
    await until(() => hash().includes('split=val'), 'the val filter');
    pick('Train').click();
    await until(() => hash().includes('split=train'), 'the train filter');
    expect(history.length).toBe(depth + 1);
    const back = vi.spyOn(history, 'back');
    target.querySelector<HTMLButtonElement>('button[aria-label="Clear filters"]')!.click();
    expect(back).toHaveBeenCalledOnce();
    await until(() => hash() === '#/datasets/ds_shapes', 'the unfiltered view');
  });
});
