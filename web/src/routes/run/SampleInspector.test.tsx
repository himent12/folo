// @vitest-environment happy-dom
// The validation sample inspector: the file name in its title, Previous / Next and the arrow keys.
import { flush as flushSync, signal } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import SampleInspector, { imageFileName } from './SampleInspector';
import Samples from './Samples';

let target: HTMLElement;
afterEach(() => target?.remove());

describe('the sample inspector', () => {
  it('names the image by its file and steps through the samples by button and arrow key', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    hub.mode = 'mock';
    const runs = await api.runs();
    const done = runs.find((r) => r.status === 'completed' && r.config.dataset_id === 'ds_shapes')!;
    const { samples, epoch } = await api.runSamples(done.id);
    const files = await Promise.all(samples.slice(0, 2).map(async (s) => (await api.datasetImages('ds_shapes', { offset: Number(s.image_id.replace(/^\D+/, '')), limit: 1 })).images[0]!.file.split('/').pop()));
    const index = signal<number | null>(0);
    target = document.createElement('div');
    document.body.appendChild(target);
    // happy-dom has no modal dialogs: render the dialog's content as if shown.
    HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
      this.open = true;
    };
    const c = render(
      () => (
        <SampleInspector
          samples={samples}
          index={index.get()}
          onindex={(i) => index.set(i)}
          epoch={epoch}
          classes={['circle', 'square', 'triangle']}
          preds={(s) => s.pred}
          truth={(s) => s.truth}
          describe={() => ''}
        />
      ),
      target,
    );
    const title = () => document.querySelector('dialog h2')?.textContent;
    await vi.waitFor(() => {
      flushSync();
      expect(title()).toBe(`${files[0]} · epoch ${epoch}`);
    });
    const button = (label: string) => [...document.querySelectorAll<HTMLButtonElement>('dialog footer button')].find((b) => b.textContent?.trim() === label)!;
    expect(button('Previous').disabled).toBe(true);
    button('Next').click();
    flushSync();
    expect(index.get()).toBe(1);
    await vi.waitFor(() => {
      flushSync();
      expect(title()).toBe(`${files[1]} · epoch ${epoch}`);
    });
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
    flushSync();
    expect(index.get()).toBe(0);
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
    expect(index.get()).toBe(0);

    // Position indicator is a polite live region.
    const pos = document.querySelector<HTMLElement>('dialog footer .pos')!;
    expect(pos.getAttribute('aria-live')).toBe('polite');

    // Another modal dialog on top suppresses arrow stepping underneath.
    const topDialog = document.createElement('dialog');
    topDialog.open = true;
    document.body.appendChild(topDialog);
    try {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
      flushSync();
      expect(index.get()).toBe(0);
    } finally {
      topDialog.remove();
    }
    c.unmount();
  });
});

describe('imageFileName', () => {
  it('asks each server once, never answering for one with another’s name (a switch of server)', async () => {
    const a = new MockApi('normal', 0);
    const b = new MockApi('normal', 0);
    const ask = (api: MockApi, file: string) => vi.spyOn(api, 'datasetImages').mockResolvedValue({ total: 1, images: [{ id: 'img_3', file, split: 'val', width: 64, height: 64, boxes: [] }] });
    const fromA = ask(a, 'D:/a/one.jpg');
    const fromB = ask(b, 'E:/b/other.jpg');
    hub.api = a;
    expect(await imageFileName('ds_x', 'img_3')).toBe('one.jpg');
    expect(await imageFileName('ds_x', 'img_3')).toBe('one.jpg');
    hub.api = b;
    expect(await imageFileName('ds_x', 'img_3')).toBe('other.jpg');
    expect([fromA.mock.calls.length, fromB.mock.calls.length]).toEqual([1, 1]);
  });

  it('returns null without querying offset 0 for non-numeric image ids', async () => {
    const api = new MockApi('normal', 0);
    const spy = vi.spyOn(api, 'datasetImages');
    hub.api = api;
    expect(await imageFileName('ds_x', 'no_digits')).toBeNull();
    expect(spy).not.toHaveBeenCalled();
  });
});

describe('Samples epoch and inspector bounds', () => {
  it('resets a picked epoch past epochsDone back to latest and closes the inspector if samples shrink', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    hub.mode = 'mock';
    const runs = await hub.api.runs();
    const done = runs.find((r) => r.status === 'completed' && r.config.dataset_id === 'ds_shapes')!;
    const full = await hub.api.runSamples(done.id, 10);
    const spy = vi.spyOn(hub.api, 'runSamples').mockImplementation(async (_id, ep) => ({
      epoch: ep ?? 1,
      samples: ep === 5 ? full.samples.slice(0, 1) : full.samples.slice(0, 4),
    }));
    vi.spyOn(hub.api, 'imageUrl').mockResolvedValue('data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7');
    const epochsDone = signal(10);
    target = document.createElement('div');
    document.body.appendChild(target);
    HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
      this.open = true;
    };
    const c = render(() => <Samples runId={done.id} classes={['circle', 'square', 'triangle']} epochsDone={epochsDone.get()} />, target);
    await vi.waitFor(() => {
      flushSync();
      expect(target.querySelectorAll('.cell').length).toBe(4);
    });
    // Pick epoch 10, wait for its samples, then open sample index 2.
    const sel = target.querySelector<HTMLSelectElement>('select[aria-label="Epoch"]')!;
    sel.value = '10';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    await vi.waitFor(() => {
      flushSync();
      expect(spy.mock.calls.some((call) => call[1] === 10)).toBe(true);
      expect(target.querySelectorAll('.cell').length).toBe(4);
    });
    target.querySelectorAll<HTMLButtonElement>('.cell')[2]!.click();
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelector<HTMLDialogElement>('dialog[open]')).not.toBeNull();
    });

    // Run rewinds to epoch 5 with only 1 sample: picker falls back to latest and out-of-bounds inspector closes.
    epochsDone.set(5);
    await vi.waitFor(() => {
      flushSync();
      expect(sel.value).toBe('latest');
      expect(target.querySelectorAll('.cell').length).toBe(1);
      expect(document.querySelector<HTMLDialogElement>('dialog[open]')).toBeNull();
    });
    expect(spy.mock.calls.at(-1)?.[1]).toBe(5);
    c.unmount();
  });
});
