// @vitest-environment happy-dom
// The validation sample inspector: the file name in its title, Previous / Next and the arrow keys.
import { flush as flushSync, signal } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import SampleInspector, { imageFileName } from './SampleInspector';

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
});
