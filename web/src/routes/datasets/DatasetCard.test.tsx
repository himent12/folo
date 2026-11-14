// @vitest-environment happy-dom
// A dataset card's mosaic (design/UNIFIED.md §4, Datasets) follows the list: a folder found missing has no tiles to
// read, and when a refresh finds it again (or brings new images) the card reads its first images afresh, never keeping
// a skeleton or the old set.
import { flush, signal } from 'fw';
import { afterEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import DatasetCard from './DatasetCard';
import type { ListedDataset } from './problem';

let target: HTMLElement | undefined;
let view: { unmount(): void } | undefined;
afterEach(() => {
  view?.unmount();
  view = undefined;
  target?.remove();
});

describe('DatasetCard', () => {
  it('reads its mosaic again when the refresh finds the folder back or the dataset changes', async () => {
    // happy-dom never reports a card in view: without an observer a card counts as seen at once.
    vi.stubGlobal('IntersectionObserver', undefined);
    onTestFinished(() => {
      vi.unstubAllGlobals();
    });
    const api = new MockApi('normal', 0);
    hub.api = api;
    hub.mode = 'mock';
    const [first] = await api.datasets();
    const asked = vi.spyOn(api, 'datasetImages');
    const ds = signal<ListedDataset>({ ...first!, problem: 'missing' });
    target = document.body.appendChild(document.createElement('div'));
    view = render(() => <DatasetCard dataset={ds.get()} />, target);
    flush();
    // Missing: the unavailable mark, nothing asked.
    expect(asked).not.toHaveBeenCalled();
    expect(target.querySelector('.mosaic .none')).not.toBeNull();

    // Found again: asked once, and the tiles fill in.
    ds.set({ ...first! });
    flush();
    expect(asked).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => {
      flush();
      if (!target!.querySelector('.mosaic .dsimg')) throw new Error('still waiting for the mosaic');
    });
    expect(target.querySelector('.mosaic .none')).toBeNull();

    // A refresh that brings the same dataset reads nothing; one that changes it (images added) reads again.
    ds.set({ ...first! });
    flush();
    expect(asked).toHaveBeenCalledTimes(1);
    ds.set({ ...first!, splits: { ...first!.splits, train: first!.splits.train + 8 } });
    flush();
    expect(asked).toHaveBeenCalledTimes(2);
  });

  it('offers Retry when its read fails, and tries again by itself on the next refresh', async () => {
    vi.stubGlobal('IntersectionObserver', undefined);
    onTestFinished(() => {
      vi.unstubAllGlobals();
    });
    const api = new MockApi('normal', 0);
    hub.api = api;
    hub.mode = 'mock';
    const [first] = await api.datasets();
    const real = api.datasetImages.bind(api);
    let down = true;
    const asked = vi.spyOn(api, 'datasetImages').mockImplementation((...a) => (down ? Promise.reject(new Error('offline')) : real(...a)));
    const ds = signal<ListedDataset>({ ...first! });
    target = document.body.appendChild(document.createElement('div'));
    view = render(() => <DatasetCard dataset={ds.get()} />, target);
    flush();
    const retry = await vi.waitFor(() => {
      flush();
      const b = target!.querySelector<HTMLButtonElement>('.mosaic .none button');
      if (!b) throw new Error('no Retry yet');
      return b;
    });
    expect(retry.getAttribute('aria-label')).toBe(`Retry the images of ${first!.name}`);
    expect(asked).toHaveBeenCalledTimes(1);

    // Retry while the server is still down: asked again, still failed.
    retry.click();
    flush();
    expect(asked).toHaveBeenCalledTimes(2);
    await vi.waitFor(() => {
      flush();
      if (!target!.querySelector('.mosaic .none button')) throw new Error('not failed again yet');
    });

    // The list refreshes (the server answers again) with the same dataset: the failed mosaic reads again and fills.
    down = false;
    ds.set({ ...first! });
    flush();
    expect(asked).toHaveBeenCalledTimes(3);
    await vi.waitFor(() => {
      flush();
      if (!target!.querySelector('.mosaic .dsimg')) throw new Error('still waiting for the mosaic');
    });
    expect(target.querySelector('.mosaic .none')).toBeNull();
  });
});
