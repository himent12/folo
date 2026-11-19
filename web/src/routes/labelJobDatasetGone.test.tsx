// @vitest-environment happy-dom
// A labeling job whose dataset was removed from FOLO (sweep 4, #319): the job keeps the dataset's id, and its page linked
// to it for good ("made a dataset" opening "Dataset not found"). It now links only to a dataset the list has, says when
// it was removed (also while the page is open), and links nowhere while the list loads.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import LabelJob from './LabelJob';

let target: HTMLElement;
let c: Rendered | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

const made = () => target.querySelector('[data-made]');
const link = () => made()?.querySelector('a[href^="#/datasets/"]');

/** The job page of `lb_51d0` on `api`, once its header names the dataset the job made. */
async function open(api: MockApi, until: () => boolean) {
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse('#/datasets/label/lb_51d0');
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <LabelJob />, target);
  await vi.waitFor(
    () => {
      flushSync();
      if (!until()) throw new Error('not yet');
    },
    { timeout: 4000, interval: 10 },
  );
}

describe('a labeling job whose dataset was removed', () => {
  it('says the dataset was removed from FOLO and links nowhere', async () => {
    const api = new MockApi('normal', 0);
    const ds = await api.createLabelDataset('lb_51d0', {});
    await api.deleteDataset(ds.id);
    await open(api, () => !!made()?.textContent?.includes('removed'));
    expect(made()?.textContent).toBe('made a dataset, since removed from FOLO');
    expect(link()).toBeNull();
    expect(made()?.querySelector('[title]')?.getAttribute('title')).toMatch(/no longer registered in FOLO\. Its files stay/);
  });

  it('links nowhere and calls nothing removed while the datasets load', async () => {
    const api = new MockApi('normal', 0);
    const ds = await api.createLabelDataset('lb_51d0', {});
    const datasets = api.datasets.bind(api);
    let release!: () => void;
    const held = new Promise<void>((r) => (release = r));
    vi.spyOn(api, 'datasets').mockImplementation(async () => (await held, datasets()));
    await open(api, () => !!made());
    expect(made()?.textContent).toBe('made a dataset');
    expect(link()).toBeNull();
    release();
    await vi.waitFor(() => {
      flushSync();
      if (!link()) throw new Error('not linked');
    });
    expect(link()?.getAttribute('href')).toBe(`#/datasets/${ds.id}`);
    expect(made()?.textContent).toBe(`made ${ds.name}`);
  });

  it('stops linking when the dataset is removed while the page is open', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const api = new MockApi('normal', 0);
    const ds = await api.createLabelDataset('lb_51d0', {});
    await open(api, () => !!link());
    await api.deleteDataset(ds.id);
    await vi.advanceTimersByTimeAsync(5000);
    await vi.waitFor(
      () => {
        flushSync();
        if (!made()?.textContent?.includes('removed')) throw new Error('still linked');
      },
      { timeout: 4000, interval: 10 },
    );
    expect(link()).toBeNull();
  });

  it('never calls a dataset made in another tab removed: the list is read again for it', async () => {
    const api = new MockApi('normal', 0);
    await open(api, () => !!target.querySelector('h1') && !target.querySelector('[aria-busy="true"]'));
    expect(made()).toBeNull();
    const shown: string[] = [];
    const ds = await api.createLabelDataset('lb_51d0', {});
    await vi.waitFor(
      () => {
        flushSync();
        shown.push(made()?.textContent ?? '');
        if (!link()) throw new Error('not linked');
      },
      { timeout: 4000, interval: 1 },
    );
    expect(made()?.textContent).toBe(`made ${ds.name}`);
    expect(shown.some((t) => t.includes('removed'))).toBe(false);
  });
});
