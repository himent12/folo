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
});

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  vi.restoreAllMocks();
});

function open(path: string) {
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
