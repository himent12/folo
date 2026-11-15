// @vitest-environment happy-dom
// Review round 2 (s6-shell): the distance condition's truth comes from a second request. Its failure was swallowed, so
// the previous condition's boxes stayed drawn over the moved image with no error. Now the truth is cleared while
// unknown and a failure says so, with Retry.
import { flush as flushSync, signal } from 'fw';
import { render, type Rendered } from '../../../framework/testing';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { ApiError } from '$lib/api/client';
import { MockApi } from '$lib/api/mock';
import type { BenchImagesQuery } from '$lib/api/types';
import Preview from './Preview';

// happy-dom lays nothing out, so DetectionView would draw no boxes: a stand-in shows the truth it was handed.
vi.mock('$lib/detect/DetectionView', async () => {
  const { effect } = await import('fw');
  return {
    default: (p: { truth?: unknown[] }) => {
      const el = document.createElement('div');
      el.className = 'dv';
      effect(() => {
        el.dataset.truth = JSON.stringify(p.truth ?? []);
      });
      return el;
    },
  };
});

let target: HTMLElement;
let c: Rendered | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  c?.unmount();
  c = undefined;
  target.remove();
  vi.restoreAllMocks();
});

async function settle(rounds = 12) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
}

const truth = (side: 'plain' | 'changed') => document.querySelector<HTMLElement>(`.stage .side.${side} .dv`)?.dataset.truth;

it('never leaves another condition’s boxes over the distance image, and retries its truth', async () => {
  const bench = await hub.api.bench('bn_9f03');
  const catalog = await hub.api.benchCatalog();
  vi.spyOn(hub.api, 'benchImageUrl').mockImplementation(async (_b, im, o) => `blob:${im}/${o?.condition ?? ''}`);
  const real = hub.api.benchImages.bind(hub.api);
  let failTruth = false;
  const list = vi.spyOn(hub.api, 'benchImages').mockImplementation(async (id: string, q?: BenchImagesQuery) => {
    if (failTruth && q?.condition === 'distance') throw new ApiError(200, 'bad_response', 'The server sent an answer of an unexpected shape (no page of images).');
    return real(id, q);
  });
  const cond = signal('haze');
  c = render(() => <Preview bench={bench} catalog={catalog} bind:condition={cond} />, target);
  await settle();
  const plain = truth('plain');
  expect(JSON.parse(plain!).length).toBeGreaterThan(0);
  expect(truth('changed')).toBe(plain);

  failTruth = true;
  cond.set('distance');
  await settle();
  expect(truth('changed')).toBe('[]');
  expect(target.textContent).toMatch(/Could not load the boxes under/);
  const retry = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => /Retry|Try again/.test(b.textContent ?? ''));
  expect(retry).toBeDefined();

  failTruth = false;
  const calls = list.mock.calls.length;
  retry!.click();
  await settle();
  expect(list.mock.calls.length).toBe(calls + 1);
  expect(target.textContent).not.toMatch(/Could not load the boxes under/);
  const far = (await real(bench.id, { offset: 0, limit: 1, condition: 'distance', severity: 3 })).images[0]!.boxes;
  expect(truth('changed')).toBe(JSON.stringify(far));
  expect(truth('changed')).not.toBe(plain);
});

// Review round 3: the first page of images rejected (a page of the wrong shape) showed an error with no Retry, and
// nothing but remounting the preview loaded it again.
it('retries the first images after they failed to load', async () => {
  const bench = await hub.api.bench('bn_9f03');
  const catalog = await hub.api.benchCatalog();
  vi.spyOn(hub.api, 'benchImageUrl').mockImplementation(async (_b, im, o) => `blob:${im}/${o?.condition ?? ''}`);
  const real = hub.api.benchImages.bind(hub.api);
  let fail = true;
  const list = vi.spyOn(hub.api, 'benchImages').mockImplementation(async (id: string, q?: BenchImagesQuery) => {
    if (fail) throw new ApiError(200, 'bad_response', 'The server sent an answer of an unexpected shape (no page of images).');
    return real(id, q);
  });
  c = render(() => <Preview bench={bench} catalog={catalog} condition="haze" />, target);
  await settle();
  expect(target.textContent).toMatch(/Could not load the images/);
  const retry = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => /Retry|Try again/.test(b.textContent ?? ''));
  expect(retry).toBeDefined();

  fail = false;
  const calls = list.mock.calls.length;
  retry!.click();
  await settle();
  expect(list.mock.calls.length).toBeGreaterThan(calls);
  expect(target.textContent).not.toMatch(/Could not load the images/);
  expect(JSON.parse(truth('plain')!).length).toBeGreaterThan(0);
});
