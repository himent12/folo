// @vitest-environment happy-dom
// A labeling job whose dataset was removed from FOLO (sweep 4, #319): the job keeps the dataset's id, and its page linked
// to it for good ("made a dataset" opening "Dataset not found"). It now links only to a dataset the list has, says when
// it was removed (also while the page is open), and links nowhere while the list loads.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import LabelJob from './LabelJob';

let target: HTMLElement;
let c: Rendered | undefined;

afterEach(() => {
  tests++;
  c?.unmount();
  c = undefined;
  target?.remove();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

// Fake timers here are plain: with `shouldAdvanceTime` the clock also follows real time, so a machine busy enough to
// stall the test for a poll's 5 s gets an extra poll (and an extra datasets read) the test did not ask for.
// A wait ends when its condition holds. What it waits on is work, not time (the mock answers without timers), so it has
// no bound of its own: a hang ends at the test's timeout (`vi.waitFor`'s 4 s and 1 s were exceeded on a busy machine).
// Under fake timers each step moves the clock by `step` ms, as `vi.waitFor` did. A wait whose test has ended (timed out)
// stops and touches nothing more: it would otherwise read the next test's page and move the next test's clock.
const realTimeout = globalThis.setTimeout.bind(globalThis);
let tests = 0;
async function until(holds: () => boolean, step = 10): Promise<void> {
  const mine = tests;
  for (;;) {
    if (tests !== mine) return new Promise<void>(() => {});
    flushSync();
    if (holds()) return;
    if (vi.isFakeTimers()) await vi.advanceTimersByTimeAsync(step);
    await new Promise((r) => realTimeout(r, vi.isFakeTimers() ? 0 : step));
  }
}
const made = () => target.querySelector('[data-made]');
const link = () => made()?.querySelector('a[href^="#/datasets/"]');

/** The job page of `job` (`lb_51d0`) on `api`, once `ready` holds. */
async function open(api: MockApi, ready: () => boolean, job = 'lb_51d0') {
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse(`#/datasets/label/${job}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <LabelJob />, target);
  await until(ready);
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
    await until(() => !!link());
    expect(link()?.getAttribute('href')).toBe(`#/datasets/${ds.id}`);
    expect(made()?.textContent).toBe(`made ${ds.name}`);
  });

  it('stops linking when the dataset is removed while the page is open', async () => {
    vi.useFakeTimers();
    const api = new MockApi('normal', 0);
    const ds = await api.createLabelDataset('lb_51d0', {});
    await open(api, () => !!link());
    await api.deleteDataset(ds.id);
    await vi.advanceTimersByTimeAsync(5000);
    await until(() => !!made()?.textContent?.includes('removed'));
    expect(link()).toBeNull();
  });

  // Review of #335, round 3: the poll that lacked the removed dataset was requested after the page knew its id, so it
  // settles it alone; the page read the list once more to confirm, kept the link meanwhile, and for good when that failed.
  it('stops linking after one poll without the dataset, and a failed read after it changes nothing', async () => {
    vi.useFakeTimers();
    const api = new MockApi('normal', 0);
    const ds = await api.createLabelDataset('lb_51d0', {});
    const real = api.datasets.bind(api);
    let good = Infinity;
    const reads = vi.spyOn(api, 'datasets').mockImplementation(async () => {
      if (reads.mock.calls.length > good) throw new ApiError(503, 'unavailable', 'The server is busy.');
      return real();
    });
    await open(api, () => !!link());
    await api.deleteDataset(ds.id);
    good = reads.mock.calls.length + 1; // the next poll answers (without the dataset); every read after it fails
    await vi.advanceTimersByTimeAsync(5000);
    await until(() => !!made()?.textContent?.includes('removed'));
    expect(reads).toHaveBeenCalledTimes(good);
    await vi.advanceTimersByTimeAsync(5000); // a poll that fails
    expect(reads.mock.calls.length).toBeGreaterThan(good);
    flushSync();
    expect(made()?.textContent).toBe('made a dataset, since removed from FOLO');
    expect(link()).toBeNull();
  });

  // Review of #335, round 3: a job that is not there names no dataset, yet its "not found" page read the list every 5 s.
  it('reads the datasets list at most once for a job that is not found', async () => {
    vi.useFakeTimers();
    const api = new MockApi('normal', 0);
    const reads = vi.spyOn(api, 'datasets');
    await open(api, () => !!target.textContent?.includes('Labeling job not found'), 'lb_gone');
    const atLoad = reads.mock.calls.length;
    expect(atLoad).toBeLessThanOrEqual(1);
    await vi.advanceTimersByTimeAsync(15_000);
    expect(reads).toHaveBeenCalledTimes(atLoad);
  });

  it('never calls a dataset made in another tab removed: the list is read again for it', async () => {
    vi.useFakeTimers();
    const api = new MockApi('normal', 0);
    const reads = vi.spyOn(api, 'datasets');
    const jobReads = vi.spyOn(api, 'labelJob');
    await open(api, () => !!target.querySelector('h1') && !target.querySelector('[aria-busy="true"]'));
    // The stream opens (30 ms) and the page reads the job again for it, all before the export: only the export's own
    // stream event (as the server's) can tell the page the new dataset.
    await until(() => jobReads.mock.calls.length >= 2);
    await vi.advanceTimersByTimeAsync(100);
    expect(made()).toBeNull();
    const shown: string[] = [];
    const before = reads.mock.calls.length;
    const at = Date.now();
    const ds = await api.createLabelDataset('lb_51d0', {});
    await until(() => (shown.push(made()?.textContent ?? ''), !!link() || Date.now() - at >= 5000), 1);
    expect(link(), 'linked by the read for it, before any poll').not.toBeNull();
    expect(reads.mock.calls.length).toBeGreaterThan(before);
    expect(made()?.textContent).toBe(`made ${ds.name}`);
    expect(shown.some((t) => t.includes('removed'))).toBe(false);
  });

  // Review of #335: the job exported again (another tab) after its first export was removed; the list the page has
  // predates the new export and the read for it fails (503). An older list without the id is no reason to call it
  // removed: the line claims nothing until a read succeeds.
  it('never calls a dataset removed when the read for it fails', async () => {
    vi.useFakeTimers();
    const api = new MockApi('normal', 0);
    const ds = await api.createLabelDataset('lb_51d0', {});
    const real = api.datasets.bind(api);
    let reads = 0;
    let failing = true;
    vi.spyOn(api, 'datasets').mockImplementation(async () => {
      reads++;
      if (reads === 1) return (await real()).filter((d) => d.id !== ds.id);
      if (failing) throw new ApiError(503, 'unavailable', 'The server is busy.');
      return real();
    });
    const shown: string[] = [];
    const look = () => (flushSync(), shown.push(made()?.textContent ?? ''), shown[shown.length - 1]!);
    await open(api, () => !!look() && reads >= 2);
    await vi.advanceTimersByTimeAsync(0);
    expect(look()).toBe('made a dataset');
    expect(link()).toBeNull();
    await vi.advanceTimersByTimeAsync(5000); // a poll that fails too
    expect(reads).toBeGreaterThanOrEqual(3);
    expect(look()).toBe('made a dataset');
    failing = false;
    await vi.advanceTimersByTimeAsync(5000);
    await until(() => !!look() && !!link());
    expect(made()?.textContent).toBe(`made ${ds.name}`);
    expect(shown.some((t) => t.includes('removed'))).toBe(false);
  });

  // Review of #335, round 2: a job that made no dataset links to none, yet its page read the datasets every 5 s. It reads
  // them once while the job loads, as main does (an exported job's link would otherwise wait for the job, then the list).
  it('polls no datasets list for a job that made no dataset', async () => {
    vi.useFakeTimers();
    const api = new MockApi('normal', 0);
    const reads = vi.spyOn(api, 'datasets');
    await open(api, () => !!target.querySelector('h1') && !target.querySelector('[aria-busy="true"]'));
    const atLoad = reads.mock.calls.length;
    expect(atLoad).toBeLessThanOrEqual(1);
    await vi.advanceTimersByTimeAsync(15_000);
    expect(made()).toBeNull();
    expect(reads).toHaveBeenCalledTimes(atLoad);
  });
});
