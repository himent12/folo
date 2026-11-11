// @vitest-environment happy-dom
// Overview against the mock server: it reads the run list the sidebar reads (shell/runList), so the Training card
// never says "Nothing is training" beside the sidebar's "Training" (shell F13), and Best models catches up when a run
// ends (runs F4).
import { flush } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Run } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { watched } from '$lib/shell/runList';
import Overview from './Overview';

let target: HTMLElement;
let c: { unmount(): void } | undefined;

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  watched.runs = undefined;
  watched.since = null;
});

function open(api: MockApi) {
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse('#/');
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <Overview />, target);
  flush();
}

/** The Training card's text once `ready` holds for it. */
const training = (ready: (text: string) => boolean, what: string) =>
  vi.waitFor(() => {
    flush();
    const text = target.querySelector('.live')?.textContent ?? '';
    if (!ready(text)) throw new Error(`still waiting for ${what}: ${text}`);
    return text;
  });

/** The mock's runs with none training, so the slot is whatever a test puts in it. */
async function idleRuns(): Promise<Run[]> {
  const list = await new MockApi('normal', 0).runs();
  return list.map((r) => (r.status === 'running' || r.status === 'stopping' || r.status === 'queued' ? { ...r, status: 'completed', queue_position: null } : r));
}

describe('Overview', () => {
  it('shows the run the sidebar shows, before its own list has loaded', async () => {
    const runs = await idleRuns();
    const run = runs.find((r) => r.status === 'completed')!;
    // This page's own requests never answer; the shell's list already has the run training.
    watched.runs = runs.map((r) => (r.id === run.id ? { ...r, status: 'running' } : r));
    open(new MockApi('loading'));
    const text = await training((t) => t.includes(run.name), 'the run');
    expect(text).not.toContain('Nothing is training');
  });

  it('shows a run the worker has taken as starting, not as nothing training', async () => {
    const runs = await idleRuns();
    const run = runs.find((r) => r.status === 'completed')!;
    watched.runs = runs.map((r) => (r.id === run.id ? { ...r, status: 'queued', queue_position: null, epoch: 0, step: 0, best: null } : r));
    open(new MockApi('loading'));
    const text = await training((t) => t.includes(run.name), 'the starting run');
    expect(text).toContain('Starting');
    expect(text).not.toContain('Queued');
  });

  it('loads in the shape of the common state, not of a training run', () => {
    open(new MockApi('loading'));
    const slot = target.querySelector('.live')!;
    expect(slot.querySelector('.idle-sk')).not.toBeNull();
    expect(slot.textContent).not.toContain('Epoch');
  });

  it('names a run by its dataset, a placeholder while datasets load, never by a raw id', async () => {
    const runs = await idleRuns();
    watched.runs = runs;
    const api = new MockApi('normal', 0);
    const recent = () => target.querySelector('#recent-runs tbody')!;
    let answer: (v: Awaited<ReturnType<MockApi['datasets']>>) => void = () => {};
    api.datasets = () => new Promise((res) => (answer = res));
    open(api);
    await vi.waitFor(() => expect(recent().querySelectorAll('tr.link').length).toBeGreaterThan(0));
    expect(recent().textContent).not.toMatch(/ds_/);
    expect(recent().querySelector('.ds .sk')).not.toBeNull();

    // The datasets arrive without the first run's: it was deleted.
    const all = await new MockApi('normal', 0).datasets();
    const gone = runs.find((r) => r.status === 'completed')!.config.dataset_id;
    answer(all.filter((d) => d.id !== gone));
    await vi.waitFor(() => {
      flush();
      expect(recent().textContent).toContain('deleted dataset');
    });
    expect(recent().textContent).not.toMatch(/ds_/);
  });

  it('names what failed, and does not blame the split when the runs failed to load', async () => {
    const api = new MockApi('normal', 0);
    api.runs = () => Promise.reject(new ApiError(500, 'internal', 'the run index could not be read'));
    open(api);
    const text = () => target.textContent ?? '';
    await vi.waitFor(() => {
      flush();
      expect(text()).toContain('Could not load recent runs');
      expect(target.querySelector('#best-models .unranked')).not.toBeNull();
    });
    expect(text()).toContain('Could not check what is training');
    expect(text()).not.toContain('Could not load this');
    expect(target.querySelector('#best-models .unranked')!.textContent).toMatch(/not ranked: could not load runs\.$/);
  });

  it('with no dataset left, its actions open Add dataset rather than a New run that cannot start', async () => {
    watched.runs = await idleRuns();
    const api = new MockApi('normal', 0);
    api.datasets = async () => [];
    open(api);
    const links = await vi.waitFor(() => {
      flush();
      const found = [...target.querySelectorAll<HTMLAnchorElement>('.live a, #datasets a')].filter((a) => a.textContent?.includes('Add dataset'));
      if (found.length < 2) throw new Error('still waiting for the empty states');
      return found;
    });
    for (const a of links) expect(a.getAttribute('href')).toBe('#/datasets?add=folder');
    expect(target.querySelector('.live')!.textContent).not.toContain('New run');
  });

  it('counts one class and one image in the singular', async () => {
    const api = new MockApi('normal', 0);
    const [first] = await api.datasets();
    const one = { ...first!, classes: ['part'], splits: { train: 1, val: 0, test: 0 } };
    api.datasets = async () => [one];
    api.dataset = async () => one;
    open(api);
    const card = await vi.waitFor(() => {
      flush();
      const li = target.querySelector('#datasets li.d-card');
      if (!li) throw new Error('still waiting for the datasets');
      return li;
    });
    expect(card.textContent).toContain('1 image');
    expect(card.textContent).toContain('1 class');
    expect(card.textContent).not.toMatch(/1 (images|classes)/);
  });

  it('shows a labeling job and a benchmark at work, which are not training runs', async () => {
    watched.runs = await idleRuns();
    const api = new MockApi('normal', 0);
    const [job] = await api.labelJobs();
    const [bench] = await api.benches();
    api.labelJobs = async () => [{ ...job!, status: 'running', run: { ...job!.run!, finished: null, total: 40, done: 12 } }];
    api.benches = async () => [{ ...bench!, active: true }];
    open(api);
    const strip = await vi.waitFor(() => {
      flush();
      const el = target.querySelector('.activity');
      if (!el?.textContent?.includes(bench!.name)) throw new Error('still waiting for the strip');
      return el;
    });
    expect(strip.textContent).toContain(job!.name);
    expect(strip.textContent).toContain('12 of 40 images');
    expect(strip.querySelector('a')!.getAttribute('href')).toBe(`#/datasets/label/${job!.id}`);
    // Nothing at work: nothing to show, once both lists have answered.
    const answers: Promise<unknown>[] = [];
    api.labelJobs = () => (answers[0] = Promise.resolve([{ ...job!, status: 'idle' as const }]));
    api.benches = () => (answers[1] = Promise.resolve([{ ...bench!, active: false }]));
    c!.unmount();
    target.remove();
    open(api);
    await vi.waitFor(() => expect(answers.filter(Boolean)).toHaveLength(2));
    await Promise.all(answers);
    flush();
    expect(target.querySelector('.activity')).toBeNull();
  });

  it('points people with unlabeled images to Label images from the first step', async () => {
    open(new MockApi('empty', 0));
    const link = await vi.waitFor(() => {
      flush();
      const a = [...target.querySelectorAll<HTMLAnchorElement>('.onboard a')].find((x) => x.textContent?.includes('Label them with an agent'));
      if (!a) throw new Error('still waiting for the first-run card');
      return a;
    });
    expect(link.getAttribute('href')).toBe('#/datasets/label');
  });

  it('refetches Best models when a run ends, and not on an unchanged list', async () => {
    const runs = await idleRuns();
    const run = runs.find((r) => r.status === 'completed')!;
    const api = new MockApi('normal', 0);
    const models = vi.spyOn(api, 'models');
    const list = runs.map((r) => (r.id === run.id ? { ...r, status: 'running' as const } : r));
    watched.runs = list;
    open(api);
    await training((t) => t.includes(run.name), 'the run');
    await vi.waitFor(() => expect(models).toHaveBeenCalledTimes(1));

    watched.runs = [...list];
    flush();
    expect(models).toHaveBeenCalledTimes(1);

    watched.runs = list.map((r) => (r.id === run.id ? { ...r, status: 'completed' } : r));
    flush();
    expect(models).toHaveBeenCalledTimes(2);
    await training((t) => t.includes('Nothing is training'), 'the empty slot');
  });
});
