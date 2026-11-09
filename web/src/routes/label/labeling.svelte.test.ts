// @vitest-environment happy-dom
import { flushSync, mount, unmount, type Component } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi, type Scenario } from '$lib/api/mock';
import type { LabelJob } from '$lib/api/types';
import { parse, router } from '$lib/router.svelte';
import { undescribedFields } from '$lib/ui/fieldA11y';
import { toast } from '$lib/ui/toast.svelte';
import LabelJobPage from '../LabelJob.svelte';
import LabelJobs from '../LabelJobs.svelte';
import DatasetDialog from './DatasetDialog.svelte';
import JobDialog from './JobDialog.svelte';

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
  vi.restoreAllMocks();
});

function open<P extends Record<string, unknown>>(comp: Component<P>, props: P) {
  c = mount(comp as Component<Record<string, unknown>>, { target, props });
  flushSync();
}

function type(el: HTMLInputElement | HTMLTextAreaElement, v: string) {
  el.value = v;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

const submitButton = () => document.querySelector<HTMLButtonElement>('dialog[open] button[type="submit"]')!;
const why = (b: HTMLElement) => document.getElementById(b.getAttribute('aria-describedby') ?? '')?.textContent;
const text = () => document.body.textContent ?? '';

describe('New labeling job', () => {
  it('describes every control, and keeps Create disabled, saying why, until it has a name and a class', () => {
    open(JobDialog, { open: true });
    expect(undescribedFields(document)).toEqual([]);
    expect(submitButton().disabled).toBe(true);
    expect(why(submitButton())).toBe('Enter a name');
    type(document.querySelector('input[id$="-name"]')!, 'street');
    expect(why(submitButton())).toBe('Add at least one class');
    // A class typed and not yet added as a chip counts: it is added on submit.
    type(document.querySelector('input[id$="-classes"]')!, 'car');
    expect(submitButton().disabled).toBe(false);
  });

  it('says what is wrong with the classes and gates Create on it', async () => {
    open(JobDialog, { open: true });
    type(document.querySelector('input[id$="-name"]')!, 'street');
    type(document.querySelector('input[id$="-classes"]')!, 'car, Car');
    submitButton().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    flushSync();
    expect(document.querySelector('[role="alert"]')?.textContent).toMatch(/listed twice/);
    expect(why(submitButton())).toBe('Fix the classes first');
  });

  it('creates the job and opens its page', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    open(JobDialog, { open: true });
    type(document.querySelector('input[id$="-name"]')!, 'street');
    type(document.querySelector('input[id$="-classes"]')!, 'car, person');
    type(document.querySelector('textarea')!, 'Label every car and person.');
    submitButton().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await vi.waitFor(() => expect(go).toHaveBeenCalled());
    expect(go.mock.calls[0]![0]).toMatch(/^\/datasets\/label\/lb_/);
    expect(toast.items[0]?.detail).toBe('street · 2 classes');
    const jobs = await hub.api.labelJobs();
    expect(jobs[0]).toMatchObject({ name: 'street', classes: ['car', 'person'], instructions: 'Label every car and person.' });
  });

  it('edits a job with boxes by renaming and adding classes, never removing one, and says why', async () => {
    const job = await hub.api.labelJob('lb_51d0');
    open(JobDialog, { open: true, job, onsave: async () => {} });
    expect(text()).toMatch(/rename a class or add one, not remove one/);
    expect(document.querySelectorAll('dialog[open] button[aria-label^="Remove"]')).toHaveLength(0);
    expect(document.querySelectorAll('dialog[open] input[aria-label^="Class"]')).toHaveLength(4);
  });
});

describe('Create dataset', () => {
  it('says what goes in and what is left out, and makes the dataset', async () => {
    const job = await hub.api.labelJob('lb_51d0');
    const oncreated = vi.fn();
    open(DatasetDialog, { open: true, job, oncreated });
    expect(undescribedFields(document)).toEqual([]);
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('fasteners-review');
    expect(text()).toMatch(/36 images with labels, \d+ boxes\./);
    expect(text()).toMatch(/About 29 images to train on and 7 images to check the model with/);
    submitButton().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await vi.waitFor(() => expect(oncreated).toHaveBeenCalled());
    expect(oncreated.mock.calls[0]![0]).toMatchObject({ name: 'fasteners-review', classes: ['bolt', 'nut', 'washer', 'screw'] });
  });

  it('leaves out the images without labels, says so, and offers a valid name', async () => {
    const job = { ...(await hub.api.labelJob('lb_9e44')), name: 'parts bench!' } as LabelJob;
    open(DatasetDialog, { open: true, job, oncreated: () => {} });
    expect(text()).toMatch(/8 images with labels, \d+ boxes\. 16 images without labels are left out\./);
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('parts-bench');
  });
});

describe('the dialogs while the job changes', () => {
  it("Create dataset keeps what was typed when the job moves on while it is open (a run's progress)", async () => {
    const job = await hub.api.labelJob('lb_51d0');
    const props = $state({ open: true, job, oncreated: () => {} });
    open(DatasetDialog, props);
    type(document.querySelector('input[id$="-name"]')!, 'my-set');
    document.querySelector<HTMLButtonElement>('dialog[open] [role="radio"]:not([aria-checked="true"])')?.click();
    flushSync();
    const share = document.querySelector('dialog[open] [role="radio"][aria-checked="true"]')?.textContent;
    props.job = { ...job, counts: { ...job.counts, labeled: job.counts.labeled - 1, pending: job.counts.pending + 1 } };
    flushSync();
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('my-set');
    expect(document.querySelector('dialog[open] [role="radio"][aria-checked="true"]')?.textContent).toBe(share);
    // Opened again, it starts afresh.
    props.open = false;
    flushSync();
    props.open = true;
    flushSync();
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('fasteners-review');
  });

  it('Edit job keeps the edits when the job changes while it is open (an upload finishing)', async () => {
    const job = await hub.api.labelJob('lb_7c21');
    const props = $state({ open: true, job, onsave: async () => {} });
    open(JobDialog, props);
    type(document.querySelector('input[id$="-name"]')!, 'bins-renamed');
    type(document.querySelector<HTMLInputElement>('dialog[open] input[aria-label="Class 1"]')!, 'hex bolt');
    props.job = { ...job, counts: { ...job.counts, images: job.counts.images + 50, pending: job.counts.pending + 50 } };
    flushSync();
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('bins-renamed');
    expect(document.querySelector<HTMLInputElement>('dialog[open] input[aria-label="Class 1"]')!.value).toBe('hex bolt');
  });
});

describe('the job page', () => {
  async function page(path: string, scenario: Scenario = 'normal') {
    hub.api = new MockApi(scenario, 0);
    router.route = parse(`#${path}`);
    open(LabelJobPage, {});
    await vi.waitFor(() => {
      flushSync();
      if (!document.querySelector('h1') || document.querySelector('[aria-busy="true"][aria-label="Loading labeling job"]')) throw new Error('loading');
    });
  }
  /** Primary buttons on screen (closed dialogs hold their own). */
  const primaries = () => [...document.querySelectorAll('.btn.primary')].filter((b) => !b.closest('dialog:not([open])'));
  const button = (name: string) => [...document.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === name);

  it('with images to label: the one primary is Label N images', async () => {
    await page('/datasets/label/lb_7c21');
    expect(button('Label 40 images')?.className).toMatch(/primary/);
    expect(primaries()).toHaveLength(1);
    expect(button('Create dataset')?.disabled).toBe(true);
    expect(text()).toMatch(/Create dataset: label an image first/);
  });

  it('while an agent labels: progress and Stop, no primary', async () => {
    await page('/datasets/label/lb_3f9a');
    expect(text()).toMatch(/\d+ of 96 images labeled/);
    expect(button('Stop')).toBeTruthy();
    expect(primaries()).toHaveLength(0);
  });

  it('with every image labeled: Create dataset is the primary, Label all again asks first', async () => {
    await page('/datasets/label/lb_51d0');
    expect(button('Create dataset')?.className).toMatch(/primary/);
    expect(primaries()).toHaveLength(1);
    expect(text()).toMatch(/Tokens and cost as reported by the agent/);
    expect(button('Label all again')).toBeTruthy();
  });

  it('after a failed run: its cause, the fix and Try again', async () => {
    await page('/datasets/label/lb_9e44');
    expect(document.querySelector('[role="alert"]')?.textContent).toMatch(/Claude Code stopped.*Not logged in/);
    expect(text()).toMatch(/Sign in to Claude Code/);
    expect(button('Try again')?.className).toMatch(/primary/);
  });

  it('on a server that cannot label: one calm state', async () => {
    await page('/datasets/label/lb_7c21', 'no-labeling');
    expect(text()).toMatch(/This server cannot label images/);
    expect(document.querySelector('.err')).toBeNull();
  });

  it('for a job that does not exist: not found, with the way back', async () => {
    await page('/datasets/label/lb_none');
    expect(text()).toMatch(/Labeling job not found/);
    expect(document.querySelector('a[href="#/datasets/label"]')).toBeTruthy();
  });
});

describe('the job list', () => {
  it('lists the jobs with their status words, and says so when there are none', async () => {
    router.route = parse('#/datasets/label');
    open(LabelJobs, {});
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelectorAll('tr[data-row]')).toHaveLength(4);
    });
    expect([...document.querySelectorAll('tr[data-row]')].map((r) => r.querySelector('.badge')?.textContent?.trim())).toEqual(['Idle', 'Labeling', 'Idle', 'Agent failed']);
    unmount(c!);
    hub.api = new MockApi('empty', 0);
    c = mount(LabelJobs, { target });
    await vi.waitFor(() => {
      flushSync();
      expect(text()).toMatch(/No labeling jobs yet/);
    });
  });
});
