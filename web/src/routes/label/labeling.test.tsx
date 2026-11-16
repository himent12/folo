// @vitest-environment happy-dom
import { deep, flush as flushSync } from 'fw';
import { createComponent } from 'fw/internal';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi, type Scenario } from '$lib/api/mock';
import type { Dataset, LabelImagesAdded, LabelJob } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { undescribedFields } from '$lib/ui/fieldA11y';
import { toast } from '$lib/ui/toast';
import { render } from '../../../framework/testing';
import LabelJobPage from '../LabelJob';
import LabelJobs from '../LabelJobs';
import AddImagesDialog from './AddImagesDialog';
import DatasetDialog, { type DatasetDialogProps } from './DatasetDialog';
import JobDialog, { type JobDialogProps } from './JobDialog';
import { whyNothingAdded, type Uploads } from './uploads';

let target: HTMLElement;
/** Port of `mount(comp, { target, props })`: the props object is read live (a `deep` one changes them). */
function mount<P extends object>(comp: (props: P) => JSX.Element, o: { target: HTMLElement; props?: P }) {
  const props = (o.props ?? {}) as P;
  return render(() => createComponent(comp, props), o.target);
}
const unmount = (c: { unmount(): void }) => c.unmount();

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

function open<P extends object>(comp: (props: P) => JSX.Element, props: P) {
  c = mount(comp, { target, props });
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

  it('focuses Name on open, and keeps what was typed when it is closed and opened again', async () => {
    const props = deep({ open: true } as JobDialogProps);
    open(JobDialog, props);
    await vi.waitFor(() => expect(document.activeElement?.id).toMatch(/-name$/));
    type(document.querySelector('input[id$="-name"]')!, 'street');
    type(document.querySelector('input[id$="-classes"]')!, 'car, person');
    type(document.querySelector('textarea')!, 'Every car.');
    props.open = false;
    flushSync();
    props.open = true;
    flushSync();
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('street');
    expect(document.querySelector<HTMLInputElement>('input[id$="-classes"]')!.value).toBe('car, person');
    expect(document.querySelector('textarea')!.value).toBe('Every car.');
  });

  it('says which pasted classes did not fit under the limit', () => {
    open(JobDialog, { open: true });
    const input = document.querySelector<HTMLInputElement>('input[id$="-classes"]')!;
    type(input, Array.from({ length: 82 }, (_, i) => `c${i + 1}`).join(', '));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    flushSync();
    expect(document.querySelectorAll('.chip')).toHaveLength(80);
    expect(document.getElementById(input.getAttribute('aria-describedby') ?? '')?.textContent).toBe('Only 80 classes fit: c81 and c82 were not added.');
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
    open(JobDialog, { open: true, job, hasBoxes: true, onsave: async () => {} });
    expect(text()).toMatch(/rename a class or add one, not remove one/);
    expect(document.querySelectorAll('dialog[open] button[aria-label^="Remove"]')).toHaveLength(0);
    expect(document.querySelectorAll('dialog[open] input[aria-label^="Class"]')).toHaveLength(4);
  });
});

describe('Add images › Server folder', () => {
  function folderTab(answer: LabelImagesAdded | null = null) {
    const folder = vi.fn(async (_path: string) => answer);
    open(AddImagesDialog, { open: true, tab: 'folder' as const, uploads: { folder } as unknown as Uploads, onbrowse: () => {} });
    const input = document.querySelector<HTMLInputElement>('dialog[open] input')!;
    const submit = () => {
      submitButton().click();
      flushSync();
    };
    return { folder, input, submit };
  }

  it('sends a path pasted with quotes and spaces around (Explorer’s "Copy as path") without them', async () => {
    const { folder, input, submit } = folderTab();
    type(input, '  "D:\\photos\\unlabeled cells"  ');
    submit();
    await vi.waitFor(() => expect(folder).toHaveBeenCalledWith('D:\\photos\\unlabeled cells'));
  });

  it('stays open and says why when the folder added nothing', async () => {
    const reason = 'no JPEG, PNG, BMP or WebP images in this folder or its subfolders (it has .tif files)';
    const { input, submit } = folderTab({ added: [], skipped: [{ file: 'D:/scans', reason }], job: {} as LabelJob });
    type(input, 'D:/scans');
    submit();
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelector('dialog[open] .nothing')?.textContent).toBe(`Nothing was added${reason.replace('no', 'No')}.`);
    });
    expect(whyNothingAdded([1, 2, 3].map((i) => ({ file: `${i}.png`, reason: i < 3 ? 'the job already has this file' : 'not a readable image' })))).toBe(
      '2 files: the job already has them; 1 file: not a readable image.',
    );
  });

  it('says a relative path is not a full one, and sends nothing', () => {
    const { folder, input, submit } = folderTab();
    type(input, "'photos'");
    submit();
    expect(folder).not.toHaveBeenCalled();
    expect(document.querySelector('dialog[open] .field.invalid')?.textContent).toMatch(/Give the full path, such as D:\/photos/);
    type(input, '/srv/photos');
    expect(document.querySelector('dialog[open] .field.invalid')).toBeNull();
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

describe('Create dataset again (studio-label F7)', () => {
  const nameInput = () => document.querySelector<HTMLInputElement>('input[id$="-name"]')!;
  const named = (...names: string[]) => names.map((name) => ({ name }) as Dataset);

  it('defaults to a name no dataset has yet, and says so when the name typed is taken', async () => {
    vi.spyOn(hub.api, 'datasets').mockResolvedValue(named('Fasteners-review', 'fasteners-review-2', 'bccd'));
    const job = await hub.api.labelJob('lb_51d0');
    open(DatasetDialog, { open: true, job, oncreated: () => {} });
    await vi.waitFor(() => {
      flushSync();
      expect(nameInput().value).toBe('fasteners-review-3');
    });
    expect(text()).not.toMatch(/exists/);
    type(nameInput(), 'bccd');
    expect(text()).toMatch(/A dataset named bccd exists\. This makes another one beside it; that one, and the runs trained on it, stay as they are\./);
  });

  it('keeps a name typed before the datasets are known', async () => {
    let answer!: (d: Dataset[]) => void;
    vi.spyOn(hub.api, 'datasets').mockReturnValue(new Promise((r) => (answer = r)));
    const job = await hub.api.labelJob('lb_51d0');
    open(DatasetDialog, { open: true, job, oncreated: () => {} });
    type(nameInput(), 'my-set');
    answer(named('fasteners-review', 'my-set'));
    await vi.waitFor(() => {
      flushSync();
      expect(text()).toMatch(/A dataset named my-set exists/);
    });
    expect(nameInput().value).toBe('my-set');
  });

  it('keeps a name edited back to the default before the datasets are known', async () => {
    let answer!: (d: Dataset[]) => void;
    vi.spyOn(hub.api, 'datasets').mockReturnValue(new Promise((r) => (answer = r)));
    open(DatasetDialog, { open: true, job: await hub.api.labelJob('lb_51d0'), oncreated: () => {} });
    type(nameInput(), 'other');
    type(nameInput(), 'fasteners-review');
    answer(named('fasteners-review'));
    await vi.waitFor(() => {
      flushSync();
      expect(text()).toMatch(/A dataset named fasteners-review exists/);
    });
    expect(nameInput().value).toBe('fasteners-review');
  });

  it('waits for the datasets before it sends a default, so a quick Create never makes the duplicate', async () => {
    let answer!: (d: Dataset[]) => void;
    vi.spyOn(hub.api, 'datasets').mockReturnValue(new Promise((r) => (answer = r)));
    const create = vi.spyOn(hub.api, 'createLabelDataset');
    const oncreated = vi.fn();
    open(DatasetDialog, { open: true, job: await hub.api.labelJob('lb_51d0'), oncreated });
    submitButton().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await Promise.resolve();
    expect(create).not.toHaveBeenCalled();
    answer(named('fasteners-review'));
    await vi.waitFor(() => expect(oncreated).toHaveBeenCalled());
    expect(create.mock.calls[0]![1]).toMatchObject({ name: 'fasteners-review-2' });
  });

  it('sends nothing for a submit still waiting when the page goes, and a stale one leaves the next opening alone', async () => {
    const answers: ((d: Dataset[]) => void)[] = [];
    vi.spyOn(hub.api, 'datasets').mockImplementation(() => new Promise((r) => answers.push(r)));
    const create = vi.spyOn(hub.api, 'createLabelDataset').mockReturnValue(new Promise(() => {}));
    const props = deep<DatasetDialogProps>({ open: true, job: await hub.api.labelJob('lb_51d0'), oncreated: () => {} });
    open(DatasetDialog, props);
    const submit = () => submitButton().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    submit();
    props.open = false;
    flushSync();
    props.open = true;
    flushSync();
    submit();
    answers[1]!(named());
    await vi.waitFor(() => expect(create).toHaveBeenCalledTimes(1));
    // The first opening's answer comes late: the second's export is still under way, and Create stays busy.
    answers[0]!(named());
    await new Promise((r) => setTimeout(r));
    flushSync();
    expect(submitButton().disabled || submitButton().getAttribute('aria-busy') === 'true').toBe(true);
    // Another opening, then the page goes before its datasets come: no request.
    props.open = false;
    flushSync();
    props.open = true;
    flushSync();
    submit();
    unmount(c!);
    c = undefined;
    answers[2]!(named());
    await new Promise((r) => setTimeout(r));
    expect(create).toHaveBeenCalledTimes(1);
  });
});

describe('the dialogs while the job changes', () => {
  it("Create dataset keeps what was typed when the job moves on while it is open (a run's progress)", async () => {
    const job = await hub.api.labelJob('lb_51d0');
    const props = deep<DatasetDialogProps>({ open: true, job, oncreated: () => {} });
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
    const props = deep<JobDialogProps>({ open: true, job, onsave: async () => {} });
    open(JobDialog, props);
    type(document.querySelector('input[id$="-name"]')!, 'bins-renamed');
    type(document.querySelector<HTMLInputElement>('dialog[open] input[aria-label="Class 1"]')!, 'hex bolt');
    props.job = { ...job, counts: { ...job.counts, images: job.counts.images + 50, pending: job.counts.pending + 50 } };
    flushSync();
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('bins-renamed');
    expect(document.querySelector<HTMLInputElement>('dialog[open] input[aria-label="Class 1"]')!.value).toBe('hex bolt');
  });

  it('Edit job does not put back a class another tab renamed when only the name is changed here', async () => {
    const job = await hub.api.labelJob('lb_7c21');
    const saved: { name: string; classes: string[] }[] = [];
    const props = deep<JobDialogProps>({ open: true, job, onsave: async (c) => void saved.push(c) });
    open(JobDialog, props);
    type(document.querySelector('input[id$="-name"]')!, 'bins-renamed');
    // Another tab renames the first class and saves: the job here follows.
    const renamed = ['hex bolt', ...job.classes.slice(1)];
    props.job = { ...job, classes: renamed };
    flushSync();
    expect(document.querySelector<HTMLInputElement>('dialog[open] input[aria-label="Class 1"]')!.value).toBe('hex bolt');
    expect(document.querySelector('dialog[open] .elsewhere')).toBeNull();
    document.querySelector<HTMLFormElement>('dialog[open] form')!.requestSubmit();
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(saved).toEqual([{ name: 'bins-renamed', classes: renamed }]);
  });

  it('Edit job says so when another tab saves a field edited here', async () => {
    const job = await hub.api.labelJob('lb_7c21');
    const props = deep<JobDialogProps>({ open: true, job, onsave: async () => {} });
    open(JobDialog, props);
    type(document.querySelector('input[id$="-name"]')!, 'bins-renamed');
    props.job = { ...job, name: 'renamed-elsewhere' };
    flushSync();
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('bins-renamed');
    expect(document.querySelector('dialog[open] .elsewhere')?.textContent).toMatch(/Changed elsewhere while you edited/);
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
    // No example text in the instructions box as if it were the job's own.
    expect(document.querySelector('textarea')?.getAttribute('placeholder')).toBe('None: the agent goes by the class names alone.');
    expect(button('Label 40 images')?.className).toMatch(/primary/);
    expect(primaries()).toHaveLength(1);
    expect(button('Create dataset')?.disabled).toBe(true);
    expect(text()).toMatch(/Create dataset: label an image first/);
  });

  it('before the first answer of a run: a line that the agent is working, and for how long', async () => {
    await page('/datasets/label/lb_7c21');
    button('Label 40 images')!.click();
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelector('.p-wait')?.textContent).toMatch(/^Claude Code is working on the first images · \d+s$/);
    });
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
    // How far review has got: in the header, and as filters.
    await vi.waitFor(() => {
      flushSync();
      expect(text()).toMatch(/22 of 36 reviewed/);
    });
    const chips = [...document.querySelectorAll('[role="radio"]')].map((r) => r.textContent?.trim());
    expect(chips).toEqual(expect.arrayContaining(['To review14', 'Reviewed22']));
  });

  it('after a failed run: its cause, the fix and Try again', async () => {
    await page('/datasets/label/lb_9e44');
    expect(document.querySelector('[role="alert"]')?.textContent).toMatch(/Claude Code stopped.*Not logged in/);
    expect(text()).toMatch(/Sign in to Claude Code/);
    // v-models-label N5: the fix reads as a sentence and its command is code.
    expect(document.querySelector('.note code')?.textContent).toBe('claude');
    expect(button('Try again')?.className).toMatch(/primary/);
  });

  it('on a server that cannot label: one calm state', async () => {
    await page('/datasets/label/lb_7c21', 'no-labeling');
    expect(text()).toMatch(/This server cannot label images/);
    expect(document.querySelector('.err')).toBeNull();
  });

  // Review round 3 (s6-shell): the stream's `job` event comes at once; a first list of images that failed after it
  // (a page of the wrong shape) left the job showing with no images, as an empty job, and no Retry.
  it('when the images fail to load after the stream told of the job: the error and Retry, not an empty job', async () => {
    let heard = false;
    const events = MockApi.prototype.labelEvents;
    vi.spyOn(MockApi.prototype, 'labelEvents').mockImplementation(function (this: MockApi, id, h) {
      return events.call(this, id, { ...h, job: (j) => ((heard = true), h.job?.(j)) });
    });
    const images = MockApi.prototype.labelImages;
    let fail = true;
    vi.spyOn(MockApi.prototype, 'labelImages').mockImplementation(async function (this: MockApi, ...a: Parameters<MockApi['labelImages']>) {
      if (!fail) return images.apply(this, a);
      await vi.waitFor(() => expect(heard).toBe(true));
      throw new ApiError(200, 'bad_response', 'The server sent an answer of an unexpected shape (no page of images).');
    });
    hub.api = new MockApi('normal', 0);
    router.route = parse('#/datasets/label/lb_7c21');
    open(LabelJobPage, {});
    await vi.waitFor(() => {
      flushSync();
      expect(text()).toMatch(/Could not load this labeling job/);
    });
    expect(heard).toBe(true);
    expect(button('Label 40 images')).toBeUndefined();

    fail = false;
    [...document.querySelectorAll<HTMLButtonElement>('button')].find((b) => /Retry|Try again/.test(b.textContent ?? ''))!.click();
    await vi.waitFor(() => {
      flushSync();
      expect(button('Label 40 images')).toBeTruthy();
    });
    expect(text()).not.toMatch(/Could not load this labeling job/);
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
    expect([...document.querySelectorAll('tr[data-row]')].map((r) => r.querySelector('.badge')?.textContent?.trim())).toEqual(['Idle', 'Labeling', 'Idle', 'Last run failed']);
    // A job whose last run failed is listed under Failed, not Idle.
    const failed = [...document.querySelectorAll<HTMLButtonElement>('[role="radiogroup"] button, [role="radio"]')].find((b) => b.textContent?.startsWith('Failed'));
    expect(failed?.textContent).toBe('Failed1');
    failed!.click();
    flushSync();
    expect([...document.querySelectorAll('tr[data-row]')].map((r) => r.querySelector('.badge')?.textContent?.trim())).toEqual(['Last run failed']);
    unmount(c!);
    hub.api = new MockApi('empty', 0);
    c = mount(LabelJobs, { target });
    await vi.waitFor(() => {
      flushSync();
      expect(text()).toMatch(/No labeling jobs yet/);
    });
  });

  // Stress d1-datasets / PR #287 review: one image was "1 images" in the row's progress label and the list's footer.
  it('counts one image as one image, in the row and under the list', async () => {
    const api = hub.api as MockApi;
    const list = api.labelJobs.bind(api);
    vi.spyOn(api, 'labelJobs').mockImplementation(async () => {
      const [j] = await list();
      return [{ ...j!, counts: { ...j!.counts, images: 1, labeled: 1, pending: 0, failed: 0 } }];
    });
    router.route = parse('#/datasets/label');
    open(LabelJobs, {});
    const foot = await vi.waitFor(() => {
      flushSync();
      const f = document.querySelector('.list-foot');
      if (!f) throw new Error('no footer yet');
      return f;
    });
    expect(foot.textContent).toContain('1 image');
    expect(foot.textContent).not.toContain('1 images');
    expect(document.querySelector('tr[data-row] [role="progressbar"]')?.getAttribute('aria-label')).toBe('1 of 1 image labeled');
  });
});

describe('the job list: the palette’s New labeling job', () => {
  const dialogOpen = () => [...document.querySelectorAll<HTMLDialogElement>('dialog')].some((d) => d.open && d.textContent?.includes('New labeling job'));
  it('opens the dialog on this page too, and again after it was closed', async () => {
    // `?new=1` was read once, when the page mounted: on the page already (the app keys pages by path, not query) the
    // palette changed the URL and nothing opened; after a close the URL kept `?new=1`, so the next ask did nothing.
    history.replaceState(null, '', '#/datasets/label');
    router.route = parse('#/datasets/label');
    open(LabelJobs, {});
    expect(dialogOpen()).toBe(false);
    router.go('/datasets/label?new=1');
    flushSync();
    expect(dialogOpen()).toBe(true);
    document.querySelector<HTMLButtonElement>('dialog[open] button[aria-label="Close"]')!.click();
    flushSync();
    expect(dialogOpen()).toBe(false);
    router.go('/datasets/label?new=1');
    flushSync();
    expect(dialogOpen()).toBe(true);
  });
});
