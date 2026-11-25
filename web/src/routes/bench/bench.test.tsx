// @vitest-environment happy-dom
// The benchmark screens' behaviour against the in-browser mock: what the dialogs ask for and refuse, the plan that is
// fixed once there are entries (and the edits that are not lost), and the leaderboard's Compare selection.
import { createRoot, flush as flushSync, signal, type Component } from 'fw';
import { render, type Rendered } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { ApiError } from '$lib/api/client';
import { MockApi } from '$lib/api/mock';
import type { Bench, BenchDesign, Model } from '$lib/api/types';
import { BENCH_FEATURE } from '$lib/bench/rules';
import { parse, router } from '$lib/router';
import { undescribedFields } from '$lib/ui/fieldA11y';
import { toast } from '$lib/ui/toast';
import BenchmarkPage from '../Benchmark';
import Benchmarks from '../Benchmarks';
import { AgentChoice } from '../label/agentChoice';
import AddEntryDialog from './AddEntryDialog';
import NewBenchDialog from './NewBenchDialog';
import { FROZEN } from './words';

let target: HTMLElement;
let c: Rendered | undefined;
let stopRoot: (() => void) | undefined;

/** Port of `unmount(c)`. */
const unmount = (r: Rendered) => r.unmount();

// (Legacy only: Svelte's select binding reads the chosen option with `:checked`, which happy-dom does not match on
// <option>. fw reads `options[selectedIndex]`; the shim is kept so both suites run under the same DOM.)
const qs = HTMLSelectElement.prototype.querySelector;
HTMLSelectElement.prototype.querySelector = function (this: HTMLSelectElement, sel: string) {
  return sel === ':checked' ? (this.options[this.selectedIndex] ?? null) : qs.call(this, sel);
} as typeof qs;

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
  stopRoot?.();
  stopRoot = undefined;
  target.remove();
  vi.restoreAllMocks();
});

/** Port of `mount(comp, { target, props })` + `flushSync()`. */
function open<P extends object>(comp: Component<P>, props: P) {
  const Comp = comp as Component<Record<string, unknown>>;
  c = render(() => <Comp {...(props as Record<string, unknown>)} />, target);
  flushSync();
}

/** Lets the mock's answers (latency 0), the timers they start and the effects after those land: a timer tick a round
 *  (`setImmediate` rounds can all pass before a 1 ms timer is due, which failed in CI). */
async function settle(rounds = 12) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
}

/** Waits for what lands after a real timer (a picked file is read after a frame), polling without one. */
async function until<T>(get: () => T | null | undefined | false, what: string): Promise<T> {
  const end = performance.now() + 3000;
  for (;;) {
    flushSync();
    const v = get();
    if (v) return v;
    if (performance.now() > end) throw new Error(`still waiting for ${what}`);
    await new Promise((r) => setImmediate(r));
  }
}

function type(el: HTMLInputElement | HTMLTextAreaElement, v: string) {
  el.value = v;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

const submit = () => document.querySelector<HTMLButtonElement>('dialog[open] button[type="submit"]')!;
const send = () => submit().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
const why = (b: HTMLElement) => document.getElementById(b.getAttribute('aria-describedby') ?? '')?.textContent;
const text = () => document.body.textContent ?? '';
const button = (re: RegExp, root: ParentNode = document) => [...root.querySelectorAll<HTMLButtonElement>('button')].find((b) => re.test(b.textContent?.trim() ?? ''));

describe('New benchmark', () => {
  it('describes every control, gates Create until it has a name, warns of a training split and refuses a bad image limit', async () => {
    open(NewBenchDialog, { open: true });
    await settle();
    expect(undescribedFields(document)).toEqual([]);
    expect(submit().disabled).toBe(true);
    expect(why(submit())).toBeTruthy();
    type(document.querySelector('input[id$="-name"]')!, 'Pond fish');
    expect(submit().disabled).toBe(false);
    // A validation split by default; the training split is warned of.
    const splits = [...document.querySelectorAll<HTMLElement>('[role="radiogroup"][aria-label="Split"] [role="radio"]')];
    expect(splits[0]!.getAttribute('aria-checked')).toBe('true');
    expect(document.querySelector('.note.warn')).toBeNull();
    splits.at(-1)!.click();
    flushSync();
    expect(document.querySelector('.note.warn')).not.toBeNull();
    // An image limit outside 1–5000 is refused at the field as soon as it is typed (no "It takes all …" for it), and
    // holds Create.
    type(document.querySelector('input[id$="-max"]')!, '-5');
    expect(document.querySelector('[role="alert"]')?.textContent).toMatch(/5,000/);
    expect(document.body.textContent).not.toMatch(/It takes all/);
    expect(why(submit())).toBeTruthy();
    send();
    flushSync();
    expect(document.querySelector('[role="alert"]')?.textContent).toMatch(/5,000/);
  });

  it("says how many of a labeling job's labeled images nobody reviewed before they become the truth (sweep 3)", async () => {
    const [job] = (await hub.api.labelJobs()).filter((j) => j.counts.labeled > 0);
    const page = await hub.api.labelImages(job!.id, { status: 'labeled', limit: 1000 });
    const labeled = page.images.map((im, i) => ({ ...im, reviewed: i > 0 }));
    hub.api.labelImages = async () => ({ total: labeled.length, images: labeled });
    open(NewBenchDialog, { open: true });
    await settle();
    [...document.querySelectorAll<HTMLElement>('[role="radio"]')].find((r) => r.textContent?.includes('A labeling job'))!.click();
    await settle();
    const select = await until(() => document.querySelector<HTMLSelectElement>('dialog[open] select:not([id$="-ds"])'), 'the job picker');
    select.value = job!.id;
    select.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    const note = await until(() => document.querySelector('.note.warn[data-unreviewed]'), 'the unreviewed note');
    expect(note.textContent).toContain(`1 of its ${labeled.length} labeled images has not been reviewed`);
    expect(text()).not.toContain('as reviewed');
    // Every image reviewed: no warning, the note on training instead.
    labeled.forEach((im) => (im.reviewed = true));
    [...document.querySelectorAll<HTMLElement>('[role="radio"]')].find((r) => r.textContent?.includes('A dataset split'))!.click();
    await settle();
    [...document.querySelectorAll<HTMLElement>('[role="radio"]')].find((r) => r.textContent?.includes('A labeling job'))!.click();
    await settle();
    expect(document.querySelector('.note.warn')).toBeNull();
    expect(text()).toContain('a model trained on a dataset made from this job has seen them');
  });

  it('shows a job’s unreviewed count only for that job, and reads again when the images move while it pages', async () => {
    const jobs = (await hub.api.labelJobs()).filter((j) => j.counts.labeled > 0);
    const [a, b] = [jobs[0]!, { ...jobs[0]!, id: 'lb_other', name: 'other job' }];
    hub.api.labelJobs = async () => [a, b];
    const img = (await hub.api.labelImages(a.id, { status: 'labeled', limit: 1 })).images[0]!;
    const page = (total: number, images: number, reviewed: boolean) => ({ total, images: Array.from({ length: images }, (_, i) => ({ ...img, id: `${i}`, reviewed })) });
    // Job A: 1,001 labeled images, the last one unreviewed. Its first read is torn (an image went between the two pages:
    // the second page says 1,000 in all), the second read is whole. Job B: all reviewed, and its answer is held.
    let aReads = 0;
    let answerB: () => void = () => {};
    hub.api.labelImages = async (id, q) => {
      if (id === b.id) return new Promise((ok) => (answerB = () => ok(page(3, 3, true))));
      aReads++;
      if (!q?.offset) return page(1001, 1000, true);
      return aReads === 2 ? page(1000, 0, true) : page(1001, 1, false);
    };
    open(NewBenchDialog, { open: true });
    await settle();
    [...document.querySelectorAll<HTMLElement>('[role="radio"]')].find((r) => r.textContent?.includes('A labeling job'))!.click();
    await settle();
    const select = await until(() => document.querySelector<HTMLSelectElement>('dialog[open] select:not([id$="-ds"])'), 'the job picker');
    select.value = a.id;
    select.dispatchEvent(new Event('change', { bubbles: true }));
    const note = await until(() => document.querySelector('.note.warn[data-unreviewed]'), 'the unreviewed note');
    // Out of the labeled images of the same read, not of the job's older list.
    expect(note.textContent).toMatch(/^1 of its 1,001 labeled images/);
    expect(aReads).toBe(4);
    // Job B picked: A's count is not shown for it while B's answer is out, nor after it says every image is reviewed.
    select.value = b.id;
    select.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect(document.querySelector('.note.warn')).toBeNull();
    answerB();
    await settle();
    expect(document.querySelector('.note.warn')).toBeNull();
  });

  it('creates it with the default plan and opens its page', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    open(NewBenchDialog, { open: true });
    await settle();
    type(document.querySelector('input[id$="-name"]')!, 'Fasteners again');
    send();
    await settle();
    expect(go).toHaveBeenCalledWith(expect.stringMatching(/^\/benchmarks\/bn_[0-9a-f]{12}$/));
    const made = (await hub.api.benches())[0]!;
    expect(made.name).toBe('Fasteners again');
    expect(made.plan.conditions).toHaveLength((await hub.api.benchCatalog()).default_plan.conditions.length);
  });
});

describe('Add entry', () => {
  async function dialog(kind: 'model' | 'agent' | 'external', benchId = 'bn_3c90', only?: (m: Model) => boolean) {
    const bench: Bench = await hub.api.bench(benchId);
    const models = (await hub.api.models()).filter((m) => !only || only(m));
    const agents = await hub.api.labelAgents();
    const entries = await hub.api.benchEntries(benchId);
    const catalog = await hub.api.benchCatalog();
    let choice!: AgentChoice;
    stopRoot = createRoot((dispose) => {
      choice = new AgentChoice(() => agents);
      return dispose;
    });
    const onstarted = vi.fn();
    open(AddEntryDialog, { open: true, kind, bench, catalog, entries, models, modelsError: undefined, agents, agentsError: undefined, retryAgents: () => {}, choice, onstarted });
    await settle();
    return { onstarted };
  }

  /** Picks a file in the dialog's file input and waits for its check. */
  async function pickFile(body: string, name: string) {
    const input = document.querySelector<HTMLInputElement>('input[type="file"]')!;
    Object.defineProperty(input, 'files', { configurable: true, value: [new File([body], name, { type: 'application/json' })] });
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return until(() => document.querySelector('.f-state.ok, .f-state.bad'), 'the file checked');
  }

  it('shows the classes a model lacks, and a model with none of them cannot be picked', async () => {
    await dialog('model');
    const sel = document.querySelector<HTMLSelectElement>('select[id$="-model"]')!;
    // (happy-dom sets a select's value only among its direct options: pick the option in its group.)
    sel.querySelector<HTMLOptionElement>('option[value="m_mini_n"]')!.selected = true;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect([...document.querySelectorAll('.chip.miss')].map((x) => x.textContent?.replace(/,.*$/, '').trim())).toEqual(['washer', 'screw', 'spring', 'rivet', 'pin', 'bearing']);
    expect(sel.querySelector<HTMLOptionElement>('option[value="m_coco_s"]')?.disabled).toBe(true);
  });

  it('says when no model knows a class of the benchmark, and how to get one', async () => {
    await dialog('model', 'bn_3c90', (m) => m.id === 'm_coco_s');
    const help = document.querySelector('.no-model')!;
    expect(help.querySelector('.nm-title')?.textContent).toMatch(/^No model here knows bolt, nut, washer, screw or \d+ more$/);
    expect(help.querySelector<HTMLAnchorElement>('a')?.getAttribute('href')).toMatch(/^#\/runs\/new\?dataset=|^#\/datasets\/label\//);
    const add = submit();
    expect(add.disabled).toBe(true);
    expect(why(add)).toBe('No model knows these classes');
    [...help.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Use an agent')!.click();
    flushSync();
    expect(document.querySelector('.no-model')).toBeNull();
  });

  it('says an image count out of range as soon as it is typed', async () => {
    await dialog('model');
    type(document.querySelector('input[id$="-mimages"]')!, '99999');
    expect(document.querySelector('.field.invalid [role="alert"]')?.textContent).toMatch(/A whole number from 1 to/);
    expect(why(submit())).toBe('Fix the fields first');
    type(document.querySelector('input[id$="-mimages"]')!, '');
    expect(document.querySelector('.field.invalid')).toBeNull();
  });

  it('starts an agent entry with the defaults', async () => {
    const { onstarted } = await dialog('agent');
    expect(document.querySelector<HTMLInputElement>('input[id$="-aimages"]')!.value).toBe('48');
    send();
    await settle();
    expect(onstarted).toHaveBeenCalledWith(expect.objectContaining({ images: 48, conditions: false, scored: false }));
  });

  it('checks an external predictions file before it is sent, and names the entry after it', async () => {
    const { onstarted } = await dialog('external', 'bn_9f03');
    const bad = await pickFile('{"predictions": [{"image": "1", "boxes": [{"class": 9, "x1": 0, "y1": 0, "x2": 4, "y2": 4}]}]}', 'det-predictions.json');
    expect(bad.className).toContain('bad');
    expect(bad.textContent).toMatch(/^Record 1: box 1/);
    expect(why(submit())).toBeTruthy();
    expect(onstarted).not.toHaveBeenCalled();
    const images = (await hub.api.benchImages('bn_9f03', { limit: 1000 })).images;
    const ok = await pickFile(JSON.stringify(images.map((i) => ({ image: i.id, boxes: i.boxes.map((b) => ({ ...b, score: 0.8 })) }))), 'det-predictions.json');
    expect(ok.className).toContain('ok');
    // A name is suggested from the file's.
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('det');
    send();
    await settle();
    expect(onstarted).toHaveBeenCalledWith(expect.objectContaining({ status: 'done', name: 'det' }));
  });

  /** A file whose request would be large: every image under every pass, enough boxes to be over 8 MiB as JSON. */
  async function pickLarge() {
    const b = await hub.api.bench('bn_9f03');
    const images = (await hub.api.benchImages('bn_9f03', { limit: 1000 })).images;
    const pass = [{ c: null as string | null, s: null as number | null }, ...b.plan.conditions.flatMap((c) => c.severities.map((s) => ({ c: c.condition as string | null, s: s as number | null })))];
    const box = { class: 0, score: 0.5, x1: 1, y1: 2, x2: 30, y2: 40 };
    const records = pass.flatMap((p) => images.map((i) => ({ image: i.id, ...(p.c ? { condition: p.c, severity: p.s } : {}), boxes: Array.from({ length: 12 }, () => box) })));
    expect((await pickFile(JSON.stringify(records), 'big.json')).className).toContain('ok');
    return { records, passes: pass.length };
  }

  it('sends a large file as an open entry, a pass at a time, then finishes it', async () => {
    const { onstarted } = await dialog('external', 'bn_9f03');
    const { records, passes } = await pickLarge();
    const parts = vi.spyOn(hub.api, 'sendBenchPredictions');
    send();
    await until(() => onstarted.mock.calls.length > 0, 'the entry started');
    expect(parts).toHaveBeenCalledTimes(passes);
    expect(parts.mock.calls.reduce((n, c) => n + c[2].length, 0)).toBe(records.length);
    expect(onstarted).toHaveBeenCalledWith(expect.objectContaining({ status: 'done', name: 'big', conditions: true }));
  });

  it('leaves nothing on the leaderboard when a part fails, and shows the server’s message', async () => {
    const { onstarted } = await dialog('external', 'bn_9f03');
    await pickLarge();
    let n = 0;
    const real = hub.api.sendBenchPredictions.bind(hub.api);
    vi.spyOn(hub.api, 'sendBenchPredictions').mockImplementation(async (...a) => {
      if (++n === 3) throw new ApiError(400, 'invalid_body', 'Record 7: box 1: "class" must be a class index from 0 to 5.');
      return real(...a);
    });
    const stop = vi.spyOn(hub.api, 'stopBenchEntry');
    send();
    await until(() => /Record 7: box 1/.test(text()), 'the server’s message');
    await settle();
    expect(onstarted).not.toHaveBeenCalled();
    expect(stop).toHaveBeenCalledTimes(1);
    expect((await hub.api.benchEntries('bn_9f03')).length).toBe(0);
    expect((await hub.api.bench('bn_9f03')).active).toBe(false);
  });

  /** Triggers a file selection without waiting for its async check to complete. */
  function selectFile(f: File) {
    const input = document.querySelector<HTMLInputElement>('input[type="file"]')!;
    Object.defineProperty(input, 'files', { configurable: true, value: [f] });
    input.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
  }

  /** Creates a File whose `.text()` settles only when the returned resolver/rejecter is called. */
  function deferredFile(name: string) {
    let resolve!: (text: string) => void;
    let reject!: (err: unknown) => void;
    const promise = new Promise<string>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    let reading = false;
    const f = new File(['{}'], name, { type: 'application/json' });
    vi.spyOn(f, 'text').mockImplementation(() => {
      reading = true;
      return promise;
    });
    return { file: f, resolve, reject, isReading: () => reading };
  }

  it('ignores slow out-of-order file reads and keeps reading state bound to the latest selection', async () => {
    await dialog('external', 'bn_9f03');
    const images = (await hub.api.benchImages('bn_9f03', { limit: 1000 })).images;
    const valid = JSON.stringify(images.map((i) => ({ image: i.id, boxes: i.boxes.map((b) => ({ ...b, score: 0.8 })) })));
    const invalid = '{"predictions": [{"image": "1", "boxes": [{"class": 9, "x1": 0, "y1": 0, "x2": 4, "y2": 4}]}]}';

    // Case 1: slow valid file A is superseded by fast invalid file B; when A resolves later, B's error stays.
    const slowA = deferredFile('baseline-predictions.json');
    selectFile(slowA.file);
    await until(() => slowA.isReading() && document.querySelector('.f-state.busy'), 'reading slow file A');

    const bad = await pickFile(invalid, 'broken-predictions.json');
    expect(bad.className).toContain('bad');
    expect(document.querySelector('.f-name')?.textContent).toContain('broken-predictions.json');

    slowA.resolve(valid);
    await settle();
    expect(document.querySelector('.f-state.bad')?.textContent).toMatch(/^Record 1: box 1/);
    expect(document.querySelector('.f-state.ok')).toBeNull();
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('');

    // Case 2: slow file C rejects while file D is still reading; C's catch/finally must NOT clear D's reading state.
    const slowC = deferredFile('old-predictions.json');
    const slowD = deferredFile('candidate-predictions.json');
    selectFile(slowC.file);
    await until(() => slowC.isReading() && document.querySelector('.f-state.busy'), 'reading slow file C');
    selectFile(slowD.file);
    await until(() => slowD.isReading(), 'reading slow file D');
    slowC.reject(new Error('read aborted'));
    await settle();
    expect(document.querySelector('.f-state.busy')).not.toBeNull();
    expect(document.querySelector('.f-state.bad')).toBeNull();
    expect(why(submit())).toBe('Reading the file');

    slowD.resolve(valid);
    const ok = await until(() => document.querySelector('.f-state.ok'), 'file D validated');
    expect(ok.className).toContain('ok');
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('candidate');
  });

  it('discards an in-flight file read when the dialog closes and reopens', async () => {
    const bench: Bench = await hub.api.bench('bn_9f03');
    const models = await hub.api.models();
    const agents = await hub.api.labelAgents();
    const entries = await hub.api.benchEntries('bn_9f03');
    const catalog = await hub.api.benchCatalog();
    const images = (await hub.api.benchImages('bn_9f03', { limit: 1000 })).images;
    const valid = JSON.stringify(images.map((i) => ({ image: i.id, boxes: i.boxes.map((b) => ({ ...b, score: 0.8 })) })));
    let choice!: AgentChoice;
    stopRoot = createRoot((dispose) => {
      choice = new AgentChoice(() => agents);
      return dispose;
    });
    const isOpen = signal(true);
    const kind = signal<'model' | 'agent' | 'external'>('external');
    c = render(
      () => (
        <AddEntryDialog
          bind:open={isOpen}
          bind:kind={kind}
          bench={bench}
          catalog={catalog}
          entries={entries}
          models={models}
          modelsError={undefined}
          agents={agents}
          agentsError={undefined}
          retryAgents={() => {}}
          choice={choice}
          onstarted={() => {}}
        />
      ),
      target,
    );
    flushSync();
    await settle();

    const slow = deferredFile('stale-predictions.json');
    selectFile(slow.file);
    await until(() => slow.isReading() && document.querySelector('.f-state.busy'), 'reading stale file');

    isOpen.set(false);
    flushSync();
    isOpen.set(true);
    flushSync();
    expect(document.querySelector('.f-state.busy')).toBeNull();

    slow.resolve(valid);
    await settle();
    expect(document.querySelector('.f-state.ok')).toBeNull();
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('');
  });

  it('updates untouched auto-suggested names on file replacement, but preserves manually edited names', async () => {
    await dialog('external', 'bn_9f03');
    const images = (await hub.api.benchImages('bn_9f03', { limit: 1000 })).images;
    const valid = JSON.stringify(images.map((i) => ({ image: i.id, boxes: i.boxes.map((b) => ({ ...b, score: 0.8 })) })));
    const nameInput = () => document.querySelector<HTMLInputElement>('input[id$="-name"]')!;

    await pickFile(valid, 'baseline-predictions.json');
    expect(nameInput().value).toBe('baseline');

    // Untouched auto-suggested name updates when choosing another file.
    await pickFile(valid, 'candidate-predictions.json');
    expect(nameInput().value).toBe('candidate');

    // Manually edited name is preserved across subsequent file replacements.
    type(nameInput(), 'my-custom-detector');
    await pickFile(valid, 'other-predictions.json');
    expect(nameInput().value).toBe('my-custom-detector');

    // Clearing the manual name back to empty re-enables auto-suggestion on the next file pick.
    type(nameInput(), '');
    await pickFile(valid, 'recovered-predictions.json');
    expect(nameInput().value).toBe('recovered');
  });

  it('asks before closing mid-send, then stops the entry', async () => {
    await dialog('external', 'bn_9f03');
    await pickLarge();
    // The second part never answers: the send is in flight when the person closes the dialog.
    let n = 0;
    const real = hub.api.sendBenchPredictions.bind(hub.api);
    vi.spyOn(hub.api, 'sendBenchPredictions').mockImplementation((...a) => (++n === 2 ? new Promise(() => {}) : real(...a)));
    const stop = vi.spyOn(hub.api, 'stopBenchEntry');
    send();
    await until(() => document.querySelector('dialog[open] .f-send'), 'the send');
    button(/^Cancel$/, document.querySelector('dialog[open]')!)!.click();
    flushSync();
    // Asked first: nothing is stopped until the person says so.
    expect(stop).not.toHaveBeenCalled();
    button(/^Stop and discard$/)!.click();
    await settle();
    expect(stop).toHaveBeenCalledTimes(1);
    expect((await hub.api.benchEntries('bn_9f03')).length).toBe(0);
  });
});

describe('the benchmark page', () => {
  async function page(path: string) {
    router.route = parse(`#${path}`);
    open(BenchmarkPage, {});
    await settle(20);
  }

  it('keeps "made … ago" current while the page stays open (sweep 3)', async () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'Date'] });
    try {
      const made = await hub.api.benches().then((l) => l.find((b) => b.id === 'bn_3c90')!.created);
      vi.setSystemTime(Date.parse(made) + 10_000);
      await page('/benchmarks/bn_3c90');
      expect(text()).toContain('made just now');
      vi.setSystemTime(Date.parse(made) + 5 * 60_000);
      vi.advanceTimersByTime(30_000);
      flushSync();
      expect(text()).toContain('made 6m ago');
    } finally {
      vi.useRealTimers();
    }
  });

  it('shows a plan with entries as fixed, with no controls to change it', async () => {
    await page('/benchmarks/bn_7a41?tab=conditions');
    expect(text()).toContain(FROZEN);
    const tab = document.getElementById('bn-tab-panel')!;
    expect(tab.querySelectorAll('input[role="switch"]')).toHaveLength(0);
    expect(tab.querySelectorAll('button.sev')).toHaveLength(0);
    expect(tab.querySelectorAll('.sev.static.has').length).toBeGreaterThan(0);
    expect(button(/^Design with an agent/)).toBeUndefined();
  });

  it('edits the plan of a benchmark with no entries, holds Add entry until it is saved, and asks before leaving', async () => {
    await page('/benchmarks/bn_9f03?tab=conditions');
    const sev = [...document.querySelectorAll<HTMLButtonElement>('button.sev')].find((b) => b.getAttribute('aria-pressed') === 'false')!;
    sev.click();
    flushSync();
    const add = button(/Add entry/, document.querySelector('header')!)!;
    expect(add.disabled).toBe(true);
    expect(why(add)).toBeTruthy();
    // Leaving asks first.
    const go = vi.spyOn(history, 'pushState');
    router.go('/benchmarks');
    flushSync();
    expect(document.querySelector('dialog[open]')).not.toBeNull();
    expect(go).not.toHaveBeenCalled();
  });

  it('shows the weight the plan has again once its edits are discarded', async () => {
    await page('/benchmarks/bn_9f03?tab=conditions');
    const weight = () => document.querySelector<HTMLInputElement>('input[aria-label="Weight of Haze"]')!;
    expect(weight().value).toBe('1');
    weight().value = '3';
    weight().dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(text()).toMatch(/Unsaved changes/);
    [...document.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Discard')!.click();
    flushSync();
    expect(text()).not.toMatch(/Unsaved changes/);
    expect(weight().value).toBe('1');
  });

  it('does not save a weight field left empty with the last number typed in it', async () => {
    await page('/benchmarks/bn_9f03?tab=conditions');
    const weight = () => document.querySelector<HTMLInputElement>('input[aria-label="Weight of Haze"]')!;
    const type = (v: string) => {
      weight().value = v;
      weight().dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
    };
    const save = () => [...document.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Save plan')!;
    type('3');
    expect(save().disabled).toBe(false);
    type('');
    // The draft still holds 3: Save would store a weight the field does not show.
    expect(weight().getAttribute('aria-invalid')).toBe('true');
    expect(text()).toMatch(/Haze: the weight is empty/);
    expect(save().disabled).toBe(true);
    type('2');
    expect(weight().getAttribute('aria-invalid')).toBeNull();
    expect(save().disabled).toBe(false);
  });

  it('does not save a weight field left empty from the leave dialog either, also after a change of tab', async () => {
    await page('/benchmarks/bn_9f03?tab=conditions');
    const weight = () => document.querySelector<HTMLInputElement>('input[aria-label="Weight of Haze"]')!;
    const type = (v: string) => {
      weight().value = v;
      weight().dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
    };
    type('3');
    type('');
    // The field stays empty across a change of tab: the page holds it.
    router.go('/benchmarks/bn_9f03', { replace: true });
    await settle();
    router.go('/benchmarks/bn_9f03?tab=conditions', { replace: true });
    await settle();
    expect(weight().value).toBe('');
    expect(weight().getAttribute('aria-invalid')).toBe('true');
    const put = vi.spyOn(hub.api, 'updateBench');
    router.go('/benchmarks');
    flushSync();
    const dialog = document.querySelector('dialog[open]')!;
    expect(dialog.textContent).toMatch(/Haze: the weight is empty/);
    const save = [...dialog.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Save plan')!;
    expect(save.disabled).toBe(true);
    save.click();
    await settle();
    expect(put).not.toHaveBeenCalled();
  });

  it('says so when another tab saves the plan under unsaved edits, and keeps them', async () => {
    await page('/benchmarks/bn_9f03?tab=conditions');
    const weight = (n: string) => document.querySelector<HTMLInputElement>(`input[aria-label="Weight of ${n}"]`)!;
    weight('Haze').value = '3';
    weight('Haze').dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(text()).not.toMatch(/Changed elsewhere/);
    // Another tab saves a change of another condition.
    const b = await hub.api.bench('bn_9f03');
    const other = b.plan.conditions.find((c) => c.condition !== 'haze')!;
    await hub.api.updateBench('bn_9f03', {
      plan: { ...b.plan, conditions: b.plan.conditions.map((c) => (c === other ? { ...c, weight: 4 } : c)) },
    });
    await settle();
    expect(text()).toMatch(/Changed elsewhere while you edited/);
    expect(weight('Haze').value).toBe('3');
    // Discard shows the other tab's plan, and the note goes.
    [...document.querySelectorAll<HTMLButtonElement>('button')].find((x) => x.textContent?.trim() === 'Discard')!.click();
    flushSync();
    expect(text()).not.toMatch(/Changed elsewhere/);
    expect(weight('Haze').value).toBe('1');
  });

  it('says so in the leave dialog when another tab saved the plan while another view was open', async () => {
    await page('/benchmarks/bn_9f03?tab=conditions');
    const weight = () => document.querySelector<HTMLInputElement>('input[aria-label="Weight of Haze"]')!;
    weight().value = '3';
    weight().dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    // To the leaderboard: the plan's own note is not on screen.
    router.go('/benchmarks/bn_9f03', { replace: true });
    await settle();
    const b = await hub.api.bench('bn_9f03');
    await hub.api.updateBench('bn_9f03', {
      plan: { ...b.plan, conditions: b.plan.conditions.map((c) => (c.condition !== 'haze' ? { ...c, weight: 4 } : c)) },
    });
    await settle();
    router.go('/benchmarks');
    flushSync();
    const dialog = document.querySelector('dialog[open]')!;
    expect(dialog.textContent).toMatch(/also saved from another tab or window while you edited/);
  });

  it('says so when another tab saves the plan while this one saves, and does not say Plan saved', async () => {
    await page('/benchmarks/bn_9f03?tab=conditions');
    const weight = () => document.querySelector<HTMLInputElement>('input[aria-label="Weight of Haze"]')!;
    weight().value = '3';
    weight().dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    const real = hub.api.updateBench.bind(hub.api);
    let answer!: () => void;
    const held = new Promise<void>((r) => (answer = r));
    // This tab's save lands on the server; its answer comes late.
    vi.spyOn(hub.api, 'updateBench').mockImplementationOnce(async (id, req) => {
      const out = await real(id, req);
      await held;
      return out;
    });
    const success = vi.spyOn(toast, 'success');
    [...document.querySelectorAll<HTMLButtonElement>('button')].find((x) => x.textContent?.trim() === 'Save plan')!.click();
    await settle();
    // Another tab saves after it, while this one still waits.
    const b = await hub.api.bench('bn_9f03');
    await real('bn_9f03', {
      plan: { ...b.plan, conditions: b.plan.conditions.map((c) => (c.condition !== 'haze' ? { ...c, weight: 4 } : { ...c, weight: 1 })) },
    });
    await settle();
    answer();
    await settle();
    expect(text()).toMatch(/Changed elsewhere while you edited/);
    expect(weight().value).toBe('3');
    expect(success).not.toHaveBeenCalled();
  });

  it('keeps an emptied weight empty, with its error, across edits of other conditions', async () => {
    await page('/benchmarks/bn_9f03?tab=conditions');
    const weights = () => [...document.querySelectorAll<HTMLInputElement>('input[aria-label^="Weight of "]')];
    const haze = () => weights().find((w) => w.getAttribute('aria-label') === 'Weight of Haze')!;
    const other = () => weights().find((w) => w.getAttribute('aria-label') !== 'Weight of Haze')!;
    const type = (el: HTMLInputElement, v: string) => {
      el.value = v;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
    };
    type(haze(), '3');
    type(haze(), '');
    type(other(), '2');
    expect(haze().value).toBe('');
    expect(haze().getAttribute('aria-invalid')).toBe('true');
    expect(text()).toMatch(/Haze: the weight is empty/);
    const save = [...document.querySelectorAll<HTMLButtonElement>('button')].find((x) => x.textContent?.trim() === 'Save plan')!;
    expect(save.disabled).toBe(true);
    // Discard starts afresh.
    [...document.querySelectorAll<HTMLButtonElement>('button')].find((x) => x.textContent?.trim() === 'Discard')!.click();
    flushSync();
    expect(haze().value).toBe('1');
    expect(haze().getAttribute('aria-invalid')).toBeNull();
  });

  it('saves a weight the server rounds from the leave dialog and leaves: its own save, not another tab’s', async () => {
    await page('/benchmarks/bn_9f03?tab=conditions');
    const weight = () => document.querySelector<HTMLInputElement>('input[aria-label="Weight of Haze"]')!;
    weight().value = '2.00000001';
    weight().dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    const real = hub.api.updateBench.bind(hub.api);
    // The server keeps a weight as a 32-bit float (folo-api `BenchPlanCondition::weight`): 2.00000001 comes back as 2.
    vi.spyOn(hub.api, 'updateBench').mockImplementation((id, req) =>
      real(id, req.plan ? { ...req, plan: { ...req.plan, conditions: req.plan.conditions.map((c) => ({ ...c, weight: Math.fround(c.weight) })) } } : req),
    );
    const success = vi.spyOn(toast, 'success');
    const info = vi.spyOn(toast, 'info');
    const go = vi.spyOn(history, 'pushState');
    router.go('/benchmarks');
    flushSync();
    const dialog = document.querySelector('dialog[open]')!;
    expect(dialog.textContent).not.toMatch(/another tab/);
    [...dialog.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Save plan')!.click();
    await settle();
    expect(info).not.toHaveBeenCalled();
    expect(success).toHaveBeenCalled();
    expect(go).toHaveBeenCalled();
    expect((await hub.api.bench('bn_9f03')).plan.conditions.find((c) => c.condition === 'haze')!.weight).toBe(2);
  });

  it('takes a proposal used in place of the plan with its own weights, not a field emptied in the old one', async () => {
    await page('/benchmarks/bn_9f03?tab=conditions');
    const weight = () => document.querySelector<HTMLInputElement>('input[aria-label="Weight of Haze"]')!;
    const type = (v: string) => {
      weight().value = v;
      weight().dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
    };
    // Emptied, the plan is still as loaded (no edits): Design with an agent stays on offer.
    type('');
    expect(weight().getAttribute('aria-invalid')).toBe('true');
    const b = await hub.api.bench('bn_9f03');
    const plan = { ...b.plan, conditions: b.plan.conditions.map((c) => ({ ...c, weight: c.condition === 'haze' ? 2 : c.weight })), rationale: 'Haze matters.', designed_by: { agent: 'claude' as const, model: null } };
    vi.spyOn(hub.api, 'designBench').mockResolvedValue({ plan, usage: {} } as BenchDesign);
    button(/^Design with an agent/)!.click();
    await settle();
    const ask = await until(() => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((x) => /^Ask /.test(x.textContent?.trim() ?? '') && !x.disabled), 'Ask');
    ask.click();
    await settle();
    const use = await until(() => [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((x) => x.textContent?.trim() === 'Use this plan'), 'Use this plan');
    use.click();
    await settle();
    expect(weight().value).toBe('2');
    expect(weight().getAttribute('aria-invalid')).toBeNull();
    expect(text()).not.toMatch(/Haze: the weight is empty/);
  });

  it('keeps the preview on its image when the benchmark changes under it (a `bench` event)', async () => {
    // happy-dom has no canvas to draw the mock's images: any URL does.
    const url = vi.spyOn(hub.api, 'benchImageUrl').mockImplementation(async (_b, im, o) => `blob:${im}/${o?.condition ?? ''}`);
    await page('/benchmarks/bn_9f03?tab=conditions');
    [...document.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.getAttribute('aria-label') === 'Next image')!.click();
    await settle();
    expect(document.querySelector('.stage .side.changed')).not.toBeNull();
    const list = vi.spyOn(hub.api, 'benchImages');
    url.mockClear();
    await hub.api.updateBench('bn_9f03', { name: 'Renamed elsewhere' });
    flushSync();
    // Never back to a skeleton while it loads again: nothing loads again.
    expect(document.querySelector('.stage .side.changed')).not.toBeNull();
    await settle();
    expect(list).not.toHaveBeenCalled();
    expect(url).not.toHaveBeenCalled();
    expect(document.querySelector('.pv .file')?.textContent).toBe((await hub.api.benchImages('bn_9f03', { offset: 1, limit: 1 })).images[0]!.file);
  });

  it('opens Compare with the entries in the board’s order, one graded on an earlier version last', async () => {
    await page('/benchmarks/bn_3c90');
    const box = (id: string) => document.querySelector<HTMLInputElement>(`table.board tbody tr[data-id="${id}"] input[type="checkbox"]`)!;
    // fasteners-s-v1 now (be_f101) and as graded before the engine changed (be_f100, a higher score, shown below).
    box('be_f100').click();
    flushSync();
    box('be_f101').click();
    flushSync();
    const order = [...document.querySelectorAll<HTMLElement>('table.board tbody tr[data-id]')].map((r) => r.dataset.id).filter((x) => x === 'be_f100' || x === 'be_f101');
    expect(order).toEqual(['be_f101', 'be_f100']);
    const compare = [...document.querySelectorAll<HTMLElement>('header a')].find((b) => /^Compare/.test(b.textContent?.trim() ?? ''))!;
    expect(compare.getAttribute('href')).toBe('#/benchmarks/bn_3c90/compare?entries=be_f101,be_f100');
  });

  it('waits for its entries before it shows a leaderboard or a plan, though the stream names the benchmark first', async () => {
    vi.useFakeTimers();
    try {
      // The stream's first `bench` event (30 ms) comes before the lists (140 ms, the demo's own latency).
      hub.api = new MockApi('normal', 140);
      router.route = parse('#/benchmarks/bn_3c90');
      open(BenchmarkPage, {});
      await vi.advanceTimersByTimeAsync(60);
      flushSync();
      expect(text()).not.toMatch(/No entries yet/);
      await vi.advanceTimersByTimeAsync(400);
      flushSync();
      expect(document.querySelectorAll('table.board tbody tr[data-id]').length).toBeGreaterThan(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it('says "Complete" only of a finished entry: one that runs or failed was graded on nothing yet', async () => {
    await page('/benchmarks/bn_3c90');
    const ran = (status: string) =>
      [...document.querySelectorAll<HTMLElement>('table.board tbody tr[data-id]')]
        .filter((r) => r.textContent!.includes(status))
        .map((r) => r.querySelector<HTMLElement>('td.ran')!.textContent!.trim());
    const all = [...document.querySelectorAll<HTMLElement>('table.board tbody td.ran')].map((td) => td.textContent!.trim());
    expect(all).toContain('Complete');
    expect(ran('Failed')).toHaveLength(1);
    expect(ran('Failed')[0]).not.toContain('Complete');
    expect(ran('Running')[0]).not.toContain('Complete');
  });

  it('picks finished entries for Compare, at most four, and makes Compare the one primary from two', async () => {
    await page('/benchmarks/bn_3c90');
    const boxes = [...document.querySelectorAll<HTMLInputElement>('table.board tbody input[type="checkbox"]')];
    // Seven finished entries have a box (one graded on an earlier version of the benchmark); the running and the failed
    // one have none.
    expect(boxes).toHaveLength(7);
    // The header asks to select two: the boxes show without a hover once two entries are finished.
    expect(document.querySelector('table.board')?.classList.contains('pickable')).toBe(true);
    const compare = () => [...document.querySelectorAll<HTMLElement>('header a, header button')].find((b) => /^Compare/.test(b.textContent?.trim() ?? ''))!;
    expect(compare().getAttribute('aria-disabled') === 'true' || (compare() as HTMLButtonElement).disabled).toBe(true);
    for (const b of boxes.slice(0, 5)) {
      b.click();
      flushSync();
    }
    expect(boxes.filter((b) => b.checked)).toHaveLength(4);
    expect(boxes[4]!.disabled).toBe(true);
    // In the board's order: the complete entries first, then the partial ones.
    const first4 = [...document.querySelectorAll<HTMLElement>('table.board tbody tr[data-id]')].slice(0, 4).map((r) => r.dataset.id);
    expect(compare().getAttribute('href')).toBe(`#/benchmarks/bn_3c90/compare?entries=${first4.join(',')}`);
  });
});

describe('the list', () => {
  it('lists the benchmarks with their leaders, and says what one is when there are none', async () => {
    router.route = parse('#/benchmarks');
    open(Benchmarks, {});
    await settle(20);
    expect([...document.querySelectorAll('table.benches tbody tr[data-row] td:first-child a.name')].map((x) => x.textContent?.trim())).toEqual(['Sorting tray, night shift', 'Fasteners val', 'Pond fish', 'Shapes val']);
    unmount(c!);
    hub.api = new MockApi('empty', 0);
    open(Benchmarks, {});
    await settle(20);
    expect(document.querySelectorAll('table.benches tbody tr[data-row]')).toHaveLength(0);
    expect(document.querySelector('.empty')).not.toBeNull();
  });

  it('explains itself on a server without benchmarks, with the way back', async () => {
    hub.api = new MockApi('no-bench', 0);
    router.route = parse('#/benchmarks');
    open(Benchmarks, {});
    await settle(20);
    expect(document.querySelector('table.benches')).toBeNull();
    expect(document.querySelector('a[href="#/models"]')).not.toBeNull();
  });

  // #324: the palette's New benchmark (`?new=1`) opened its form over "This server has no vision benchmarks", and
  // Create could only fail (501). It follows the server's health (`hub.features`), at once: there is no answer to wait for.
  const formOpen = () => [...document.querySelectorAll<HTMLDialogElement>('dialog')].some((d) => d.open && d.textContent?.includes('New benchmark'));
  for (const [what, features, opens] of [
    ['lists benchmarks', [BENCH_FEATURE], true],
    ['lists features without benchmarks', ['queue_control'], false],
    ['lists no features (an older server, the demo)', null, true],
  ] as const) {
    it(`?new=1 on a server whose health ${what}: ${opens ? 'the form' : 'the explanation, never the form'}`, async () => {
      if (!opens) hub.api = new MockApi('no-bench', 0);
      hub.features = features;
      try {
        router.route = parse('#/benchmarks?new=1');
        open(Benchmarks, {});
        expect(formOpen()).toBe(opens);
        if (!opens) {
          await until(() => text().includes('This server has no vision benchmarks'), 'the explanation');
          expect(formOpen()).toBe(false);
        }
      } finally {
        hub.features = null;
      }
    });
  }
});
