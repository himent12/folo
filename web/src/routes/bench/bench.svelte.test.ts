// @vitest-environment happy-dom
// The benchmark screens' behaviour against the in-browser mock: what the dialogs ask for and refuse, the plan that is
// fixed once there are entries (and the edits that are not lost), and the leaderboard's Compare selection.
import { flushSync, mount, unmount, type Component } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { ApiError } from '$lib/api/client';
import { MockApi } from '$lib/api/mock';
import type { Bench } from '$lib/api/types';
import { parse, router } from '$lib/router.svelte';
import { undescribedFields } from '$lib/ui/fieldA11y';
import { toast } from '$lib/ui/toast.svelte';
import BenchmarkPage from '../Benchmark.svelte';
import Benchmarks from '../Benchmarks.svelte';
import { AgentChoice } from '../label/agentChoice.svelte';
import AddEntryDialog from './AddEntryDialog.svelte';
import NewBenchDialog from './NewBenchDialog.svelte';
import { FROZEN } from './words';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;
let stopRoot: (() => void) | undefined;

// Svelte's select binding reads the chosen option with `:checked`, which happy-dom does not match on <option> (as in
// RunNew.svelte.test.ts).
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

function open<P extends Record<string, unknown>>(comp: Component<P>, props: P) {
  c = mount(comp as Component<Record<string, unknown>>, { target, props });
  flushSync();
}

/** Lets the mock's answers (latency 0) and the effects they start land. */
async function settle(rounds = 12) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
}

function type(el: HTMLInputElement | HTMLTextAreaElement, v: string) {
  el.value = v;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
}

const submit = () => document.querySelector<HTMLButtonElement>('dialog[open] button[type="submit"]')!;
const why = (b: HTMLElement) => document.getElementById(b.getAttribute('aria-describedby') ?? '')?.textContent;
const text = () => document.body.textContent ?? '';

describe('New benchmark', () => {
  it('describes every control and gates Create until it has a name and a source with images', async () => {
    open(NewBenchDialog, { open: true });
    await settle();
    expect(undescribedFields(document)).toEqual([]);
    expect(submit().disabled).toBe(true);
    expect(why(submit())).toBe('Enter a name');
    type(document.querySelector('input[id$="-name"]')!, 'Pond fish');
    expect(submit().disabled).toBe(false);
    // A validation split by default, and a quiet word on keeping it out of training.
    expect(document.querySelector('[role="radiogroup"][aria-label="Split"] [aria-checked="true"]')?.textContent).toMatch(/Validation/);
    expect(text()).toMatch(/Keep these images out of training/);
  });

  it('warns that a training split measures memory, not sight', async () => {
    open(NewBenchDialog, { open: true });
    await settle();
    (document.querySelector('[role="radiogroup"][aria-label="Split"] [role="radio"]:last-child') as HTMLElement).click();
    flushSync();
    expect(text()).toMatch(/measures memory, not sight/);
  });

  it('refuses an image limit outside 1–5000, saying so', async () => {
    open(NewBenchDialog, { open: true });
    await settle();
    type(document.querySelector('input[id$="-name"]')!, 'x');
    type(document.querySelector('input[id$="-max"]')!, '0');
    submit().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    flushSync();
    expect(document.querySelector('[role="alert"]')?.textContent).toMatch(/1 to 5,000/);
    expect(why(submit())).toBe('Fix the image limit first');
  });

  it('creates it with the default plan and opens its page', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    open(NewBenchDialog, { open: true });
    await settle();
    type(document.querySelector('input[id$="-name"]')!, 'Fasteners again');
    submit().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle();
    expect(go).toHaveBeenCalledWith(expect.stringMatching(/^\/benchmarks\/bn_[0-9a-f]{12}$/));
    expect(toast.items.at(-1)?.title).toBe('Benchmark created');
  });
});

describe('Add entry', () => {
  async function dialog(kind: 'model' | 'agent' | 'external', benchId = 'bn_3c90') {
    const bench: Bench = await hub.api.bench(benchId);
    const models = await hub.api.models();
    const agents = await hub.api.labelAgents();
    const entries = await hub.api.benchEntries(benchId);
    const catalog = await hub.api.benchCatalog();
    let choice!: AgentChoice;
    stopRoot = $effect.root(() => {
      choice = new AgentChoice(() => agents);
    });
    const onstarted = vi.fn();
    open(AddEntryDialog, { open: true, kind, bench, catalog, entries, models, modelsError: undefined, agents, agentsError: undefined, retryAgents: () => {}, choice, onstarted });
    await settle();
    return { onstarted };
  }

  it('shows a model’s classes against the benchmark’s: the ones it lacks score 0', async () => {
    await dialog('model');
    const sel = document.querySelector<HTMLSelectElement>('select[id$="-model"]')!;
    // (happy-dom sets a select's value only among its direct options: pick the option in its group.)
    sel.querySelector<HTMLOptionElement>('option[value="m_mini_n"]')!.selected = true;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(text()).toMatch(/It knows 2 of the 8 classes; on the 6 it lacks it scores 0/);
    expect([...document.querySelectorAll('.chip.miss')].map((x) => x.textContent?.replace(/,.*$/, '').trim())).toEqual(['washer', 'screw', 'spring', 'rivet', 'pin', 'bearing']);
    // A model with none of them cannot be picked.
    expect(sel.querySelector<HTMLOptionElement>('option[value="m_coco_s"]')?.disabled).toBe(true);
    // It waits for the model entry that runs.
    expect(text()).toMatch(/it waits for fasteners-n-f16/);
  });

  it('says an agent entry is partial, and starts one with the defaults', async () => {
    const { onstarted } = await dialog('agent');
    expect(text()).toMatch(/Its score is partial: agents give no confidences/);
    expect(document.querySelector<HTMLInputElement>('input[id$="-aimages"]')!.value).toBe('48');
    submit().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle();
    expect(onstarted).toHaveBeenCalledWith(expect.objectContaining({ images: 48, conditions: false, scored: false }));
  });

  it('checks an external predictions file before it is sent, naming what is wrong', async () => {
    const { onstarted } = await dialog('external', 'bn_9f03');
    const input = document.querySelector<HTMLInputElement>('input[type="file"]')!;
    const pick = async (body: string) => {
      Object.defineProperty(input, 'files', { configurable: true, value: [new File([body], 'det-predictions.json', { type: 'application/json' })] });
      input.dispatchEvent(new Event('change', { bubbles: true }));
      await settle();
    };
    await pick('{"predictions": [{"image": "1", "boxes": [{"class": 9, "x1": 0, "y1": 0, "x2": 4, "y2": 4}]}]}');
    expect(document.querySelector('.f-state.bad')?.textContent).toMatch(/^Record 1: box 1: "class" must be a class index from 0 to 5/);
    expect(why(submit())).toBe('Fix the predictions file first');
    expect(onstarted).not.toHaveBeenCalled();
    const images = (await hub.api.benchImages('bn_9f03', { limit: 1000 })).images;
    await pick(JSON.stringify(images.map((i) => ({ image: i.id, boxes: i.boxes.map((b) => ({ ...b, score: 0.8 })) }))));
    expect(document.querySelector('.f-state.ok')?.textContent).toMatch(/200 images answered · unchanged only/);
    // A name is suggested from the file's.
    expect(document.querySelector<HTMLInputElement>('input[id$="-name"]')!.value).toBe('det');
    submit().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle();
    expect(onstarted).toHaveBeenCalledWith(expect.objectContaining({ status: 'done', name: 'det' }));
  });

  /** A file whose request would be large: every image under every pass, enough boxes to be over 8 MiB as JSON. */
  async function largeFile() {
    const b = await hub.api.bench('bn_9f03');
    const images = (await hub.api.benchImages('bn_9f03', { limit: 1000 })).images;
    const pass = [{ c: null as string | null, s: null as number | null }, ...b.plan.conditions.flatMap((c) => c.severities.map((s) => ({ c: c.condition as string | null, s: s as number | null })))];
    const box = { class: 0, score: 0.5, x1: 1, y1: 2, x2: 30, y2: 40 };
    const records = pass.flatMap((p) => images.map((i) => ({ image: i.id, ...(p.c ? { condition: p.c, severity: p.s } : {}), boxes: Array.from({ length: 12 }, () => box) })));
    return { records, passes: pass.length };
  }
  async function pickLarge() {
    const { records, passes } = await largeFile();
    const input = document.querySelector<HTMLInputElement>('input[type="file"]')!;
    Object.defineProperty(input, 'files', { configurable: true, value: [new File([JSON.stringify(records)], 'big.json', { type: 'application/json' })] });
    input.dispatchEvent(new Event('change', { bubbles: true }));
    await settle(40);
    return { records, passes };
  }

  it('sends a large file as an open entry, a pass at a time, then finishes it', async () => {
    const { onstarted } = await dialog('external', 'bn_9f03');
    const { records, passes } = await pickLarge();
    expect(document.querySelector('.f-state.ok')?.textContent).toMatch(new RegExp(`Sent in ${passes} parts, a pass at a time\\.`));
    const send = vi.spyOn(hub.api, 'sendBenchPredictions');
    submit().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle(60);
    expect(send).toHaveBeenCalledTimes(passes);
    expect(send.mock.calls.reduce((n, c) => n + c[2].length, 0)).toBe(records.length);
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
    submit().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle(60);
    expect(onstarted).not.toHaveBeenCalled();
    expect(stop).toHaveBeenCalledTimes(1);
    expect(text()).toMatch(/Could not add the entry: nothing was kept/);
    expect(text()).toMatch(/Record 7: box 1/);
    expect((await hub.api.benchEntries('bn_9f03')).length).toBe(0);
    expect((await hub.api.bench('bn_9f03')).active).toBe(false);
  });

  it('asks before closing mid-send, then stops the entry', async () => {
    await dialog('external', 'bn_9f03');
    await pickLarge();
    // The second part never answers: the send is in flight when the person closes the dialog.
    let n = 0;
    const real = hub.api.sendBenchPredictions.bind(hub.api);
    vi.spyOn(hub.api, 'sendBenchPredictions').mockImplementation((...a) => (++n === 2 ? new Promise(() => {}) : real(...a)));
    const stop = vi.spyOn(hub.api, 'stopBenchEntry');
    submit().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle(30);
    expect(document.querySelector('dialog[open] .f-send')?.textContent).toMatch(/Sending predictions · [\d,]+ of [\d,]+ records/);
    [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => b.textContent?.trim() === 'Cancel')!.click();
    flushSync();
    expect(text()).toMatch(/Stop sending\? The entry is discarded\./);
    expect(document.querySelector('dialog[open]')).not.toBeNull();
    [...document.querySelectorAll<HTMLButtonElement>('dialog[open] button')].find((b) => b.textContent?.trim() === 'Stop and discard')!.click();
    await settle(30);
    expect(stop).toHaveBeenCalledTimes(1);
    expect((await hub.api.benchEntries('bn_9f03')).length).toBe(0);
  });
});

describe('the benchmark page', () => {
  async function page(path: string) {
    router.route = parse(`#${path}`);
    open(BenchmarkPage as Component, {});
    await settle(20);
  }

  it('shows a plan with entries as fixed, with its reason, and no controls to change it', async () => {
    await page('/benchmarks/bn_7a41?tab=conditions');
    expect(text()).toContain(FROZEN);
    const tab = document.getElementById('bn-tab-panel')!;
    expect(tab.querySelectorAll('input[role="switch"]')).toHaveLength(0);
    expect(tab.querySelectorAll('button.sev')).toHaveLength(0);
    expect(tab.querySelectorAll('.sev.static.has').length).toBeGreaterThan(0);
    expect(text()).toMatch(/Designed by Claude Code \(Sonnet\)/);
    expect(document.querySelector('button')?.textContent).not.toMatch(/Design with an agent/);
  });

  it('edits the plan of a benchmark with no entries, holds Add entry until it is saved, and asks before leaving', async () => {
    await page('/benchmarks/bn_9f03?tab=conditions');
    const sev = [...document.querySelectorAll<HTMLButtonElement>('button.sev')].find((b) => b.getAttribute('aria-pressed') === 'false')!;
    sev.click();
    flushSync();
    expect(text()).toMatch(/Unsaved changes/);
    const add = [...document.querySelectorAll<HTMLButtonElement>('header button')].find((b) => /Add entry/.test(b.textContent ?? ''))!;
    expect(add.disabled).toBe(true);
    expect(why(add)).toBe('Save or discard the plan’s changes first');
    // Leaving asks first; Discard lets the navigation through.
    const go = vi.spyOn(history, 'pushState');
    router.go('/benchmarks');
    flushSync();
    expect(document.querySelector('dialog[open] h2')?.textContent).toBe('Save the plan?');
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

  it('picks finished entries for Compare, at most four, and makes Compare the one primary from two', async () => {
    await page('/benchmarks/bn_3c90');
    const boxes = [...document.querySelectorAll<HTMLInputElement>('table.board tbody input[type="checkbox"]')];
    // Seven finished entries have a box (one graded on an earlier version of the benchmark); the running and the failed
    // one have none.
    expect(boxes).toHaveLength(7);
    const compare = () => [...document.querySelectorAll<HTMLElement>('header a, header button')].find((b) => /^Compare/.test(b.textContent?.trim() ?? ''))!;
    expect(compare().getAttribute('aria-disabled') === 'true' || (compare() as HTMLButtonElement).disabled).toBe(true);
    for (const b of boxes.slice(0, 5)) {
      b.click();
      flushSync();
    }
    expect(boxes.filter((b) => b.checked)).toHaveLength(4);
    expect(boxes[4]!.disabled).toBe(true);
    expect(text()).toMatch(/4 selected \(the maximum\)/);
    // In the board's order: the complete entries first, then the partial ones.
    const first4 = [...document.querySelectorAll<HTMLElement>('table.board tbody tr[data-id]')].slice(0, 4).map((r) => r.dataset.id);
    expect(compare().getAttribute('href')).toBe(`#/benchmarks/bn_3c90/compare?entries=${first4.join(',')}`);
  });
});

describe('the list', () => {
  it('lists the benchmarks with their leaders, and says what one is when there are none', async () => {
    router.route = parse('#/benchmarks');
    open(Benchmarks as Component, {});
    await settle(20);
    expect([...document.querySelectorAll('table.benches tbody tr[data-row] td:first-child a.name')].map((x) => x.textContent?.trim())).toEqual(['Sorting tray, night shift', 'Fasteners val', 'Pond fish', 'Shapes val']);
    unmount(c!);
    c = undefined;
    hub.api = new MockApi('empty', 0);
    open(Benchmarks as Component, {});
    await settle(20);
    expect(text()).toMatch(/No benchmarks yet/);
    expect(text()).toMatch(/grades any model on them/);
  });

  it('explains itself on a server without benchmarks', async () => {
    hub.api = new MockApi('no-bench', 0);
    router.route = parse('#/benchmarks');
    open(Benchmarks as Component, {});
    await settle(20);
    expect(text()).toMatch(/This server has no vision benchmarks/);
  });
});
