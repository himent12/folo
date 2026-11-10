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

/** Lets the mock's answers (latency 0) and the effects they start land. `setImmediate`, not `setTimeout(0)`, which
 *  waits a timer tick (about 15 ms on Windows) each round. */
async function settle(rounds = 12) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setImmediate(r));
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
    // An image limit outside 1–5000 is refused at the field, and holds Create.
    type(document.querySelector('input[id$="-max"]')!, '0');
    send();
    flushSync();
    expect(document.querySelector('[role="alert"]')?.textContent).toMatch(/5,000/);
    expect(why(submit())).toBeTruthy();
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
    open(BenchmarkPage as Component, {});
    await settle(20);
  }

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
    hub.api = new MockApi('empty', 0);
    open(Benchmarks as Component, {});
    await settle(20);
    expect(document.querySelectorAll('table.benches tbody tr[data-row]')).toHaveLength(0);
    expect(document.querySelector('.empty')).not.toBeNull();
  });

  it('explains itself on a server without benchmarks, with the way back', async () => {
    hub.api = new MockApi('no-bench', 0);
    router.route = parse('#/benchmarks');
    open(Benchmarks as Component, {});
    await settle(20);
    expect(document.querySelector('table.benches')).toBeNull();
    expect(document.querySelector('a[href="#/models"]')).not.toBeNull();
  });
});
