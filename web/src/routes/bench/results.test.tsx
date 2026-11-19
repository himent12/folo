// @vitest-environment happy-dom
// The result screens against the in-browser mock: an entry's page in each of its states (complete, an agent's partial
// one, never reaching 90% precision, running, failed), its failure gallery's paging and large view, the compare page's
// selection and its states, the new-loop dialog's eligibility and checks, a loop's page, and the leaderboard's scale.
// Their sentences are tested on their own (./reading.test.ts, ./sentences.test.ts); here, that each shows where it should.
import { flush as flushSync } from 'fw';
import { createComponent } from 'fw/internal';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { ApiError } from '$lib/api/client';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import { render } from '../../../framework/testing';
import BenchCompare from '../BenchCompare';
import BenchEntry from '../BenchEntry';
import BenchLoop from '../BenchLoop';
import BenchmarkPage from '../Benchmark';
import type { Bench, BenchEntry as BenchEntryT, BenchResult as BenchResultT } from '$lib/api/types';
import AccuracySection from './entry/AccuracySection';
import CalibrationSection from './entry/CalibrationSection';
import { GALLERY_PAGE } from './entry/ImagesSection';
import SelfLabelSection from './entry/SelfLabelSection';
import NewLoopDialog from './NewLoopDialog';
import { FROZEN, loopPlanText } from './words';

let target: HTMLElement;
/** Port of `mount(comp, { target, props })`: the props object is read live (a `deep` one changes them). */
function mount<P extends object>(comp: (props: P) => JSX.Element, o: { target: HTMLElement; props?: P }) {
  const props = (o.props ?? {}) as P;
  return render(() => createComponent(comp, props), o.target);
}
const unmount = (c: { unmount(): void }) => c.unmount();
type Component<P extends object = Record<string, unknown>> = (props: P) => JSX.Element;

let c: ReturnType<typeof mount> | undefined;

// (Svelte's select binding read the chosen option with `:checked`, which happy-dom does not match on <option>; fw reads
// `options[selectedIndex]`, so the legacy suite's patch of HTMLSelectElement.prototype.querySelector is not needed.)

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

/** Lets the mock's answers (latency 0), the timers they start and the effects after those land: a timer tick a round
 *  (`setImmediate` rounds can all pass before a 1 ms timer is due, which failed in CI). */
async function settle(rounds = 24) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
}
async function page(comp: Component, path: string) {
  if (c) unmount(c);
  router.route = parse(`#${path}`);
  c = mount(comp, { target, props: {} });
  flushSync();
  await settle();
}
const text = () => document.body.textContent ?? '';
const button = (re: RegExp, root: ParentNode = document) => [...root.querySelectorAll<HTMLButtonElement>('button')].find((b) => re.test(b.textContent?.trim() ?? ''));
const pick = (sel: HTMLSelectElement, value: string) => {
  sel.querySelector<HTMLOptionElement>(`option[value="${value}"]`)!.selected = true;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
};
const type = (el: HTMLInputElement, v: string) => {
  el.value = v;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  flushSync();
};

/** The plots draw once they have a width; happy-dom lays nothing out, so every element reports one while `fn` runs. */
async function withWidth(fn: () => Promise<void>) {
  const was = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth');
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 600 });
  try {
    await fn();
  } finally {
    if (was) Object.defineProperty(HTMLElement.prototype, 'clientWidth', was);
  }
}

/** Focuses the chart in `root`, presses → twice and End, and gives what its live region said at each. */
function speak(root: string) {
  const svg = document.querySelector<SVGElement>(`${root} svg[aria-roledescription=chart]`)!;
  expect(svg, root).not.toBeNull();
  const said: string[] = [];
  for (const key of ['ArrowRight', 'ArrowRight', 'End']) {
    svg.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
    flushSync();
    said.push(svg.closest('.xy')!.querySelector('[aria-live]')!.textContent ?? '');
  }
  return said;
}

describe("an entry's page", () => {
  it('reads a complete entry: rows that add up, every section, its charts by keyboard, its images paged and opened large', async () => {
    await withWidth(async () => {
      await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f101');
      // The strip jumps to each section, and each is there.
      const links = [...document.querySelectorAll<HTMLAnchorElement>('nav.jump a')].map((a) => a.getAttribute('href')!);
      expect(links).toHaveLength(7);
      for (const id of links) expect(document.querySelector(id)).not.toBeNull();
      // Five axes, each its points; they add up to the total written under them.
      const pts = [...document.querySelectorAll('#score .r[data-axis] .c-pts')].map((x) => Number(x.textContent!.replace('+', '')));
      expect(pts).toHaveLength(5);
      const total = Number(document.querySelector('#score .r.sum .c-pts')!.textContent);
      expect(pts.reduce((a, b) => a + b, 0)).toBeCloseTo(total, 6);
      expect(document.querySelector('#score .big')?.textContent).toBe(String(total));
      // Where it breaks: one row per condition of the plan, worst kept first.
      const rows = document.querySelectorAll('#robustness .heat tbody tr');
      expect(rows).toHaveLength(12);
      const means = [...rows].map((r) => Number(r.querySelector('.mean')!.textContent!.replace('%', '')));
      expect(means).toEqual([...means].sort((a, b) => a - b));
      // The ramp's ink is dark in light theme and bright in dark theme: the caption names neither.
      const lead = document.querySelector('#robustness .lead')!.textContent!;
      expect(lead).toMatch(/stronger a cell’s shade, the more it lost/);
      expect(lead).not.toMatch(/darker|lighter/i);
      // Its charts announce a row at every stop.
      for (const s of speak('#self-labeling')) expect(s).toMatch(/^From [01]\.\d+ up: [\d,]+ labels?, \d+% right/);
      for (const s of speak('#calibration')) expect(s).toMatch(/\d/);
    });
    // The gallery pages, the worst first, and opens one large, stepping on to the next.
    await settle();
    const tiles = () => document.querySelectorAll('#images .tile:not(.sk)');
    expect(tiles()).toHaveLength(GALLERY_PAGE);
    button(/^Show \d+ more/)!.click();
    await settle();
    expect(tiles()).toHaveLength(2 * GALLERY_PAGE);
    (tiles()[0] as HTMLButtonElement).click();
    await settle();
    expect(document.querySelector('dialog[open] .v-pos')?.textContent).toBe('1 of 360');
    expect(document.querySelectorAll('dialog[open] .v-list li').length).toBeGreaterThan(0);
    button(/^Next/, document.querySelector('dialog[open]')!)!.click();
    await settle();
    expect(document.querySelector('dialog[open] .v-pos')?.textContent).toBe('2 of 360');
  });

  it("shows an agent's axes it did not measure without their charts", async () => {
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f104');
    expect(document.querySelector('#score')).not.toBeNull();
    expect(document.querySelector('#calibration svg')).toBeNull();
    expect(document.querySelector('#robustness .heat')).toBeNull();
    // A measured axis takes a larger share than its plan weight, and says which is which.
    const w = document.querySelector('#score .r[data-axis="accuracy"] .c-w')!.textContent!;
    expect(w).toMatch(/^weight \d+%\d+% in the plan$/);
  });

  it('shows a running entry where it is, and a failed one why', async () => {
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f108');
    expect(document.querySelectorAll('.passes .pill.done').length).toBeGreaterThan(0);
    expect(document.querySelectorAll('.passes .pill.now')).toHaveLength(1);
    expect(document.querySelector('nav.jump')).toBeNull();
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f107');
    const failed = await hub.api.benchEntry('bn_3c90', 'be_f107');
    expect(text()).toContain(failed.error!);
  });
});

describe('the compare page', () => {
  it('lines entries up with the first as the baseline, and says what is not like for like', async () => {
    await page(BenchCompare, '/benchmarks/bn_3c90/compare?entries=be_f101,be_f104');
    expect(document.querySelectorAll('.chips .chip')).toHaveLength(2);
    expect(document.querySelector('.cmp thead th.e-col .role')?.textContent).toBe('Baseline');
    expect(document.querySelectorAll('.cmp tbody tr')).toHaveLength(6);
    // The agent ran 48 of the 360 images.
    expect(text()).toMatch(/48 of 360 images/);
  });

  it('says when a result fails to load, with a retry; reads its chart by keyboard; adds and removes entries in the URL', async () => {
    const real = hub.api.benchResult.bind(hub.api);
    let fail = true;
    vi.spyOn(hub.api, 'benchResult').mockImplementation(async (b, e, s) => {
      if (fail && e === 'be_f103') throw new ApiError(500, 'internal', 'the result file could not be read');
      return real(b, e, s);
    });
    await withWidth(async () => {
      await page(BenchCompare, '/benchmarks/bn_3c90/compare?entries=be_f101,be_f103');
      expect(document.querySelector('.pair')).toBeNull();
      fail = false;
      button(/^Retry|^Try again/)!.click();
      await settle();
      expect(button(/^Retry|^Try again/)).toBeUndefined();
      // Its calibration overlay (its self-labeling trade-off has no stops: its table is what is read).
      for (const s of speak('.pair > :nth-child(1)')) expect(s).toMatch(/\d/);
    });
    // Recorded, not followed: the page asks for the URL; the router (tested on its own) follows it.
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    document.querySelector<HTMLButtonElement>('.chip button.rm[aria-label="Remove fasteners-n-best"]')!.click();
    expect(go).toHaveBeenCalledWith('/benchmarks/bn_3c90/compare?entries=be_f101', { replace: true });
    pick(document.querySelector<HTMLSelectElement>('.add select')!, 'be_f102');
    expect(go).toHaveBeenLastCalledWith('/benchmarks/bn_3c90/compare?entries=be_f101,be_f103,be_f102', { replace: true });
  });

  it('shows an entry the link names twice once', async () => {
    await page(BenchCompare, '/benchmarks/bn_3c90/compare?entries=be_f101,be_f101,be_f104');
    await settle(40);
    expect([...document.querySelectorAll('.chips .chip .nm')].map((x) => x.textContent)).toEqual(['fasteners-s-v1', 'Claude Code (sonnet)']);
    expect(document.querySelectorAll('.cmp thead th.e-col')).toHaveLength(2);
  });

  it('says why calibration was not measured for an entry with confidences that never reach 0.1', async () => {
    await page(BenchCompare, '/benchmarks/bn_7a41/compare?entries=be_p201,be_p205');
    await settle(40);
    const note = [...document.querySelectorAll('.pair p.note')].map((p) => p.textContent ?? '').find((t) => /yolo26n-pond-raw/.test(t)) ?? '';
    expect(note).toMatch(/yolo26n-pond-raw: no detection reached confidence 0\.1/);
    expect(note).not.toMatch(/no confidences/);
  });

  it('explains a link with fewer than two finished entries, with the way back', async () => {
    await page(BenchCompare, '/benchmarks/bn_3c90/compare?entries=be_f101,be_gone');
    expect(document.querySelector('.cmp')).toBeNull();
    expect(document.querySelector('a[href="#/benchmarks/bn_3c90"]')).not.toBeNull();
  });
});

describe('New loop', () => {
  async function dialog(teacher: string | null = null, benchId = 'bn_3c90') {
    const bench = await hub.api.bench(benchId);
    const [models, datasets, jobs, entries] = await Promise.all([hub.api.models(), hub.api.datasets(), hub.api.labelJobs(), hub.api.benchEntries(benchId)]);
    const onstarted = vi.fn();
    c = mount(NewLoopDialog, { target, props: { open: true, bench, models, datasets, jobs, entries, teacher, onstarted } });
    flushSync();
    await settle();
    return { onstarted };
  }
  const q = <T extends Element>(sel: string) => document.querySelector<T>(`dialog[open] ${sel}`)!;
  const submit = () => q<HTMLButtonElement>('button[type="submit"]');
  const send = async () => {
    submit().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle();
  };
  const fieldError = (input: HTMLElement) => document.getElementById(`${input.id}-error`)?.textContent ?? null;

  it('offers the models with the benchmark’s classes as teachers, the others off', async () => {
    await dialog();
    const sel = q<HTMLSelectElement>('select[id$="-teacher"]');
    const opt = (id: string) => sel.querySelector<HTMLOptionElement>(`option[value="${id}"]`)!;
    expect(opt('m_fast_n').disabled).toBe(false);
    expect(opt('m_mini_n').disabled).toBe(true);
    expect(opt('m_coco_s').disabled).toBe(true);
    // Its complete entry is generation 0, so the plan costs one entry per student.
    expect(q('.s-text').textContent).toBe(loopPlanText({ generations: 1, epochs: 30, poolImages: 1920, teacherEntry: true }));
  });

  it('reads a good teacher’s labels, checks the student’s settings, shows a refused start at its field, then starts', async () => {
    const { onstarted } = await dialog('m_fast_n');
    expect(q('.fc').getAttribute('data-forecast')).toBe('ok');
    const ep = q<HTMLInputElement>('input[id$="-ep"]');
    const sz = q<HTMLInputElement>('input[id$="-sz"]');
    type(ep, '0');
    type(sz, '650');
    await send();
    expect(fieldError(ep)).not.toBeNull();
    expect(fieldError(sz)).not.toBeNull();
    expect(onstarted).not.toHaveBeenCalled();
    // The server refuses a size the dialog lets through: its message at the field it names, not on the form.
    const start = vi.spyOn(hub.api, 'startBenchLoop').mockRejectedValueOnce(new ApiError(400, 'invalid_config', 'imgsz must be at least 64 for training, got 32'));
    type(ep, '20');
    type(sz, '32');
    await send();
    expect(start).toHaveBeenCalledTimes(1);
    expect(fieldError(sz)).toMatch(/imgsz must be at least 64/);
    expect(document.querySelector('dialog[open] .error-state')).toBeNull();
    type(sz, '640');
    await send();
    expect(onstarted).toHaveBeenCalledWith(expect.objectContaining({ status: 'running', config: expect.objectContaining({ teacher: 'm_fast_n', student: expect.objectContaining({ epochs: 20 }) }) }));
  });

  it('has the students follow the teacher picked, until a person picks their size', async () => {
    await dialog(null, 'bn_5e12');
    const teacher = q<HTMLSelectElement>('select[id$="-teacher"]');
    const arch = () => q<HTMLSelectElement>('select[id$="-arch"]').value;
    const pickTeacher = async (id: string) => {
      pick(teacher, id);
      await settle();
    };
    await pickTeacher('m_shapes_s');
    expect(arch()).toBe('folo-s');
    await pickTeacher('m_shapes_e5');
    expect(arch()).toBe('folo-n');
    // A size a person picked stays.
    pick(q<HTMLSelectElement>('select[id$="-arch"]'), 'folo-m');
    await settle();
    await pickTeacher('m_shapes_s');
    expect(arch()).toBe('folo-m');
  });

  it('opened, closed and opened again on the same page, still has the students’ defaults (they follow the teacher)', async () => {
    await page(BenchmarkPage, '/benchmarks/bn_3c90?tab=loops');
    const arch = () => document.querySelector<HTMLSelectElement>('dialog[open] select[id$="-arch"]')?.value;
    const size = () => document.querySelector<HTMLInputElement>('dialog[open] input[id$="-sz"]')?.value;
    const openIt = async () => {
      button(/^New loop$/)!.click();
      await settle();
    };
    const cancel = async () => {
      button(/^Cancel$/, document.querySelector('dialog[open]')!)!.click();
      await settle();
    };
    for (let k = 0; k < 2; k++) {
      await openIt();
      expect(arch(), `opening ${k + 1}`).toBe('folo-n');
      expect(size(), `opening ${k + 1}`).toBe('640');
      expect(submit().disabled).toBe(false);
      await cancel();
    }
    // A size picked the first time is not kept into the next opening.
    await openIt();
    pick(document.querySelector<HTMLSelectElement>('dialog[open] select[id$="-arch"]')!, 'folo-m');
    await settle();
    expect(arch()).toBe('folo-m');
    await cancel();
    await openIt();
    expect(arch()).toBe('folo-n');
  });

  it('keeps its forecast while the benchmark changes under it (a `bench` event and the lists loaded again)', async () => {
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f103');
    await settle(40);
    button(/^Test it: start a loop/)!.click();
    await settle();
    expect(document.querySelector('dialog[open] .fc')?.getAttribute('data-forecast')).toBe('ok');
    const results = vi.spyOn(hub.api, 'benchResult');
    // What the dialog shows as the event lands (this handler runs after the page's).
    let during: string | null | undefined;
    const off = hub.api.benchEvents('bn_3c90', {
      bench: () => {
        flushSync();
        during = document.querySelector('dialog[open] .fc')?.getAttribute('data-forecast');
      },
    });
    await hub.api.updateBench('bn_3c90', { name: 'Fasteners, renamed' });
    off();
    expect(during).toBe('ok');
    await settle(40);
    expect(results).not.toHaveBeenCalled();
    expect(document.querySelector('dialog[open] .fc')?.getAttribute('data-forecast')).toBe('ok');
  });

  it('requires a confidence for a teacher whose entry found no threshold, and forecasts each one live', async () => {
    // shapes-n-early on Shapes val: 90% right only on a handful of labels, too few for a threshold. A loop never
    // guesses a confidence, so one is required here before it can start.
    await dialog('m_shapes_e5', 'bn_5e12');
    const fc = () => q('.fc');
    expect(fc().getAttribute('data-forecast')).toBe('none');
    expect(fc().getAttribute('role')).toBe('alert');
    expect(q<HTMLInputElement>('input[type=checkbox]').disabled).toBe(true);
    const conf = q<HTMLInputElement>('input[id$="-conf"]');
    expect(conf.value).toBe('');
    expect(submit().disabled).toBe(true);
    // A high confidence keeps a handful: the loop would stop at labeling, and Start is let through.
    type(conf, '0.9');
    await settle();
    expect(fc().getAttribute('data-forecast')).toBe('stops');
    expect(submit().disabled).toBe(false);
    // Lower: enough labels, fewer of them right.
    const right = () => Number(/(\d+)% of its labels are right/.exec(fc().textContent!)![1]);
    type(conf, '0.30');
    await settle();
    expect(fc().getAttribute('data-forecast')).toBe('ok');
    const before = right();
    type(conf, '0.05');
    await settle();
    expect(fc().textContent).toMatch(/From 0\.050, /);
    expect(right()).toBeLessThan(before);
  });
});

describe("a loop's page", () => {
  it('gives its verdict and lists each generation, a line of labels for each student', async () => {
    await page(BenchLoop, '/benchmarks/bn_5e12/loops/bl_07d1');
    expect(document.querySelector('.v-word')?.textContent).toBe('Improved');
    expect(document.querySelectorAll('table.gens tbody tr')).toHaveLength(4);
    expect(document.querySelectorAll('table.gens tbody .ln')).toHaveLength(3);
  });

  it('at generation 0 shows one step, the benchmark grading the teacher, and the generations to come', async () => {
    hub.api = new MockApi('bench-loop-gen0', 0);
    await page(BenchLoop, '/benchmarks/bn_3c90/loops/bl_g0a1');
    expect([...document.querySelectorAll('.steps li .s-state')].map((x) => x.textContent)).toEqual(['now']);
    expect(document.querySelectorAll('table.gens tbody tr.future')).toHaveLength(2);
  });

  it('says why a loop starved in the server’s words, with no steps', async () => {
    await page(BenchLoop, '/benchmarks/bn_5e12/loops/bl_9e40');
    const loop = await hub.api.benchLoop('bn_5e12', 'bl_9e40');
    expect(text()).toContain(loop.error!);
    expect(document.querySelector('[role=alert]')).not.toBeNull();
    expect(document.querySelector('.steps')).toBeNull();
  });

  it('shows the step a running loop is on, and offers Stop', async () => {
    await page(BenchLoop, '/benchmarks/bn_3c90/loops/bl_f3a0');
    expect([...document.querySelectorAll('.steps li .s-state')].map((x) => x.textContent)).toEqual(['done', 'now', 'next']);
    expect(button(/^Stop loop/)).toBeTruthy();
  });

  it('opens from the loops’ old links (`?tab=loops&loop=`)', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    await page(BenchmarkPage, '/benchmarks/bn_5e12?tab=loops&loop=bl_07d1');
    expect(go).toHaveBeenCalledWith('/benchmarks/bn_5e12/loops/bl_07d1', { replace: true });
  });

describe('the loops tab', () => {
  it('gives a loop with one generation scored one score on phones, never a change from it to itself', async () => {
    await page(BenchmarkPage, '/benchmarks/bn_5e12?tab=loops');
    const loop = (await hub.api.benchLoops('bn_5e12')).find((l) => l.name === 'shapes-n-early loop')!;
    const scored = loop.generations.filter((g) => g.score !== null);
    expect(scored).toHaveLength(1);
    const li = [...document.querySelectorAll('ul.rows li')].find((r) => /shapes-n-early loop/.test(r.textContent ?? ''))!;
    expect(li.querySelector('.m-meta')?.textContent).not.toMatch(/→/);
  });
});

describe('the failure gallery', () => {
  it('keeps its pages and the image open when the benchmark changes under it (a `bench` event)', async () => {
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f101');
    await settle(40);
    const tiles = () => document.querySelectorAll('#images .tile:not(.sk)');
    button(/^Show 24 more/)!.click();
    await settle(40);
    expect(tiles()).toHaveLength(2 * GALLERY_PAGE);
    (tiles()[30] as HTMLButtonElement).click();
    await settle();
    expect(document.querySelector('dialog[open] .v-pos')?.textContent).toBe('31 of 360');
    const pages = vi.spyOn(hub.api, 'benchEntryImages');
    // A rename elsewhere: the stream sends the benchmark again, and the lists load again.
    await hub.api.updateBench('bn_3c90', { name: 'Fasteners, renamed' });
    await settle(40);
    expect(pages).not.toHaveBeenCalled();
    expect(tiles()).toHaveLength(2 * GALLERY_PAGE);
    expect(document.querySelector('dialog[open] .v-pos')?.textContent).toBe('31 of 360');
    expect(document.querySelectorAll('dialog[open] .v-list li').length).toBeGreaterThan(0);
  });
});

describe('the calibration panel', () => {
  it('keeps the whole reading of a model close on average with one full bin far off, under its tone', async () => {
    // 100 detections: one bin of 22 eleven points overconfident (ECE 0.024, under 0.03: close), the rest thin and on the
    // diagonal. No bin near the diagonal is full enough to quote, and one is far off: "read the bins".
    const bin = (lo: number, count: number, gap = 0) => ({ lo, hi: lo + 0.1, count, confidence: lo + 0.05, precision: lo + 0.05 - gap });
    const bins = [{ lo: 0, hi: 0.1, count: 0, confidence: null, precision: null }, bin(0.1, 0), bin(0.2, 15), bin(0.3, 15), bin(0.4, 15), bin(0.5, 22, 0.11), bin(0.6, 15), bin(0.7, 18), bin(0.8, 0), bin(0.9, 0)];
    const result = { calibration: { ece: 0.0242, detections: 100, bins } } as unknown as BenchResultT;
    c = mount(CalibrationSection, { target, props: { result, entry: {} as BenchEntryT, bench: {} as Bench } });
    flushSync();
    expect(document.querySelector('#calibration .tone')?.textContent).toBe('Well calibrated');
    expect(document.querySelector('#calibration .sentence')?.textContent).toBe('Close to the diagonal on average (ECE 0.02). Some of its scores are off by more: read the bins.');
  });
});

describe('an entry none of whose detections is right', () => {
  it('says it has no best F1, in its accuracy and its images, instead of calling 1.00 its best', async () => {
    // F1 0 at every confidence: the server reports `best_conf` 1 (the grid's top) and grades the images at every detection.
    const real = hub.api.benchResult.bind(hub.api);
    vi.spyOn(hub.api, 'benchResult').mockImplementation(async (id, entryId) => {
      const r = await real(id, entryId);
      return { ...r, accuracy: { ...r.accuracy, precision: 0, recall: 0, f1: 0, best_conf: 1 } };
    });
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f101');
    await settle(40);
    const accuracy = document.querySelector('#accuracy')!.textContent ?? '';
    const images = document.querySelector('#images .what')!.textContent ?? '';
    expect(accuracy).toMatch(/No confidence gives a right detection/);
    expect(images).toMatch(/^Every detection it made \(none is right at any confidence, so there is no best F1\)/);
    expect(accuracy + images).not.toMatch(/1\.00|\(best F1\)|its best F1/);
  });
});

describe('the accuracy panel', () => {
  it('says a size is missing from the images an entry ran, not from a benchmark it ran a part of', async () => {
    // An agent's 48 of 360 images hold no small object; the others may (the server measures the images it ran).
    const accuracy = { map50_95: 0.4, ci95: [0.35, 0.45], map50: 0.6, map75: 0.4, precision: 0.8, recall: 0.5, f1: 0.6, best_conf: 1, map_small: null, map_medium: 0.4, map_large: 0.45, per_class: [] };
    const bench = { classes: [], images: 360 } as unknown as Bench;
    c = mount(AccuracySection, { target, props: { result: { accuracy } as unknown as BenchResultT, entry: { images: 48, scored: false, missing_classes: [] } as unknown as BenchEntryT, bench } });
    flushSync();
    const small = [...document.querySelectorAll('#accuracy .sizes li')].find((li) => /Small/.test(li.textContent ?? ''))!;
    expect(small.textContent).not.toMatch(/in this benchmark/);
    expect(small.textContent).toMatch(/None in the 48 images it ran/);
  });
});

describe('the self-labeling panel', () => {
  it('calls the labels of a contestant without confidences that reaches 90% its labels, not detections none of which are kept', async () => {
    // An agent: one operating point, 40 boxes, 95% right, covering 62% of all objects: its axis is 0.62.
    const point = { confidence: 0, precision: 0.95, recall: 0.62, labels: 40 };
    const self_labeling = { target_precision: 0.9, threshold: null, precision: 0.95, recall: 0.62, mean_iou: 0.8, labels_per_image: 0.83, labels: 40, readiness: 'ready', curve: [point] };
    const result = { self_labeling } as unknown as BenchResultT;
    c = mount(SelfLabelSection, { target, props: { result, entry: { scored: false, images: 48 } as BenchEntryT, loop: null } });
    flushSync();
    const kv = Object.fromEntries([...document.querySelectorAll('#self-labeling .kv > div')].map((d) => [d.querySelector('dt')!.textContent, d.querySelector('dd')!.textContent]));
    expect(kv['Axis (objects at 90%)']).toBe('62%');
    expect(Object.keys(kv).some((k) => /none kept/.test(k))).toBe(false);
  });
});
  it('refetches the lists when the loop names an id they lack, and calls it deleted only if it is still missing', async () => {
    const datasets = vi.spyOn(hub.api, 'datasets');
    await page(BenchLoop, '/benchmarks/bn_3c90/loops/bl_f3a0');
    // Its first student's dataset is not in the list loaded at mount: asked again, once.
    expect(datasets.mock.calls.length).toBe(2);
    await settle();
    expect(datasets.mock.calls.length).toBe(2);
    // The mock's loop datasets are not datasets of the server: after the second look, deleted.
    expect(text()).toMatch(/dataset deleted/);
  });

  it('shows a dataset made after the page loaded as a link, never as deleted', async () => {
    const real = hub.api.datasets.bind(hub.api);
    const loop = await hub.api.benchLoop('bn_3c90', 'bl_f3a0');
    const made = loop.generations[1]!.labels!.dataset_id;
    let calls = 0;
    vi.spyOn(hub.api, 'datasets').mockImplementation(async (s) => {
      const list = await real(s);
      return ++calls === 1 ? list : [...list, { ...list[0]!, id: made, name: 'fasteners-n-best loop gen 1' }];
    });
    await page(BenchLoop, '/benchmarks/bn_3c90/loops/bl_f3a0');
    expect(text()).not.toMatch(/dataset deleted/);
    expect(document.querySelector(`a[href$="/datasets/${made}"]`)).not.toBeNull();
  });
});

describe('the loops tab', () => {
  it('says why a loop failed under its name', async () => {
    await page(BenchmarkPage, '/benchmarks/bn_5e12?tab=loops');
    const loop = await hub.api.benchLoop('bn_5e12', 'bl_9e40');
    const row = [...document.querySelectorAll('table.loops tbody tr')].find((r) => r.textContent?.includes(loop.name))!;
    expect(row.querySelector('.cause')?.textContent).toContain(loop.error!.slice(0, 40));
  });
});

describe('an open external entry', () => {
  it('shows on its page the records received, and no passes', async () => {
    const images = (await hub.api.benchImages('bn_3c90', { limit: 1000 })).images;
    const e = await hub.api.startBenchEntry('bn_3c90', { name: 'yolo-big', contestant: { kind: 'external', open: true, predictions: images.slice(0, 120).map((i) => ({ image: i.id, boxes: [] })) } });
    await page(BenchEntry, `/benchmarks/bn_3c90/entries/${e.id}`);
    expect(text()).toMatch(/120 of 13,320 records/);
    expect(document.querySelector('.passes')).toBeNull();
  });
});

describe('the plan of a benchmark with only failed or stopped entries', () => {
  it('stays editable: only a finished or running entry freezes it', async () => {
    const e = await hub.api.startBenchEntry('bn_9f03', { name: 'gone', contestant: { kind: 'external', open: true } });
    await page(BenchmarkPage, '/benchmarks/bn_9f03?tab=conditions');
    // An open entry runs: frozen.
    expect(text()).toContain(FROZEN);
    await hub.api.stopBenchEntry('bn_9f03', e.id);
    await page(BenchmarkPage, '/benchmarks/bn_9f03?tab=conditions');
    expect(text()).not.toContain(FROZEN);
    expect(button(/^Design with an agent/)?.disabled).toBe(false);
  });
});

describe('the leaderboard', () => {
  it('ranks an entry graded on an earlier version below the current ones, unranked, and keeps the leaders readable', async () => {
    await page(BenchmarkPage, '/benchmarks/bn_3c90');
    const rows = [...document.querySelectorAll<HTMLElement>('table.board tbody tr[data-id]')];
    const ids = rows.map((r) => r.dataset.id);
    const rank = (r: HTMLElement) => r.querySelector('td.rank')?.textContent?.trim() ?? '';
    // Below every current finished entry, not ranked; the current leader keeps rank 1.
    const old = rows.find((r) => r.dataset.id === 'be_f100')!;
    expect(rank(old)).not.toMatch(/^\d+$/);
    const finishedCurrent = rows.filter((r) => /^\d+$/.test(rank(r)));
    expect(finishedCurrent.length).toBeGreaterThan(1);
    for (const r of finishedCurrent) expect(ids.indexOf(r.dataset.id)).toBeLessThan(ids.indexOf('be_f100'));
    expect(rows[0]!.dataset.id).toBe('be_f101');
    // fasteners-mini-n (0.158) is off the leaders' scale: an arrowhead at its left edge, no whiskers.
    const mini = rows.find((r) => /fasteners-mini-n/.test(r.textContent ?? ''))!;
    expect(mini.querySelector('path.off')).not.toBeNull();
    expect(mini.querySelector('line.whisker')).toBeNull();
    expect(rows[0]!.querySelector('line.whisker')).not.toBeNull();
  });

  it('marks an entry graded on an earlier version on Compare, with no difference against it', async () => {
    await page(BenchCompare, '/benchmarks/bn_3c90/compare?entries=be_f101,be_f100');
    expect([...document.querySelectorAll('.chips .chip')].map((x) => /earlier version/.test(x.textContent ?? ''))).toEqual([false, true]);
    expect([...document.querySelectorAll('.cmp tbody .dd')].map((x) => x.textContent)).not.toContain('±0');
  });
});
