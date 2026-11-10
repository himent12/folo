// @vitest-environment happy-dom
// The result screens against the in-browser mock: an entry's page in each of its states (complete, an agent's partial
// one, never reaching 90% precision, running, failed), its failure gallery's paging and large view, the compare page's
// selection and its states, the new-loop dialog's eligibility and checks, a loop's page, and the leaderboard's scale.
import { flushSync, mount, unmount, type Component } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { ApiError } from '$lib/api/client';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router.svelte';
import { toast } from '$lib/ui/toast.svelte';
import BenchCompare from '../BenchCompare.svelte';
import BenchEntry from '../BenchEntry.svelte';
import BenchLoop from '../BenchLoop.svelte';
import BenchmarkPage from '../Benchmark.svelte';
import { GALLERY_PAGE } from './entry/ImagesSection.svelte';
import NewLoopDialog from './NewLoopDialog.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

// Svelte's select binding reads the chosen option with `:checked`, which happy-dom does not match on <option>.
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
  target.remove();
  vi.restoreAllMocks();
});

async function settle(rounds = 24) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
}
async function page(comp: Component, path: string) {
  router.route = parse(`#${path}`);
  c = mount(comp as Component<Record<string, unknown>>, { target, props: {} });
  flushSync();
  await settle();
}
const text = () => document.body.textContent ?? '';
const button = (re: RegExp, root: ParentNode = document) => [...root.querySelectorAll<HTMLButtonElement>('button')].find((b) => re.test(b.textContent?.trim() ?? ''));

describe("an entry's page", () => {
  it('reads a complete entry top to bottom: the score in rows that add up, every section, a jump strip', async () => {
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f101');
    // The strip jumps to each section, and each is there.
    const links = [...document.querySelectorAll<HTMLAnchorElement>('nav.jump a')].map((a) => a.getAttribute('href'));
    expect(links).toEqual(['#score', '#accuracy', '#robustness', '#calibration', '#self-labeling', '#localization', '#images']);
    for (const id of links) expect(document.querySelector(id!)).not.toBeNull();
    // Five axes, each its points; they add up to the total written under them.
    const pts = [...document.querySelectorAll('#score .r[data-axis] .c-pts')].map((x) => Number(x.textContent!.replace('+', '')));
    expect(pts).toHaveLength(5);
    const total = Number(document.querySelector('#score .r.sum .c-pts')!.textContent);
    expect(pts.reduce((a, b) => a + b, 0)).toBeCloseTo(total, 6);
    expect(document.querySelector('#score .big')?.textContent).toBe(String(total));
    // The numbers behind the axes, in their units.
    // mAP with its interval: "± h" when symmetric as printed, else the interval itself.
    expect(text()).toMatch(/mAP50-95 0\.\d{3} (± 0\.\d{3}|\(0\.\d{3}–0\.\d{3}\))/);
    expect(text()).toMatch(/keeps \d+% of its accuracy/);
    // Where it breaks: one row per condition of the plan, worst kept first, its cells at the planned severities.
    const rows = document.querySelectorAll('#robustness .heat tbody tr');
    expect(rows).toHaveLength(12);
    expect(rows[0]!.querySelectorAll('button.cell')).toHaveLength(3);
    const means = [...rows].map((r) => Number(r.querySelector('.mean')!.textContent!.replace('%', '')));
    expect(means).toEqual([...means].sort((a, b) => a - b));
    // The calibration reads itself, and self-labeling says its threshold in words.
    expect(document.querySelector('#calibration .sentence')?.textContent).toMatch(/^When it says 0\.\d\d, it is right \d+% of the time/);
    expect(text()).toMatch(/From confidence 0\.\d\d up: [\d,]+ labels \([\d.]+ per image\), 9\d% right, \d+% of all objects/);
  });

  it('pages its images and opens one large, stepping on to the next', async () => {
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f101');
    await settle(40);
    const tiles = () => document.querySelectorAll('#images .tile:not(.sk)');
    expect(tiles()).toHaveLength(GALLERY_PAGE);
    expect(document.querySelector('#images .count')?.textContent).toBe(`${GALLERY_PAGE} of 360`);
    button(/^Show 24 more/)!.click();
    await settle(40);
    expect(tiles()).toHaveLength(2 * GALLERY_PAGE);
    // The worst first: a tile's counts come down the grid.
    (tiles()[0] as HTMLButtonElement).click();
    await settle();
    expect(document.querySelector('dialog[open] .v-pos')?.textContent).toBe('1 of 360');
    expect(document.querySelectorAll('dialog[open] .v-list li').length).toBeGreaterThan(0);
    button(/^Next/, document.querySelector('dialog[open]')!)!.click();
    await settle();
    expect(document.querySelector('dialog[open] .v-pos')?.textContent).toBe('2 of 360');
  });

  it("says in place why an agent's robustness and calibration were not measured, and shows its one operating point", async () => {
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f104');
    expect(text()).toMatch(/It ran the unchanged images only: for an agent every condition is another call per image/);
    expect(text()).toMatch(/Agents give no confidences/);
    expect(text()).toMatch(/Robustness and Calibration were not measured: the other three axes share the score/);
    expect(text()).toMatch(/It gives no confidences, so it has one operating point/);
    expect(document.querySelector('#calibration svg')).toBeNull();
  });

  it('says when its confidences never reach 90% right, with no threshold drawn', async () => {
    await page(BenchEntry, '/benchmarks/bn_7a41/entries/be_p204');
    expect(text()).toMatch(/No confidence keeps 10 labels or more of which 90% are right/);
    expect(document.querySelector('#self-labeling .verdict')?.textContent).toBe('Not ready');
    expect(document.querySelector('#self-labeling .mark')).toBeNull();
  });

  it('shows a running entry where it is, and a failed one why, with what it was asked to run', async () => {
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f108');
    expect(text()).toMatch(/Haze, severity 3/);
    expect(document.querySelectorAll('.passes .pill.done').length).toBeGreaterThan(0);
    expect(document.querySelectorAll('.passes .pill.now')).toHaveLength(1);
    expect(document.querySelector('nav.jump')).toBeNull();
    unmount(c!);
    c = undefined;
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f107');
    expect(text()).toMatch(/It failed: nothing was scored/);
    expect(text()).toMatch(/Not logged in/);
    expect(text()).toMatch(/What it was asked to run/);
  });
});

describe('the compare page', () => {
  it('lines entries up, says what is not like for like, and whether their intervals tell them apart', async () => {
    await page(BenchCompare, '/benchmarks/bn_3c90/compare?entries=be_f101,be_f104');
    await settle(40);
    expect(document.querySelectorAll('.chips .chip')).toHaveLength(2);
    expect(text()).toMatch(/Not like for like/);
    expect(text()).toMatch(/Claude Code \(sonnet\) ran 48 of 360 images/);
    expect(text()).toMatch(/fasteners-s-v1 is clearly ahead on mAP50-95/);
    // The first is the baseline; the other's values say how far they are from it.
    expect(document.querySelector('.cmp thead th.e-col .role')?.textContent).toBe('Baseline');
    expect(document.querySelectorAll('.cmp tbody tr')).toHaveLength(6);
  });

  it('adds and removes entries without leaving, the URL following', async () => {
    await page(BenchCompare, '/benchmarks/bn_3c90/compare?entries=be_f101,be_f103');
    // Recorded, not followed: the page asks for the URL; the router (tested on its own) follows it.
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const rm = document.querySelector<HTMLButtonElement>('.chip button.rm[aria-label="Remove fasteners-n-best"]')!;
    rm.click();
    expect(go).toHaveBeenCalledWith('/benchmarks/bn_3c90/compare?entries=be_f101', { replace: true });
    const add = document.querySelector<HTMLSelectElement>('.add select')!;
    add.querySelector<HTMLOptionElement>('option[value="be_f102"]')!.selected = true;
    add.dispatchEvent(new Event('change', { bubbles: true }));
    expect(go).toHaveBeenLastCalledWith('/benchmarks/bn_3c90/compare?entries=be_f101,be_f103,be_f102', { replace: true });
  });

  it('explains a link with fewer than two finished entries, with the way back', async () => {
    await page(BenchCompare, '/benchmarks/bn_3c90/compare?entries=be_f101,be_gone');
    expect(text()).toMatch(/Add one more entry to compare/);
    expect(text()).toMatch(/One entry in this link is not on Fasteners val or not finished/);
    expect([...document.querySelectorAll('a')].some((a) => /Back to the leaderboard/.test(a.textContent ?? ''))).toBe(true);
  });
});

describe('New loop', () => {
  async function dialog(teacher: string | null = null, benchId = 'bn_3c90') {
    const bench = await hub.api.bench(benchId);
    const [models, datasets, jobs, entries] = await Promise.all([hub.api.models(), hub.api.datasets(), hub.api.labelJobs(), hub.api.benchEntries(benchId)]);
    const onstarted = vi.fn();
    c = mount(NewLoopDialog as Component<Record<string, unknown>>, { target, props: { open: true, bench, models, datasets, jobs, entries, teacher, onstarted } });
    flushSync();
    await settle();
    return { onstarted };
  }
  const submit = () => document.querySelector<HTMLButtonElement>('dialog[open] button[type="submit"]')!;
  const send = async () => {
    submit().form!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await settle();
  };

  it('offers the models with the benchmark’s classes as teachers, the others off with their reason', async () => {
    await dialog();
    const sel = document.querySelector<HTMLSelectElement>('select[id$="-teacher"]')!;
    const opt = (id: string) => sel.querySelector<HTMLOptionElement>(`option[value="${id}"]`)!;
    expect(opt('m_fast_n').disabled).toBe(false);
    expect(opt('m_mini_n').disabled).toBe(true);
    expect(opt('m_mini_n').textContent).toMatch(/Knows 2 of these 8 classes/);
    expect(opt('m_coco_s').textContent).toMatch(/None of these classes/);
    // Its complete entry is generation 0, so the plan costs one entry per student.
    expect(document.querySelector('dialog[open] .s-text')?.textContent).toBe('1 generation: 1 training run of 30 epochs on up to 1,920 images, 1 benchmark entry');
    expect(text()).toMatch(/where its entry keeps [\d,]+ labels, 90% right/);
  });

  it('checks the student’s settings and the confidence, then starts the loop', async () => {
    const { onstarted } = await dialog('m_fast_n');
    const ep = document.querySelector<HTMLInputElement>('input[id$="-ep"]')!;
    ep.value = '0';
    ep.dispatchEvent(new Event('input', { bubbles: true }));
    const sz = document.querySelector<HTMLInputElement>('input[id$="-sz"]')!;
    sz.value = '650';
    sz.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    await send();
    expect(text()).toMatch(/A whole number from 1 to 1,000/);
    expect(text()).toMatch(/A multiple of 32 from 32 to 4,096/);
    expect(onstarted).not.toHaveBeenCalled();
    ep.value = '20';
    ep.dispatchEvent(new Event('input', { bubbles: true }));
    sz.value = '640';
    sz.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    await send();
    expect(onstarted).toHaveBeenCalledWith(expect.objectContaining({ status: 'running', config: expect.objectContaining({ teacher: 'm_fast_n', student: expect.objectContaining({ epochs: 20 }) }) }));
  });

  it('has the students follow the teacher picked, until a person picks their size', async () => {
    await dialog(null, 'bn_5e12');
    const teacherSel = document.querySelector<HTMLSelectElement>('select[id$="-teacher"]')!;
    const arch = () => document.querySelector<HTMLSelectElement>('select[id$="-arch"]')!.value;
    const pickTeacher = async (id: string) => {
      teacherSel.querySelector<HTMLOptionElement>(`option[value="${id}"]`)!.selected = true;
      teacherSel.dispatchEvent(new Event('change', { bubbles: true }));
      await settle();
    };
    await pickTeacher('m_shapes_s');
    expect(arch()).toBe('folo-s');
    await pickTeacher('m_shapes_e5');
    expect(arch()).toBe('folo-n');
    // A size a person picked stays.
    const a = document.querySelector<HTMLSelectElement>('select[id$="-arch"]')!;
    a.querySelector<HTMLOptionElement>('option[value="folo-m"]')!.selected = true;
    a.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    await pickTeacher('m_shapes_s');
    expect(arch()).toBe('folo-m');
  });

  it('opened, closed and opened again on the same page, still has the students’ defaults (they follow the teacher)', async () => {
    await page(BenchmarkPage, '/benchmarks/bn_3c90?tab=loops');
    const newLoop = () => [...document.querySelectorAll<HTMLButtonElement>('button')].find((b) => /^New loop$/.test(b.textContent?.trim() ?? ''))!;
    const arch = () => document.querySelector<HTMLSelectElement>('dialog[open] select[id$="-arch"]')?.value;
    const size = () => document.querySelector<HTMLInputElement>('dialog[open] input[id$="-sz"]')?.value;
    for (let k = 0; k < 2; k++) {
      newLoop().click();
      await settle();
      expect(arch(), `opening ${k + 1}`).toBe('folo-n');
      expect(size(), `opening ${k + 1}`).toBe('640');
      expect(submit().disabled).toBe(false);
      button(/^Cancel$/, document.querySelector('dialog[open]')!)!.click();
      await settle();
    }
    // A size picked the first time is not kept into the next opening.
    newLoop().click();
    await settle();
    const a = document.querySelector<HTMLSelectElement>('dialog[open] select[id$="-arch"]')!;
    a.querySelector<HTMLOptionElement>('option[value="folo-m"]')!.selected = true;
    a.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect(arch()).toBe('folo-m');
    button(/^Cancel$/, document.querySelector('dialog[open]')!)!.click();
    await settle();
    newLoop().click();
    await settle();
    expect(arch()).toBe('folo-n');
  });

  it('shows a refused start at the field it names', async () => {
    await dialog('m_fast_n');
    vi.spyOn(hub.api, 'startBenchLoop').mockRejectedValue(new ApiError(400, 'invalid_config', 'imgsz must be at least 64 for training, got 32'));
    const sz = document.querySelector<HTMLInputElement>('input[id$="-sz"]')!;
    sz.value = '32';
    sz.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    await send();
    const field = sz.closest('.field') ?? sz.parentElement!.parentElement!;
    expect(document.querySelector('dialog[open]')?.textContent).toMatch(/imgsz must be at least 64 for training, got 32/);
    expect(field.textContent ?? document.body.textContent).toMatch(/imgsz must be at least 64/);
    expect(document.querySelector('dialog[open] .error-state, dialog[open] [role=alert]')?.textContent ?? '').not.toMatch(/Could not start the loop/);
  });

  it('reads a good teacher’s labels off its entry, quietly', async () => {
    await dialog('m_fast_n');
    const fc = document.querySelector('dialog[open] .fc')!;
    expect(fc.getAttribute('data-forecast')).toBe('ok');
    expect(fc.textContent).toMatch(/^\s*At its own threshold, 0\.\d\d, 9\d% of its labels are right and they cover \d+% of all objects \((partly ready|ready)\): about [\d,]+ of the pool’s 1,920 images would get a box\./);
    expect(submit().textContent).toMatch(/Start loop/);
  });

  it('requires a confidence for a teacher whose entry found no threshold, and says what each one keeps and costs', async () => {
    // shapes-n-early on Shapes val: 90% right only on a handful of labels, too few for a threshold. A loop never
    // guesses a confidence, so one is required here before it can start.
    await dialog('m_shapes_e5', 'bn_5e12');
    const fc = () => document.querySelector('dialog[open] .fc')!;
    expect(fc().getAttribute('data-forecast')).toBe('none');
    expect(fc().getAttribute('role')).toBe('alert');
    expect(fc().textContent).toMatch(/Its entry found no threshold: choose a confidence/);
    expect(document.querySelector<HTMLInputElement>('dialog[open] [role=switch], dialog[open] input[type=checkbox]')?.disabled).toBe(true);
    const conf = document.querySelector<HTMLInputElement>('input[id$="-conf"]')!;
    expect(conf).not.toBeNull();
    expect(conf.value).toBe('');
    expect(text()).toMatch(/Required for this teacher\./);
    expect(submit().disabled).toBe(true);
    expect(document.querySelector('dialog[open] .why')?.textContent).toBe('Choose a confidence: its entry found no threshold');
    expect(submit().textContent).toMatch(/Start loop/);
    expect(text()).not.toMatch(/0\.50: its entry/);
    const type = async (v: string) => {
      conf.value = v;
      conf.dispatchEvent(new Event('input', { bubbles: true }));
      await settle();
    };
    // A high confidence keeps a handful: told the loop would stop at labeling, Start reads "Start anyway".
    await type('0.9');
    expect(fc().getAttribute('data-forecast')).toBe('stops');
    expect(fc().textContent).toMatch(/Too few labels: the loop would stop at labeling/);
    expect(submit().disabled).toBe(false);
    expect(submit().textContent).toMatch(/Start anyway/);
    // A lower one: what it keeps and what it costs in precision, read off the curve.
    await type('0.30');
    expect(fc().getAttribute('data-forecast')).toBe('ok');
    expect(fc().textContent).toMatch(/^\s*From 0\.\d+, \d+% of its labels are right and they cover \d+% of all objects: about [\d,]+ of the pool’s 800 images would get a box\./);
    expect(submit().textContent).toMatch(/Start loop/);
    // Live: a lower confidence still, fewer right.
    const right = (t: string) => Number(/(\d+)% of its labels are right/.exec(t)![1]);
    const before = right(fc().textContent!);
    conf.value = '0.05';
    conf.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
    expect(fc().textContent).toMatch(/^\s*From 0\.050, /);
    expect(right(fc().textContent!)).toBeLessThan(before);
  });
});

describe("a loop's page", () => {
  it('states its question and verdict, and lists each generation', async () => {
    await page(BenchLoop, '/benchmarks/bn_5e12/loops/bl_07d1');
    expect(text()).toMatch(/^.*Can shapes-n-best label the training split of shapes well enough, by itself, to train a better folo-s, generation after generation\?/);
    expect(document.querySelector('.v-word')?.textContent).toBe('Improved');
    expect(document.querySelector('.v-text')?.textContent).toMatch(/^Generation 3 \(0\.\d{3}\) is above the first teacher’s 95% interval/);
    expect(document.querySelectorAll('table.gens tbody tr')).toHaveLength(4);
  });

  it('at generation 0 shows one step, the benchmark grading the teacher, and nothing done', async () => {
    hub.api = new MockApi('bench-loop-gen0', 0);
    await page(BenchLoop, '/benchmarks/bn_3c90/loops/bl_g0a1');
    const steps = [...document.querySelectorAll('.steps li')].map((li) => `${li.querySelector('.s-name')!.textContent}: ${li.className.replace(/\s*svelte-\S+/g, '').trim()}`);
    expect(steps).toEqual(['Benchmark grades the teacher: now']);
    expect(text()).toMatch(/Benchmarking the teacher/);
    expect(text()).toMatch(/Its entry: fasteners-n-f16, waiting for the GPU\./);
    expect(text()).toMatch(/No generation graded yet/);
    // Gen 1 and 2 are to come.
    expect(document.querySelectorAll('table.gens tbody tr.future')).toHaveLength(2);
  });

  it('says why a loop starved: the server’s message, and its generation with no labels', async () => {
    await page(BenchLoop, '/benchmarks/bn_5e12/loops/bl_9e40');
    expect(document.querySelector('[role=alert]')?.textContent).toMatch(/It failed at generation 1/);
    expect(text()).toMatch(/'shapes-n-early' never reaches 90% precision on at least 10 labels on this benchmark, so there is no confidence to label the pool from\./);
    expect(text()).toMatch(/Set min_confidence: its entry's self-labeling chart/);
    const row = document.querySelectorAll('table.gens tbody tr')[1]!;
    expect(row.textContent).toMatch(/Not trained/);
    expect(row.textContent).toMatch(/None: the loop failed here/);
    expect(document.querySelectorAll('table.gens tbody tr')[2]!.textContent).toMatch(/Not run/);
    expect(document.querySelector('.steps')).toBeNull();
  });

  it('writes each generation’s labels as one line', async () => {
    await page(BenchLoop, '/benchmarks/bn_5e12/loops/bl_07d1');
    const lines = [...document.querySelectorAll('table.gens tbody .ln')].map((x) => x.textContent);
    expect(lines).toHaveLength(3);
    for (const l of lines) expect(l).toMatch(/^kept [\d,]+ boxes on [\d,]+ of 800 pool images from confidence 0\.\d\d$/);
  });

  it('shows the step a running loop is on, and offers Stop', async () => {
    await page(BenchLoop, '/benchmarks/bn_3c90/loops/bl_f3a0');
    const steps = [...document.querySelectorAll('.steps li')].map((li) => li.className.replace(/\s*svelte-\S+/g, '').trim());
    expect(steps).toEqual(['done', 'now', 'next']);
    expect(button(/^Stop loop/)).toBeTruthy();
  });

  it('opens from the loops’ old links (`?tab=loops&loop=`)', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    await page(BenchmarkPage, '/benchmarks/bn_5e12?tab=loops&loop=bl_07d1');
    expect(go).toHaveBeenCalledWith('/benchmarks/bn_5e12/loops/bl_07d1', { replace: true });
  });
});

describe('the loops tab', () => {
  it('says why a loop failed under its name, and what a running one does for which generation', async () => {
    await page(BenchmarkPage, '/benchmarks/bn_5e12?tab=loops');
    const row = [...document.querySelectorAll('table.loops tbody tr')].find((r) => /shapes-n-early loop/.test(r.textContent ?? ''))!;
    expect(row.querySelector('.cause')?.textContent).toMatch(/^'shapes-n-early' never reaches 90% precision on at least 10 labels/);
    unmount(c!);
    c = undefined;
    await page(BenchmarkPage, '/benchmarks/bn_3c90?tab=loops');
    expect(text()).toMatch(/Training gen 1/);
  });
});

describe('the failure gallery', () => {
  it('gives its tiles the commonest shape of its photos, each photo whole on its well', async () => {
    await page(BenchEntry, '/benchmarks/bn_7a41/entries/be_p201');
    await settle(40);
    const tiles = [...document.querySelectorAll<HTMLElement>('#images .tile:not(.sk) .img')];
    expect(tiles.length).toBeGreaterThan(0);
    // One shape for every tile (4:3, the pond's commonest), never the first photo's.
    expect(new Set(tiles.map((t) => Number(/aspect-ratio:\s*([\d.]+)/.exec(t.getAttribute('style') ?? '')?.[1]).toFixed(2)))).toEqual(new Set(['1.33']));
  });
});

describe('an open external entry', () => {
  it('reads on the leaderboard and its page as receiving predictions, with the records received', async () => {
    const images = (await hub.api.benchImages('bn_3c90', { limit: 1000 })).images;
    const e = await hub.api.startBenchEntry('bn_3c90', { name: 'yolo-big', contestant: { kind: 'external', open: true, predictions: images.slice(0, 120).map((i) => ({ image: i.id, boxes: [] })) } });
    await page(BenchmarkPage, '/benchmarks/bn_3c90');
    const row = [...document.querySelectorAll('table.board tbody tr')].find((r) => /yolo-big/.test(r.textContent ?? ''))!;
    expect(row.textContent).toMatch(/Receiving/);
    expect(row.querySelector('.prog-text')?.textContent).toMatch(/^Receiving predictions\s*·?\s*120 of 13,320 records$/);
    unmount(c!);
    c = undefined;
    await page(BenchEntry, `/benchmarks/bn_3c90/entries/${e.id}`);
    expect(text()).toMatch(/Receiving predictions/);
    expect(text()).toMatch(/120 of 13,320 records, sent in parts/);
    expect(document.querySelector('.passes')).toBeNull();
  });
});

describe('reading a chart with the keyboard', () => {
  /** Focuses the chart in `root`, presses → twice, and gives what its live region said. */
  async function speak(root: string) {
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

  it('announces a row at every stop on the self-labeling, calibration and Compare charts', async () => {
    // The plots draw once they have a width; happy-dom lays nothing out, so every element reports one.
    const was = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth');
    Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 600 });
    try {
      await announces();
    } finally {
      if (was) Object.defineProperty(HTMLElement.prototype, 'clientWidth', was);
    }
  });

  async function announces() {
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f101');
    for (const s of await speak('#self-labeling')) expect(s).toMatch(/^From [01]\.\d+ up: [\d,]+ labels?, \d+% right, [\d.]+% of all objects$/);
    for (const s of await speak('#calibration')) expect(s.length).toBeGreaterThan(5);
    unmount(c!);
    c = undefined;
    await page(BenchCompare, '/benchmarks/bn_3c90/compare?entries=be_f101,be_f103');
    await settle(40);
    // Compare's calibration overlay (its self-labeling trade-off has no stops: its table is what is read).
    for (const s of await speak('.pair > :nth-child(1)')) expect(s).toMatch(/\d/);
  }
});

describe('the compare page when a result fails', () => {
  it('says so with a retry instead of skeletons, and loads it on retry', async () => {
    const real = hub.api.benchResult.bind(hub.api);
    let fail = true;
    vi.spyOn(hub.api, 'benchResult').mockImplementation(async (b, e, s) => {
      if (fail && e === 'be_f103') throw new ApiError(500, 'internal', 'the result file could not be read');
      return real(b, e, s);
    });
    await page(BenchCompare, '/benchmarks/bn_3c90/compare?entries=be_f101,be_f103');
    await settle(40);
    expect(text()).toMatch(/Could not load the result of fasteners-n-best/);
    fail = false;
    button(/^Retry|^Try again/)!.click();
    await settle(40);
    expect(text()).not.toMatch(/Could not load the result/);
    expect(document.querySelector('.pair')).not.toBeNull();
  });
});

describe('a live loop page', () => {
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
    await settle();
    expect(text()).not.toMatch(/dataset deleted/);
    expect(document.querySelector(`a[href$="/datasets/${made}"]`)).not.toBeNull();
  });
});

describe('the plan of a benchmark with only failed or stopped entries', () => {
  it('stays editable: only a finished or running entry freezes it', async () => {
    const e = await hub.api.startBenchEntry('bn_9f03', { name: 'gone', contestant: { kind: 'external', open: true } });
    await page(BenchmarkPage, '/benchmarks/bn_9f03?tab=conditions');
    // An open entry runs: frozen.
    expect(text()).toMatch(/This benchmark has finished or running entries; its plan is fixed/);
    unmount(c!);
    c = undefined;
    await hub.api.stopBenchEntry('bn_9f03', e.id);
    await page(BenchmarkPage, '/benchmarks/bn_9f03?tab=conditions');
    expect(text()).not.toMatch(/its plan is fixed/);
    expect(button(/^Design with an agent/)?.disabled).toBe(false);
  });
});

describe('an entry graded on an earlier version of the benchmark', () => {
  it('is marked and ranked below the current ones on the leaderboard, said on its page, and called out on Compare', async () => {
    await page(BenchmarkPage, '/benchmarks/bn_3c90');
    const rows = [...document.querySelectorAll<HTMLElement>('table.board tbody tr[data-id]')];
    const ids = rows.map((r) => r.dataset.id);
    const old = rows.find((r) => r.dataset.id === 'be_f100')!;
    expect(old.querySelector('td.ran')?.textContent).toMatch(/earlier version of this benchmark/);
    expect(old.querySelector('td.rank')?.textContent?.trim()).toMatch(/^—not ranked: graded on an earlier version/);
    expect(old.querySelector('.stale-note')?.textContent).toMatch(/not on this scale/);
    // Below every current finished entry, above the unfinished ones; the current leader keeps rank 1.
    const finishedCurrent = rows.filter((r) => r.dataset.id !== 'be_f100' && r.querySelector('td.rank')?.textContent?.trim().match(/^\d+$/));
    for (const r of finishedCurrent) expect(ids.indexOf(r.dataset.id)).toBeLessThan(ids.indexOf('be_f100'));
    expect(rows[0]!.dataset.id).toBe('be_f101');
    unmount(c!);
    c = undefined;
    await page(BenchEntry, '/benchmarks/bn_3c90/entries/be_f100');
    expect(text()).toMatch(/Graded on an earlier version of this benchmark/);
    expect(text()).toMatch(/Run it again to compare/);
    expect(text()).not.toMatch(/\d(st|nd|rd|th) of \d+ on this benchmark/);
    unmount(c!);
    c = undefined;
    await page(BenchCompare, '/benchmarks/bn_3c90/compare?entries=be_f101,be_f100');
    await settle(40);
    expect(text()).toMatch(/fasteners-s-v1 was graded on an earlier version of this benchmark/);
    expect(text()).toMatch(/not all on its current fingerprint/);
    // Marked on its chip, and no difference against it (they do not compare).
    expect([...document.querySelectorAll('.chips .chip')].map((x) => x.textContent)).toEqual([expect.not.stringMatching(/earlier version/), expect.stringMatching(/earlier version/)]);
    expect([...document.querySelectorAll('.cmp tbody .dd')].map((x) => x.textContent)).not.toContain('±0');
  });
});

describe('a partial entry on the leaderboard', () => {
  it('names the reason: calibration not measured, for a complete scored entry with no confidence from 0.1', async () => {
    await page(BenchmarkPage, '/benchmarks/bn_7a41');
    const row = [...document.querySelectorAll('table.board tbody tr')].find((r) => /yolo26n-pond-raw/.test(r.textContent ?? ''))!;
    expect(row.querySelector('td.ran')?.textContent).toMatch(/calibration not measured/);
    expect(row.textContent).not.toMatch(/some conditions left out/);
  });
});

describe('the leaderboard', () => {
  it('keys its axes in place and keeps the leaders readable, marking an entry far behind off the scale', async () => {
    await page(BenchmarkPage, '/benchmarks/bn_3c90');
    expect([...document.querySelectorAll('.axes-key .ak')].map((x) => x.textContent)).toEqual(['A', 'R', 'S', 'C', 'L']);
    expect(text()).toMatch(/The bars by each score, left to right: Accuracy, Robustness, Self-labeling, Calibration, Localization/);
    // fasteners-mini-n (0.158) is off the leaders' scale: an arrowhead at its left edge, no whiskers.
    const mini = [...document.querySelectorAll('table.board tbody tr')].find((r) => /fasteners-mini-n/.test(r.textContent ?? ''))!;
    expect(mini.querySelector('path.off')).not.toBeNull();
    expect(mini.querySelector('line.whisker')).toBeNull();
    expect(document.querySelectorAll('table.board tfoot .tk').length).toBeGreaterThanOrEqual(3);
  });
});
