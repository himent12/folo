// @vitest-environment happy-dom
// The result screens against the in-browser mock: an entry's page in each of its states (complete, an agent's partial
// one, never reaching 90% precision, running, failed), its failure gallery's paging and large view, the compare page's
// selection and its states, the new-loop dialog's eligibility and checks, a loop's page, and the leaderboard's scale.
import { flushSync, mount, unmount, type Component } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
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
    expect(text()).toMatch(/From confidence 0\.\d\d up, \d+% of its labels are right and they cover \d+% of all objects/);
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
    expect(text()).toMatch(/At no confidence are 90% of its labels right/);
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
    expect(text()).toMatch(/where its entry’s labels are 90% right/);
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

  it('reads a good teacher’s labels off its entry, quietly', async () => {
    await dialog('m_fast_n');
    const fc = document.querySelector('dialog[open] .fc')!;
    expect(fc.getAttribute('data-forecast')).toBe('ok');
    expect(fc.textContent).toMatch(/^\s*At its own threshold, 0\.\d\d, 9\d% of its labels are right and they cover \d+% of all objects \((partly ready|ready)\): about [\d,]+ of the pool’s 1,920 images would get a box\./);
    expect(submit().textContent).toMatch(/Start loop/);
  });

  it('tells, before it starts, that a not-ready teacher would label too little, with the lower confidence as the way out', async () => {
    // shapes-n-early on Shapes val: 90% right only from ~0.75, where it keeps almost nothing.
    await dialog('m_shapes_e5', 'bn_5e12');
    const fc = () => document.querySelector('dialog[open] .fc')!;
    expect(fc().getAttribute('data-forecast')).toBe('stops');
    expect(fc().getAttribute('role')).toBe('alert');
    expect(fc().textContent).toMatch(/Too few labels: the loop would stop at labeling/);
    expect(fc().textContent).toMatch(/\(not ready\): about \d of the pool’s 800 images would get a box\. A loop stops when fewer than 10 pool images get a box\./);
    // Told at the button too, which says what pressing it means; it is not taken away.
    expect(document.querySelector('dialog[open] .why')?.textContent).toBe('Its labels would be too few: the loop would stop at labeling');
    expect(submit().textContent).toMatch(/Start anyway/);
    expect(submit().disabled).toBe(false);
    // The way out: the confidence field, set lower, with what that costs read off the curve.
    button(/^Keep labels from a lower confidence/)!.click();
    await settle();
    const conf = document.querySelector<HTMLInputElement>('input[id$="-conf"]')!;
    expect(conf).not.toBeNull();
    expect(Number(conf.value)).toBeLessThan(0.75);
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
    expect(text()).toMatch(/The teacher 'shapes-n-early' kept \d+ box(es)? on \d of the pool's 800 images at confidence 0\.\d+: too few to train a student on \(at least 10 images\)\./);
    expect(text()).toMatch(/A lower min_confidence keeps more of its detections/);
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
    expect(row.querySelector('.cause')?.textContent).toMatch(/^The teacher 'shapes-n-early' kept .* too few to train a student on/);
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
