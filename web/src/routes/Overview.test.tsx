// @vitest-environment happy-dom
// Overview against the mock server: it reads the run list the sidebar reads (shell/runList), so the Training card
// never says "Nothing is training" beside the sidebar's "Training" (shell F13), and Best models catches up when a run
// ends (runs F4). That list is the server's in use: after a switch, nothing of the server before shows (review R4).
/// <reference types="node" />
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { flush } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { ApiError, type RunEventHandlers } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { deviceName } from '$lib/format';
import type { Run, SystemInfo } from '$lib/api/types';
import { parse, router } from '$lib/router';
import RunWatch from '$lib/shell/RunWatch.tsx';
import SidebarLive from '$lib/shell/SidebarLive';
import { activeRuns, watched } from '$lib/shell/runList';
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

  it('never says there are no runs while the only run is training (issue #328)', async () => {
    const run = (await idleRuns()).find((r) => r.status === 'completed')!;
    watched.runs = [{ ...run, status: 'running' }];
    open(new MockApi('loading'));
    await training((t) => t.includes(run.name), 'the run');
    const recent = target.querySelector('#recent-runs')!.textContent ?? '';
    expect(recent).not.toContain('No runs yet');
    expect(recent).toContain('No other runs');
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

    // The datasets arrive without the first run's: it may be new, so the list is read once more; still without it, it
    // was removed, and goes by the name the run recorded.
    const all = await new MockApi('normal', 0).datasets();
    const done = runs.find((r) => r.status === 'completed')!;
    const first = answer;
    first(all.filter((d) => d.id !== done.config.dataset_id));
    await vi.waitFor(() => expect(answer).not.toBe(first));
    expect(recent().textContent).not.toContain('(removed)');
    answer(all.filter((d) => d.id !== done.config.dataset_id));
    await vi.waitFor(() => {
      flush();
      expect(recent().textContent).toContain(`${done.dataset!.name} (removed)`);
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

  it("after a switch to another server, shows that server's failure, never the runs of the one before (review R4)", async () => {
    // Server A: the shell watches its list, with a run training, and the sidebar shows it.
    const runs = await idleRuns();
    const run = { ...runs.find((r) => r.status === 'completed')!, name: 'only-on-server-a', status: 'running' as const };
    const a = new MockApi('normal', 0);
    a.runs = async () => runs.map((r) => (r.id === run.id ? run : r));
    hub.api = a;
    hub.mode = 'mock';
    const shell = document.body.appendChild(document.createElement('div'));
    const view = render(
      () => (
        <>
          <RunWatch />
          <SidebarLive collapsed={false} />
        </>
      ),
      shell,
    );
    onTestFinished(() => {
      view.unmount();
      shell.remove();
    });
    await vi.waitFor(() => {
      flush();
      expect(shell.textContent).toContain(run.name);
    });

    // Settings saves server B, which answers its health check but not its run list.
    const b = new MockApi('normal', 0);
    b.runs = () => Promise.reject(new ApiError(500, 'internal', 'the run index could not be read'));
    hub.api = b;
    hub.epoch++;
    flush();
    expect(watched.runs).toBeUndefined();
    expect(shell.textContent).not.toContain(run.name);
    open(b);
    await vi.waitFor(() => {
      flush();
      expect(target.textContent).toContain('Could not load recent runs');
    });
    await vi.waitFor(() => {
      flush();
      expect(target.textContent).toContain('Could not check what is training');
    });
    expect(target.textContent).not.toContain(run.name);
    expect(shell.textContent).not.toContain(run.name);
  });

  it('asks once for the run list and the system info on a cold load, with the shell and the live card (#309)', async () => {
    vi.useFakeTimers(); // no poll can fire: the clock moves only when this test says so
    onTestFinished(() => void vi.useRealTimers());
    const api = new MockApi('normal', 0);
    // The run list answers when this test says: reads a turn apart are not joined by `together`, as on a real load.
    let answer!: () => void;
    const held = new Promise<void>((open) => (answer = open));
    const list = api.runs.bind(api);
    const runs = vi.spyOn(api, 'runs').mockImplementation(async () => {
      await held;
      return list();
    });
    const system = vi.spyOn(api, 'system');
    hub.api = api;
    hub.mode = 'mock';
    router.route = parse('#/');
    // The shell first, as on a load: the screen's chunk arrives while the shell's read is out.
    const shell = document.body.appendChild(document.createElement('div'));
    const view = render(() => <RunWatch />, shell);
    onTestFinished(() => {
      view.unmount();
      shell.remove();
    });
    flush();
    await vi.advanceTimersByTimeAsync(0);
    expect(runs).toHaveBeenCalledTimes(1);
    target = document.createElement('div');
    document.body.appendChild(target);
    c = render(() => <Overview />, target);
    flush();
    await vi.advanceTimersByTimeAsync(0);
    expect(system).toHaveBeenCalledTimes(1);
    // The list lands, the live card and its machine foot mount: a turn after the page asked for the system info.
    answer();
    const foot = () => target.querySelector('footer[aria-label="Machine"]');
    for (let i = 0; i < 100 && !foot(); i++) {
      await vi.advanceTimersByTimeAsync(0);
      flush();
    }
    expect(foot(), 'the live card and its machine foot').not.toBeNull();
    expect(runs).toHaveBeenCalledTimes(1);
    expect(system).toHaveBeenCalledTimes(1);
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
      const li = target.querySelector('#datasets li.card[data-id]');
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
    // v-shell N12: the benchmark's row says how far its entry is, as the labeling row does.
    const [entry] = await api.benchEntries(bench!.id);
    api.benchEntries = async () => [{ ...entry!, name: 'folo-n-best', status: 'running', progress: { done: 1479, total: 3219, condition: null, severity: null } }];
    open(api);
    const strip = await vi.waitFor(() => {
      flush();
      const el = target.querySelector('.activity');
      if (!el?.textContent?.includes('folo-n-best · 45%')) throw new Error('still waiting for the strip');
      return el;
    });
    expect(strip.textContent).toContain(bench!.name);
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

  // v-shell N11: a server just started answers /api/system seconds late, and the line popped in under the buttons.
  it('keeps the labeling line in its place while the server says what it has', async () => {
    const api = new MockApi('empty', 0);
    let answer!: () => void;
    const system = api.system.bind(api);
    api.system = () => new Promise<SystemInfo>((ok) => (answer = () => void system().then(ok)));
    open(api);
    const line = await vi.waitFor(() => {
      flush();
      const p = target.querySelector<HTMLElement>('.onboard .ob-alt');
      if (!p) throw new Error('still waiting for the first-run card');
      return p;
    });
    expect([line.classList.contains('held'), line.getAttribute('aria-hidden'), line.querySelector('a')]).toEqual([true, 'true', null]);
    answer();
    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.onboard .ob-alt:not(.held) a')?.textContent).toBe('Label them with an agent');
    });
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

  it('refetches Best models when a run ends or is deleted, and not on an unchanged list', async () => {
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

    // Deleting the run takes its models with it.
    watched.runs = list.filter((r) => r.id !== run.id);
    flush();
    expect(models).toHaveBeenCalledTimes(3);
  });

  it('names a dataset made after the page loaded, and lists it, rather than calling it removed (sweep 3 F4)', async () => {
    const runs = await idleRuns();
    const all = await new MockApi('normal', 0).datasets();
    const run = runs.find((r) => r.status === 'completed')!;
    const made = all.find((d) => d.id === run.config.dataset_id)!;
    // The first read predates the dataset (a loop wrote it a moment later); the server has it from then on.
    let reads = 0;
    const api = new MockApi('normal', 0);
    api.datasets = async () => (++reads === 1 ? all.filter((d) => d.id !== made.id) : all);
    watched.runs = runs;
    open(api);
    const recent = () => target.querySelector('#recent-runs tbody')?.textContent ?? '';
    await vi.waitFor(() => {
      flush();
      expect(reads).toBe(2);
      expect(target.querySelector('#datasets')!.textContent).toContain(made.name);
    });
    expect(recent()).toContain(made.name);
    expect(recent()).not.toContain('(removed)');
    // Asked about once: the next poll of the runs does not read the datasets again.
    watched.runs = [...runs];
    flush();
    expect(reads).toBe(2);
  });

  it('shows a loop at work as a loop, with what it does now (labeling for its first student), opening the loop (sweep 3 F5)', async () => {
    watched.runs = await idleRuns();
    const api = new MockApi('normal', 0);
    const benches = await api.benches();
    const loops = await Promise.all(benches.map((x) => api.benchLoops(x.id)));
    const bench = benches[loops.findIndex((l) => l.length > 0)];
    const loop = loops.find((l) => l.length > 0)![0];
    api.labelJobs = async () => [];
    api.benches = async () => [{ ...bench!, active: true }];
    api.benchEntries = async () => [];
    const gen1 = { ...loop!.generations[0]!, generation: 1, labels: null, run_id: null, model_id: null, model_name: null, entry_id: null, score: null, map50_95: null, ci95: null };
    api.benchLoops = async () => [{ ...loop!, status: 'running', phase: 'labeling', finished: null, verdict: null, generations: [loop!.generations[0]!, gen1] }];
    open(api);
    const strip = await vi.waitFor(() => {
      flush();
      const el = target.querySelector('.activity');
      if (!el?.textContent?.includes(loop!.name)) throw new Error('still waiting for the loop');
      return el;
    });
    expect(strip.textContent).toContain('Labeling for gen 1');
    expect(strip.textContent).not.toContain('An entry or a loop is running');
    const links = [...strip.querySelectorAll('a')].map((a) => a.getAttribute('href'));
    expect(links).toEqual([`#/benchmarks/${bench!.id}/loops/${loop!.id}`]);
  });

  it('shows the first-run card when the only models are published weights (sweep 3 F6)', async () => {
    const api = new MockApi('empty', 0);
    const [m] = await new MockApi('normal', 0).models();
    api.models = async () => [{ ...m!, id: 'folo-n-coco', name: 'folo-n-coco', run_id: null, source: 'published' }];
    open(api);
    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.onboard')).not.toBeNull();
    });
  });

  it('counts the datasets past the sixth apart from the models it cannot rank (sweep 3 F7)', async () => {
    const api = new MockApi('normal', 0);
    const [d0] = await api.datasets();
    const [r0] = await idleRuns();
    const [m0] = await api.models();
    const n = 9;
    const datasets = Array.from({ length: n }, (_, i) => ({ ...d0!, id: `ds_t${i}`, name: `set ${i}` }));
    const runs = datasets.map((d, i) => ({ ...r0!, id: `run_t${i}`, name: `run ${i}`, status: 'completed' as const, config: { ...r0!.config, dataset_id: d.id }, dataset: { name: d.name, classes: d.classes } }));
    const models = [
      ...runs.map((r, i) => ({ ...m0!, id: `m_t${i}`, name: `${r.name}-best`, run_id: r.id, source: 'run' as const })),
      { ...m0!, id: 'folo-n-coco', name: 'folo-n-coco', run_id: null, source: 'published' as const },
    ];
    api.datasets = async () => datasets;
    api.dataset = async (id: string) => datasets.find((d) => d.id === id)!;
    api.models = async () => models;
    watched.runs = runs;
    open(api);
    const note = await vi.waitFor(() => {
      flush();
      const p = target.querySelector('#best-models .unranked');
      if (!p) throw new Error('still waiting for Best models');
      return p.textContent!;
    });
    expect(note).toBe('3 more datasets in All models. 1 model trained elsewhere (imported or downloaded) is not ranked.');
  });

  it('says a cloud instance whose termination is not confirmed may still be billed (studio-cloud handover)', async () => {
    watched.runs = await idleRuns();
    const api = new MockApi('normal', 0);
    api.labelJobs = async () => [];
    api.benches = async () => [];
    const [inst] = await new MockApi('cloud-trouble', 0).cloudInstances();
    const stuck = { ...inst!, status: 'terminating' as const, error: 'the provider did not answer (502)' };
    api.cloudInstances = async () => [stuck, { ...stuck, id: 'ci_gone', name: 'gone', status: 'terminated' as const, error: null }];
    open(api);
    const strip = await vi.waitFor(() => {
      flush();
      const el = target.querySelector('.activity');
      if (!el?.textContent?.includes(stuck.name)) throw new Error('still waiting for the instance');
      return el;
    });
    expect(strip.textContent).toContain('Termination not confirmed: it may still be billed');
    expect(strip.textContent).not.toContain('gone');
    expect(strip.querySelector('a')!.getAttribute('href')).toBe('#/settings?section=cloud');
  });

  it('says what is there in its header facts line, beside New run (design/UNIFIED.md §4 Overview)', async () => {
    watched.runs = await idleRuns();
    const api = new MockApi('normal', 0);
    open(api);
    const [ds, models] = await Promise.all([api.datasets(), api.models()]);
    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toBe(`${ds.length} datasets · ${models.length} models · nothing training`);
    });
    const primary = [...target.querySelectorAll<HTMLAnchorElement>('header a')].find((a) => a.textContent?.trim() === 'New run');
    expect(primary?.getAttribute('href')).toBe('#/runs/new');
  });

  it('lists Best models as single-line rows (Dataset, Best model, mAP50-95) and Datasets as the Datasets page cards', async () => {
    watched.runs = await idleRuns();
    const api = new MockApi('normal', 0);
    open(api);
    const table = await vi.waitFor(() => {
      flush();
      const t = target.querySelector('#best-models table[aria-label="Best models"]');
      if (!t?.querySelector('tbody tr') || !target.querySelector('#datasets li.card[data-id]')) throw new Error('still waiting for the lists');
      return t;
    });
    expect([...table.querySelectorAll('thead th')].map((th) => th.textContent)).toEqual(['Dataset', 'Best model', 'mAP50-95']);
    // One line per evaluation dataset: no meta line under the name.
    const row = table.querySelector('tbody tr')!;
    expect(row.querySelectorAll('td')).toHaveLength(3);
    expect(row.querySelector('td a.name')?.getAttribute('href')).toMatch(/^#\/models\//);
    // The cards, one row of up to four.
    const datasets = await api.datasets();
    expect(target.querySelectorAll('#datasets li.card[data-id]')).toHaveLength(Math.min(4, datasets.length));
    // From 1280 px the panel takes its third but at least 380 px (design/UNIFIED.md §4 Overview); in a narrow panel the
    // dataset's name is cut last (its whole name in the tooltip), never wrapped, and only the split tag goes under it.
    expect(row.querySelector('td.ds .ds-name')).not.toBeNull();
    const src = (f: string) => readFileSync(resolve(process.cwd(), f), 'utf8');
    expect(src('src/routes/Overview.tsx')).toMatch(/@media \(min-width: 1280px\) \{[^@]*grid-template-columns: minmax\(0, 2fr\) minmax\(380px, 1fr\);/);
    const narrow = src('src/routes/overview/BestModels.tsx').match(/@container \(max-width: 479\.98px\) \{([\s\S]*?)\n {2}\}/)![1]!;
    expect(narrow).toContain('.ds-line');
    expect(narrow).not.toMatch(/white-space: normal|overflow-wrap/);
  });

  it('shows the live run as one panel: the rail under its header, four tiles beside the chart, the machine at its foot', async () => {
    const api = new MockApi('normal', 0);
    const running = (await api.runs()).find((r) => r.status === 'running')!;
    open(api);
    const card = await vi.waitFor(() => {
      flush();
      const el = target.querySelector('.live .card');
      if (!el?.querySelector('.foot') || !el.querySelector('.legend')) throw new Error('still waiting for the live panel');
      return el;
    });
    expect(card.querySelector('header h2')?.textContent).toBe(running.name);
    expect(card.querySelector('header .rail[role=progressbar]')).not.toBeNull();
    expect([...card.querySelectorAll('.kpis .tile .label')].map((l) => l.textContent)).toEqual(['Epoch', 'mAP50-95 · latest', 'Throughput', 'Time left']);
    expect(card.querySelector('.kpis .tile .sub')?.textContent).toMatch(/^\d+% complete$/);
    // The run's device in the meta line, whatever the foot's readings.
    expect(card.querySelector('header .meta')?.textContent).toContain(deviceName(running.device));
    const sys = await api.system();
    expect(card.querySelector('.foot')?.textContent).toContain(`CPU · ${sys.cpu.threads} threads`);
    expect(card.querySelector('.foot')?.textContent).toContain('Memory');
    // The chart in the series hues: mAP50-95 and mAP50, with the best in the legend.
    expect([...card.querySelectorAll('.legend li .name')].map((n) => n.textContent)).toEqual(['mAP50-95', 'mAP50', 'best']);
  });

  it("names a cloud run's GPU in its meta line and has no machine foot (this server does not read that machine)", async () => {
    const api = new MockApi('cloud-training', 0);
    const runs = await api.runs();
    const cloud = runs.find((r) => r.id === 'run_c7b1')!;
    watched.runs = runs.map((r) => (r.id === cloud.id ? r : r.status === 'running' || r.status === 'queued' ? { ...r, status: 'completed', queue_position: null } : r));
    open(api);
    const card = await vi.waitFor(() => {
      flush();
      const el = target.querySelector('.live .card');
      if (el?.querySelector('header h2')?.textContent !== cloud.name || !el.querySelector('.legend')) throw new Error('still waiting for the cloud run');
      return el;
    });
    expect(card.querySelector('header .meta')?.textContent).toContain('RTX 4090');
    await new Promise((ok) => setTimeout(ok, 50));
    flush();
    expect(card.querySelector('.foot')).toBeNull();
  });

  it("names a CPU run's processor in its meta line, and still when the machine's readings are gone", async () => {
    const api = new MockApi('normal', 0);
    const runs = await api.runs();
    const running = runs.find((r) => r.status === 'running')!;
    const cpu: Run = { ...running, device: 'AMD Ryzen 9 7950X (cpu)', config: { ...running.config, device: 'cpu' } };
    const get = api.run.bind(api);
    api.run = (id, ...rest) => (id === cpu.id ? Promise.resolve(cpu) : get(id, ...rest));
    api.system = () => Promise.reject(new Error('down'));
    watched.runs = runs.map((r) => (r.id === cpu.id ? cpu : r));
    open(api);
    const card = await vi.waitFor(() => {
      flush();
      const el = target.querySelector('.live .card');
      if (!el?.querySelector('.legend')) throw new Error('still waiting for the live panel');
      return el;
    });
    expect(card.querySelector('header .meta')?.textContent).toContain('Ryzen 9 7950X');
    expect(card.querySelector('.foot')).toBeNull();
  });

  it('keeps keyboard focus inside the Training slot when the live card swaps runs or becomes idle', async () => {
    const runs = await idleRuns();
    const [first, second] = runs;
    const api = new MockApi('normal', 0);
    const getRun = api.run.bind(api);
    api.run = (id: string) => {
      const listed = watched.runs?.find((r) => r.id === id);
      return listed ? Promise.resolve(listed) : getRun(id);
    };
    watched.runs = runs.map((r) => (r.id === first!.id ? { ...r, status: 'running' as const } : r));
    open(api);
    await training((t) => t.includes(first!.name), 'first live run');

    const cta = () => target.querySelector<HTMLAnchorElement>('.live .cta a')!;
    cta().focus();
    cta().dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    expect(document.activeElement).toBe(cta());

    // First run completes and second run takes over the keyed LiveRunCard: simulate Chromium dispatching a
    // synchronous focusout (relatedTarget: null) right as the focused link is detached during the swap.
    cta().dispatchEvent(new FocusEvent('focusout', { bubbles: true, relatedTarget: null }));
    watched.runs = runs.map((r) => (r.id === second!.id ? { ...r, status: 'running' as const } : r));
    flush();
    await training((t) => t.includes(second!.name), 'second live run');
    await vi.waitFor(() => {
      flush();
      expect(document.activeElement).toBe(cta());
      expect(cta().getAttribute('href')).toBe(`#/runs/${second!.id}`);
    });

    // Second run completes with nothing else training: focus lands on the route heading rather than <body>.
    watched.runs = runs;
    flush();
    await training((t) => t.includes('Nothing is training'), 'idle training panel');
    await vi.waitFor(() => {
      flush();
      expect(document.activeElement).toBe(target.querySelector('h1[data-route-focus]'));
    });
  });

  it('reconciles Overview summary, live slot, Recent runs, and sidebar immediately when SSE reports completion before the run list refreshes (playtest R15)', async () => {
    const api = new MockApi('normal', 0);
    const liveRun = (await api.runs()).find((r) => r.status === 'running')!;
    const runs = await idleRuns();
    const initialList = runs.map((r) => (r.id === liveRun.id ? liveRun : r));
    let handlers: RunEventHandlers = {};
    let eventOpens = 0;
    api.runEvents = (_id, h) => {
      eventOpens++;
      handlers = h;
      return () => {};
    };
    api.run = async () => liveRun;
    let resolveNextList: ((v: Run[]) => void) | null = null;
    let runsCalls = 0;
    api.runs = () => {
      runsCalls++;
      if (runsCalls === 1) return Promise.resolve(initialList);
      return new Promise<Run[]>((res) => (resolveNextList = res));
    };

    hub.api = api;
    hub.mode = 'mock';
    const shell = document.body.appendChild(document.createElement('div'));
    const shellView = render(
      () => (
        <>
          <RunWatch />
          <SidebarLive collapsed={false} />
        </>
      ),
      shell,
    );
    onTestFinished(() => {
      shellView.unmount();
      shell.remove();
    });
    open(api);

    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
      expect(target.querySelector('.live .card h2')?.textContent).toBe(liveRun.name);
      expect(shell.querySelector('a.live-run')).not.toBeNull();
      expect(activeRuns().training).toBe(1);
    });
    // Ensure LiveRun finished initial loading before SSE completion arrives.
    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.live .card .legend')).not.toBeNull();
    });
    expect(eventOpens).toBe(1);

    // SSE delivers completion while the refreshed GET /api/runs is still in flight.
    handlers.status?.({ status: 'completed', error: null });
    flush();

    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(target.querySelector('.live')?.textContent).toContain('Nothing is training');
    expect(target.querySelector('#recent-runs tbody')?.textContent).toContain(liveRun.name);
    expect(target.querySelector('#recent-runs tbody')?.textContent).toContain('Completed');
    expect(shell.querySelector('a.live-run')).toBeNull();
    expect(activeRuns()).toMatchObject({ training: 0, live: false });
    // No remount oscillation and immediate authoritative refresh triggered.
    expect(eventOpens).toBe(1);
    expect(runsCalls).toBe(2);
    expect(resolveNextList).not.toBeNull();

    // Authoritative list lands: UI stays consistent.
    resolveNextList!(initialList.map((r) => (r.id === liveRun.id ? { ...r, status: 'completed' } : r)));
    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
      expect(target.querySelector('#recent-runs tbody')?.textContent).toContain('Completed');
    });
    expect(eventOpens).toBe(1);
  });

  it('advances the live slot to a queued run and moves the stopped run to Recent runs on SSE stop (playtest R15)', async () => {
    const api = new MockApi('normal', 0);
    const first = (await api.runs()).find((r) => r.status === 'running')!;
    const runs = await idleRuns();
    const nextBase = runs.find((r) => r.id !== first.id)!;
    const next = { ...nextBase, status: 'queued' as const, queue_position: 1, epoch: 0, step: 0, best: null, cloud: undefined };
    const list = runs.map((r) => (r.id === first.id ? first : r.id === next.id ? next : r));
    const byId = new Map<string, RunEventHandlers>();
    api.runEvents = (id, h) => {
      byId.set(id, h);
      return () => {};
    };
    api.run = async (id) => list.find((r) => r.id === id)!;
    api.runs = () => new Promise<Run[]>(() => {}); // hold poll so only SSE drives the transition
    watched.runs = list;
    open(api);

    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.facts')?.textContent).toContain('1 run training');
      expect(target.querySelector('.live .card .legend')).not.toBeNull();
    });

    // Stopped from Run detail in another tab -> SSE delivers `stopped`.
    byId.get(first.id)?.status?.({ status: 'stopped', error: null });
    flush();

    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(target.querySelector('.live .card h2')?.textContent).toBe(next.name);
    expect(target.querySelector('.live .card')?.textContent).toContain('Starting');
    expect(target.querySelector('#recent-runs tbody')?.textContent).toContain(first.name);
    expect(target.querySelector('#recent-runs tbody')?.textContent).toContain('Stopped');
    expect(activeRuns()).toMatchObject({ training: 0, queued: 1, live: false });
  });

  it('keeps the streamed terminal state if the post-end refresh fails, and clears it on server epoch switch (playtest R15)', async () => {
    const api = new MockApi('normal', 0);
    const liveRun = (await api.runs()).find((r) => r.status === 'running')!;
    const runs = await idleRuns();
    const initialList = runs.map((r) => (r.id === liveRun.id ? liveRun : r));
    let handlers: RunEventHandlers = {};
    api.runEvents = (_id, h) => {
      handlers = h;
      return () => {};
    };
    api.run = async () => liveRun;
    let calls = 0;
    api.runs = () => (++calls === 1 ? Promise.resolve(initialList) : Promise.reject(new ApiError(0, 'offline', 'down')));

    hub.api = api;
    hub.mode = 'mock';
    const shell = document.body.appendChild(document.createElement('div'));
    const shellView = render(() => <RunWatch />, shell);
    onTestFinished(() => {
      shellView.unmount();
      shell.remove();
    });
    open(api);

    await vi.waitFor(() => {
      flush();
      expect(target.querySelector('.live .card .legend')).not.toBeNull();
    });

    handlers.status?.({ status: 'completed', error: null });
    flush();
    // Wait for the failing refresh to settle: Overview must stay on `nothing training`, never snapping back to stale `running`.
    await vi.waitFor(() => expect(calls).toBeGreaterThanOrEqual(2));
    await Promise.resolve();
    flush();
    expect(target.querySelector('.facts')?.textContent).toContain('nothing training');
    expect(target.querySelector('.live')?.textContent).toContain('Nothing is training');

    // Switching servers clears the earlier server's terminal override.
    hub.epoch++;
    flush();
    watched.runs = initialList;
    flush();
    expect(activeRuns().training).toBe(1);
  });
});
