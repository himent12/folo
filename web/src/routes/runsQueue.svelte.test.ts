// @vitest-environment happy-dom
// The Runs list on a server with queue control (docs/API.md, POST /api/runs/{id}/queue): a waiting run's row says its
// place and has a ⋯ menu (Start next, Move up, Move down); the new order shows at once and a refusal puts it back.
import { flushSync, mount, unmount, type Component } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, type RunEventHandlers } from '$lib/api/client';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router.svelte';
import { toast } from '$lib/ui/toast.svelte';
import Run from './Run.svelte';
import Runs from './Runs.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target?.remove();
  vi.restoreAllMocks();
});

async function open(api: MockApi) {
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse('#/runs');
  target = document.createElement('div');
  document.body.appendChild(target);
  c = mount(Runs as Component, { target });
  // Rows and the server's features both loaded: the waiting rows have their menus.
  await vi.waitFor(
    () => {
      flushSync();
      if (!target.querySelector('tr[data-id="run_q7c5"] td.act button')) throw new Error('not loaded');
    },
    { timeout: 4000, interval: 10 },
  );
}

const row = (id: string) => target.querySelector<HTMLTableRowElement>(`table.runs tr[data-id="${id}"]`)!;
const place = (id: string) => row(id).querySelector('.place')?.textContent?.trim();
async function menu(id: string): Promise<HTMLButtonElement[]> {
  row(id).querySelector<HTMLButtonElement>('td.act button')!.click();
  flushSync();
  return [...row(id).querySelectorAll<HTMLButtonElement>('[role="menuitem"]')];
}

describe('Runs: the training queue', () => {
  it('shows each waiting run its place instead of a progress bar, and a menu with the moves that apply', async () => {
    await open(new MockApi('normal', 0));
    expect(place('run_q4n2')).toBe('1st in queue');
    expect(place('run_q7c5')).toBe('2nd in queue');
    expect(row('run_7d1e').querySelector('td.act button')).toBeNull();
    // The front run cannot move up; the live run trains, so Start next (not "now"), and only for the one behind. The
    // queue's one action set ends with Remove from queue, as on the run's own page.
    expect((await menu('run_q4n2')).map((b) => b.textContent?.trim())).toEqual(['Move down', 'Remove from queue']);
    expect((await menu('run_q7c5')).map((b) => b.textContent?.trim())).toEqual(['Start next', 'Move up', 'Remove from queue']);
  });

  it('moves at once, and the server agrees', async () => {
    const api = new MockApi('normal', 0);
    await open(api);
    const move = vi.spyOn(api, 'moveRun');
    const ok = vi.spyOn(toast, 'success');
    (await menu('run_q7c5')).find((b) => b.textContent?.trim() === 'Start next')!.click();
    flushSync();
    expect(place('run_q7c5')).toBe('1st in queue');
    expect(place('run_q4n2')).toBe('2nd in queue');
    await vi.waitFor(() => expect(ok).toHaveBeenCalledWith('fasteners-m starts next', 'When fasteners-s finishes.'));
    expect(move).toHaveBeenCalledWith('run_q7c5', { action: 'start' });
    expect(Object.fromEntries((await api.runs()).filter((r) => r.queue_position).map((r) => [r.id, r.queue_position]))).toEqual({ run_q7c5: 1, run_q4n2: 2 });
  });

  it('puts the order back when the run already started, and says so plainly', async () => {
    const api = new MockApi('normal', 0);
    await open(api);
    let answer!: (e: unknown) => void;
    vi.spyOn(api, 'moveRun').mockReturnValue(new Promise((_, reject) => (answer = reject)));
    const info = vi.spyOn(toast, 'info');
    (await menu('run_q7c5')).find((b) => b.textContent?.trim() === 'Move up')!.click();
    flushSync();
    expect(place('run_q7c5')).toBe('1st in queue');
    answer(new ApiError(409, 'run_not_queued', "run 'run_q7c5' is running, not waiting in the queue"));
    await vi.waitFor(() => {
      flushSync();
      expect(place('run_q7c5')).toBe('2nd in queue');
    });
    expect(info).toHaveBeenCalledWith('It already started', 'fasteners-m left the queue before the move.');
  });

  it('hides the controls from servers without queue control', async () => {
    const api = new MockApi('normal', 0);
    vi.spyOn(api, 'system').mockResolvedValue({ cpu: { name: 'cpu', threads: 1, util: 0 }, memory: { used_bytes: 0, total_bytes: 1 }, gpus: [], features: [] });
    hub.api = api;
    hub.mode = 'mock';
    router.route = parse('#/runs');
    target = document.createElement('div');
    document.body.appendChild(target);
    c = mount(Runs as Component, { target });
    await vi.waitFor(
      () => {
        flushSync();
        if (!target.querySelector('tr[data-id="run_q7c5"]')) throw new Error('not loaded');
      },
      { timeout: 4000, interval: 10 },
    );
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    expect(target.querySelectorAll('td.act button')).toHaveLength(0);
    expect(place('run_q7c5')).toBeUndefined();
  });
});

describe('a cloud run in its hand-over', () => {
  /** Mounts `page` at `path` on the `cloud-upload` mock (shapes-hard-x, run_c9e4, uploading its dataset). */
  async function show(page: Component, path: string, api = new MockApi('cloud-upload', 0)) {
    hub.api = api;
    hub.mode = 'mock';
    router.route = parse(`#${path}`);
    target = document.createElement('div');
    document.body.appendChild(target);
    c = mount(page, { target });
  }

  it('is listed by its phase and without a place in the queue, even where the server gives it one', async () => {
    const api = new MockApi('cloud-upload', 0);
    const list = api.runs.bind(api);
    vi.spyOn(api, 'runs').mockImplementation(async () => (await list()).map((r) => (r.id === 'run_c9e4' ? { ...r, queue_position: 3 } : r)));
    await show(Runs as Component, '/runs', api);
    await vi.waitFor(
      () => {
        flushSync();
        if (!target.querySelector('tr[data-id="run_q7c5"] td.act button')) throw new Error('not loaded');
      },
      { timeout: 4000, interval: 10 },
    );
    expect(row('run_c9e4').querySelector('.badge')?.textContent?.trim()).toBe('Uploading data');
    expect(row('run_c9e4').querySelector('.badge')?.classList.contains('queued')).toBe(true);
    expect(place('run_c9e4')).toBeUndefined();
    expect(row('run_c9e4').querySelector('td.act button')).toBeNull();
    // Phones: the status word leads the row's meta line.
    const phone = target.querySelector(`ul.rows a[href$="/runs/run_c9e4"] .r-meta`);
    expect(phone?.textContent).toMatch(/^Uploading data/);
  });

  it("titles its own page's header with the phase, not Queued", async () => {
    await show(Run as Component, '/runs/run_c9e4');
    const badge = await vi.waitFor(
      () => {
        flushSync();
        const b = target.querySelector<HTMLElement>('.badge');
        if (!b) throw new Error('not loaded');
        return b;
      },
      { timeout: 4000, interval: 10 },
    );
    expect(badge.textContent?.trim()).toBe('Uploading data');
  });
});

describe('Run: a waiting run on a server without estimates', () => {
  it('says the runs in front have no known length, never "starts in about" the training run’s time left alone', async () => {
    hub.api = new MockApi('no-throughput', 0);
    hub.mode = 'mock';
    router.route = parse('#/runs/run_q7c5');
    target = document.createElement('div');
    document.body.appendChild(target);
    c = mount(Run as Component, { target });
    const text = await vi.waitFor(
      () => {
        flushSync();
        const t = [...target.querySelectorAll('p, div')].map((e) => e.textContent ?? '').find((t) => /Starts (in|after)/.test(t) && /\(.*\)\./.test(t));
        if (!t) throw new Error('no queue line yet');
        return t.replace(/\s+/g, ' ');
      },
      { timeout: 4000, interval: 10 },
    );
    expect(text).not.toMatch(/Starts in about/);
    expect(text).toMatch(/Starts in more than|Starts after/);
    expect(text).toMatch(/not known yet/);
  });
});

describe('Run: Start next on a run the worker took meanwhile', () => {
  it('keeps the status the stream reported after the refresh was asked for (never back to Queued)', async () => {
    // Start next answers `409 run_not_queued` (the worker took the run), and the page reads the run again. That
    // snapshot was asked for while the trainer was still being built (`queued`); the stream's `running` came first. The
    // late snapshot put the run back to Queued, and no other status event comes until the run ends.
    const api = new MockApi('normal', 0);
    let handlers: RunEventHandlers = {};
    vi.spyOn(api, 'runEvents').mockImplementation((_id, h) => {
      handlers = h;
      return () => {};
    });
    hub.api = api;
    hub.mode = 'mock';
    router.route = parse('#/runs/run_q7c5');
    target = document.createElement('div');
    document.body.appendChild(target);
    c = mount(Run as Component, { target });
    const start = await vi.waitFor(
      () => {
        flushSync();
        const b = [...target.querySelectorAll<HTMLButtonElement>('.callout button')].find((x) => x.textContent?.includes('Start next'));
        if (!b) throw new Error('not loaded');
        return b;
      },
      { timeout: 4000, interval: 10 },
    );
    const queued = await api.run('run_q7c5');
    let answer!: (r: typeof queued) => void;
    const real = api.run.bind(api);
    let first = true;
    vi.spyOn(api, 'run').mockImplementation((id: string) => {
      if (!first) return real(id);
      first = false;
      return new Promise<typeof queued>((r) => (answer = r));
    });
    vi.spyOn(api, 'moveRun').mockRejectedValue(new ApiError(409, 'run_not_queued', "run 'run_q7c5' is starting, not waiting in the queue"));
    start.click();
    await vi.waitFor(() => expect(answer).toBeTypeOf('function'));
    handlers.status?.({ status: 'running', error: null, steps_per_epoch: 50, epochs: queued.epochs });
    flushSync();
    const badge = () => target.querySelector('.badge')?.textContent?.trim();
    expect(badge()).toBe('Running');
    answer({ ...queued, status: 'queued', queue_position: null });
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    expect(badge()).toBe('Running');
  });
});
