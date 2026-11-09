// @vitest-environment happy-dom
// The Runs list on a server with queue control (docs/API.md, POST /api/runs/{id}/queue): a waiting run's row says its
// place and has a ⋯ menu (Start next, Move up, Move down); the new order shows at once and a refusal puts it back.
import { flushSync, mount, unmount, type Component } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '$lib/api/client';
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
