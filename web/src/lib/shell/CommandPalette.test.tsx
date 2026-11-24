// @vitest-environment happy-dom
// The palette's actions on what the query names: "New run on <dataset>" and "Open <model> in Playground" show once a
// query names one, rank with the entities, are options a screen reader can name, and run from the keyboard.
import { flush as flushSync, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Run } from '$lib/api/types';
import { LABEL_FEATURE } from '$lib/label';
import { router } from '$lib/router';
import CommandPalette, { type CommandPaletteProps } from './CommandPalette';

/** Port of `mount(CommandPalette, { target, props })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof CommandPalette, o: { target: HTMLElement; props: CommandPaletteProps }) => render(() => <CommandPalette {...o.props} />, o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  target = document.createElement('div');
  document.body.appendChild(target);
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
  HTMLDialogElement.prototype.close ??= function (this: HTMLDialogElement) {
    this.open = false;
    this.dispatchEvent(new Event('close'));
  };
  Element.prototype.scrollIntoView ??= () => {};
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
  vi.restoreAllMocks();
});

async function openWith(query: string, ready = '[role="option"][id^="pi-act-"]') {
  c = mount(CommandPalette, { target, props: { open: true, onshortcuts: () => {} } });
  flushSync();
  const input = document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
  // The lists load on open.
  await vi.waitFor(() => {
    input.value = query;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    if (!document.querySelector(ready)) throw new Error('waiting for the lists');
  });
  return input;
}
const options = () => [...document.querySelectorAll<HTMLElement>('[role="option"]')].map((o) => o.querySelector('.label')?.textContent?.trim());

describe('CommandPalette actions on entities', () => {
  it('a dataset name offers a run on it, below the dataset, and Enter starts it', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const input = await openWith('fasteners');
    const labels = options();
    expect(labels).toContain('New run on fasteners');
    expect(labels.indexOf('fasteners')).toBeLessThan(labels.indexOf('New run on fasteners'));
    const opt = document.getElementById('pi-act-train-ds_fasteners')!;
    expect(opt.getAttribute('role')).toBe('option');
    expect(opt.textContent).toContain('Train a detector on this dataset');
    // Arrow down to it: the combobox points at it (aria-activedescendant), Enter runs it.
    const at = labels.indexOf('New run on fasteners');
    for (let i = 0; i < at; i++) input.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    flushSync();
    expect(input.getAttribute('aria-activedescendant')).toBe('pi-act-train-ds_fasteners');
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    expect(go).toHaveBeenCalledWith('/runs/new?dataset=ds_fasteners');
  });

  it('a model name offers it in the Playground', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    await openWith('playground shapes-s-best');
    const first = document.querySelector<HTMLElement>('[role="option"]')!;
    expect(first.querySelector('.label')?.textContent?.trim()).toBe('Open shapes-s-best in Playground');
    expect(first.getAttribute('aria-selected')).toBe('true');
    first.click();
    expect(go).toHaveBeenCalledWith(expect.stringMatching(/^\/playground\?model=/));
  });

  it('finds a run by the id its page shows, and opens it (shell F3)', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    await openWith('run_q4n2', '#pi-run-run_q4n2');
    const first = document.querySelector<HTMLElement>('[role="option"]')!;
    expect(first.id).toBe('pi-run-run_q4n2');
    first.click();
    expect(go).toHaveBeenCalledWith('/runs/run_q4n2');
  });

  it('writes a cloud run in its hand-over by its phase, not as Queued', async () => {
    hub.api = new MockApi('cloud-upload', 0);
    await openWith('shapes-hard-x', '#pi-run-run_c9e4');
    expect(document.querySelector('#pi-run-run_c9e4 .hint')?.textContent?.trim()).toMatch(/^Uploading data/);
  });

});

describe('CommandPalette: labeling jobs and loops', () => {
  it('finds a labeling job and a loop by name, like a benchmark, and opens them', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    await openWith('warehouse-bins', '#pi-label-lb_7c21');
    const job = document.getElementById('pi-label-lb_7c21')!;
    expect(job.querySelector('.label')?.textContent?.trim()).toBe('warehouse-bins');
    expect(job.textContent).toMatch(/0 of 40 images labeled/);
    job.click();
    expect(go).toHaveBeenCalledWith('/datasets/label/lb_7c21');
    unmount(c!);
    c = undefined;
    const loop = (await Promise.all((await hub.api.benches()).map((b) => hub.api.benchLoops(b.id)))).flat()[0]!;
    await openWith(loop.name, `#pi-loop-${loop.id}`);
    document.getElementById(`pi-loop-${loop.id}`)!.click();
    expect(go).toHaveBeenLastCalledWith(`/benchmarks/${loop.bench_id}/loops/${loop.id}`);
  });
});

describe('CommandPalette with no query', () => {
  const key = (input: HTMLElement, k: string) => {
    input.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }));
    flushSync();
  };
  const selected = () => document.querySelector('[role="option"][aria-selected="true"] .label')?.textContent?.trim();

  it('offers no action on an entity before a query names one, and wraps the selection at both ends', () => {
    c = mount(CommandPalette, { target, props: { open: true, onshortcuts: () => {} } });
    flushSync();
    expect(document.querySelector('[id^="pi-act-"]')).toBeNull();
    const input = document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
    const first = selected();
    key(input, 'ArrowUp');
    const last = options().at(-1);
    expect(selected()).toBe(last);
    key(input, 'ArrowDown');
    expect(selected()).toBe(first);
  });

  it('lists the runs in flight, by their status, not their display word', async () => {
    const api = hub.api as MockApi;
    const active = (await api.runs()).filter((r) => ['running', 'stopping', 'queued'].includes(r.status));
    expect(active.length).toBeGreaterThan(0);
    await openWith('', '[id^="pi-run-"]');
    const listed = [...document.querySelectorAll<HTMLElement>('[id^="pi-run-"]')].map((o) => o.id.replace('pi-run-', ''));
    expect(listed.sort()).toEqual(active.map((r) => r.id).sort());
  });

  it('announces the result count once typing pauses, and keeps focus in the field', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    try {
      const input = await openWith('fasteners');
      input.focus();
      const status = document.querySelector<HTMLElement>('[role="status"]')!;
      expect(status.getAttribute('aria-live')).toBe('polite');
      expect(status.textContent).toBe('');
      vi.advanceTimersByTime(600);
      flushSync();
      expect(status.textContent).toBe(`${options().length} results`);
      input.value = 'zzzzqqq';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      vi.advanceTimersByTime(600);
      flushSync();
      expect(status.textContent).toBe('No results');
      expect(document.activeElement).toBe(input);
    } finally {
      vi.useRealTimers();
    }
  });

  it('lists the page commands first, then the items chosen last', async () => {
    localStorage.setItem('folo.palette.recent', JSON.stringify(['nav-models']));
    const { pageCommandList } = await import('./pageCommands');
    const stop = vi.fn();
    pageCommandList.items = [{ id: 'stop', label: 'Stop fasteners-s', icon: (await import('$icons/square')).default, run: stop }];
    c = mount(CommandPalette, { target, props: { open: true, onshortcuts: () => {} } });
    flushSync();
    const groups = [...document.querySelectorAll('.group')].map((g) => g.textContent?.trim());
    expect(groups.slice(0, 3)).toEqual(['This page', 'Recent', 'Go to']);
    expect(options().slice(0, 2)).toEqual(['Stop fasteners-s', 'Models']);
    key(document.querySelector<HTMLInputElement>('input[role="combobox"]')!, 'Enter');
    expect(stop).toHaveBeenCalledOnce();
    pageCommandList.items = [];
    localStorage.removeItem('folo.palette.recent');
  });
});

describe('CommandPalette: Settings and touch (shell F14, F17)', () => {
  it("finds Settings' sections by what they hold, and opens the section", async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const input = await openWith('cloud', '[id="pi-settings-cloud"]');
    expect(options()).toEqual(['Cloud']);
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    expect(go).toHaveBeenCalledWith('/settings?section=cloud');
  });

  it('offers no Cloud section on a server without cloud compute', async () => {
    hub.api = new MockApi('cloud-off', 0);
    await openWith('settings', '[id="pi-settings-connection"]');
    // Wait until the system answer is in: the sections then settle.
    await vi.waitFor(async () => expect((await hub.api.system()).features).not.toContain('cloud'));
    flushSync();
    expect(document.querySelector('[id="pi-settings-cloud"]')).toBeNull();
  });

  it('shows no key hints, footer or sidebar action on a touch-only device', async () => {
    const real = window.matchMedia;
    vi.spyOn(window, 'matchMedia').mockImplementation((q: string) => ({ ...real.call(window, q), matches: q.includes('hover: none') }) as MediaQueryList);
    await openWith('o', '[role="option"]');
    expect(options()).toContain('Overview');
    expect(options()).not.toContain('Collapse sidebar');
    expect(document.querySelector('.palette footer')).toBeNull();
    expect(document.querySelector('.palette kbd')).toBeNull();
  });
});

// #324: on `folo serve --mock` the palette offered New labeling job over "This server cannot label images", and the
// form it opened could only fail (501). It follows the server's health (`hub.features`), which a later health replaces.
describe('CommandPalette: what the server can do', () => {
  afterEach(() => (hub.features = null));
  const offered = (id: string) => document.getElementById(`pi-${id}`) !== null;
  const press = (input: HTMLInputElement, key: string) => {
    input.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
    flushSync();
  };
  /** Arrows down to `id`: the combobox then points at it. */
  function arrowTo(input: HTMLInputElement, id: string) {
    for (let i = 0; i < 40 && input.getAttribute('aria-activedescendant') !== `pi-${id}`; i++) press(input, 'ArrowDown');
    expect(input.getAttribute('aria-activedescendant')).toBe(`pi-${id}`);
  }

  for (const [what, features, shown] of [
    ['lists labeling', [LABEL_FEATURE], true],
    ['lists features without labeling', ['queue_control'], false],
    ['lists no features (an older server, the demo)', null, true],
  ] as const) {
    it(`${shown ? 'offers' : 'does not offer'} New labeling job when the server's health ${what}; Label images always`, async () => {
      hub.features = features;
      const input = await openWith('new', '[id="pi-new-run"]');
      expect(offered('label-new')).toBe(shown);
      input.value = 'label images';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      // A place, as Datasets' own link to it: there the server says why it cannot.
      expect(offered('label')).toBe(true);
    });
  }

  it('follows a later health answer while open (a server restarted with other features)', async () => {
    await openWith('new', '[id="pi-new-run"]');
    expect(offered('label-new')).toBe(true);
    hub.features = ['queue_control'];
    flushSync();
    expect(offered('label-new')).toBe(false);
    hub.features = ['queue_control', LABEL_FEATURE];
    flushSync();
    expect(offered('label-new')).toBe(true);
  });

  // Review of #341: the highlight was a position, so when New labeling job went the one under it took its place, and
  // Enter started a run on a dataset nobody had highlighted.
  it('keeps the highlight on its command, and gives it to the first when that command goes', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const input = await openWith('new', '[id^="pi-act-train-"]');
    arrowTo(input, 'label-new');
    hub.features = ['queue_control'];
    flushSync();
    const first = document.querySelector<HTMLElement>('[role="option"]')!;
    expect(input.getAttribute('aria-activedescendant')).toBe(first.id);
    expect(first.getAttribute('aria-selected')).toBe('true');
    press(input, 'Enter');
    expect(go).toHaveBeenCalledTimes(1);
    expect(first.id).not.toMatch(/^pi-act-train-/);
    expect(go.mock.calls[0]![0]).not.toMatch(/dataset=/);
  });

  // Review of #341, round 2: the first item was highlighted by its place, so a row that loaded above it took the
  // highlight, and Enter ran a command nobody had highlighted.
  it('keeps the first item highlighted when a row loads above it', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    let late!: () => void;
    const api = hub.api as MockApi;
    const benches = api.benches.bind(api);
    vi.spyOn(api, 'benches').mockImplementation(() => new Promise((r) => (late = () => r(benches()))));
    const input = await openWith('new', '[id="pi-new-run"]');
    expect(document.querySelector('[role="option"]')!.id).toBe('pi-new-run');
    expect(input.getAttribute('aria-activedescendant')).toBe('pi-new-run');
    late();
    await vi.waitFor(() => (flushSync(), expect(document.querySelector('[role="option"]')!.id).toBe('pi-bench-new')));
    expect(input.getAttribute('aria-activedescendant')).toBe('pi-new-run');
    press(input, 'Enter');
    expect(go).toHaveBeenCalledWith('/runs/new');
  });

  it('keeps the highlight on its command when a list loads late', async () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    let late!: () => void;
    const api = hub.api as MockApi;
    const datasets = api.datasets.bind(api);
    vi.spyOn(api, 'datasets').mockImplementation(() => new Promise((r) => (late = () => r(datasets()))));
    const input = await openWith('new', '[id="pi-label-new"]');
    arrowTo(input, 'label-new');
    late();
    await vi.waitFor(() => (flushSync(), expect(document.querySelector('[id^="pi-act-train-"]')).not.toBeNull()));
    expect(input.getAttribute('aria-activedescendant')).toBe('pi-label-new');
    press(input, 'Enter');
    expect(go).toHaveBeenCalledWith('/datasets/label?new=1');
  });
});

describe('CommandPalette after a switch to another server', () => {
  it('lists nothing of the server before: neither what it listed nor an answer that comes late (review R4)', async () => {
    const open = signal(true);
    c = render(() => <CommandPalette bind:open={open} onshortcuts={() => {}} />, target);
    const runs = () => document.querySelectorAll('[id^="pi-run-"]').length;
    await vi.waitFor(() => {
      flushSync();
      expect(runs()).toBeGreaterThan(0);
    });
    open.set(false);
    flushSync();

    // Server B fails its run list; server A's own answer to an opening just before the switch arrives after it.
    let late!: (r: Run[]) => void;
    const a = new MockApi('normal', 0);
    const aRuns = await a.runs();
    a.runs = () => new Promise((r) => (late = r));
    hub.api = a;
    open.set(true);
    flushSync();
    open.set(false);
    flushSync();
    const b = new MockApi('normal', 0);
    b.runs = () => Promise.reject(new ApiError(500, 'internal', 'the run index could not be read'));
    hub.api = b;
    hub.epoch++;
    open.set(true);
    flushSync();
    late(aRuns);
    // A query that names server A's runs as well as the dataset B lists.
    const input = document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
    await vi.waitFor(() => {
      input.value = 'fasteners';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      expect(document.querySelector('[id^="pi-ds-"]')).not.toBeNull();
    });
    expect(aRuns.some((r) => r.name.includes('fasteners'))).toBe(true);
    expect(runs()).toBe(0);
  });
});

describe('CommandPalette: async hardening, deferred loops, clear button, and pointer guard', () => {
  afterEach(() => {
    hub.features = null;
  });

  it('aborts and discards stale fetches across open/close cycles on the same server', async () => {
    const api = hub.api as MockApi;
    const allRuns = await api.runs();
    const staleRun: Run = { ...allRuns[0]!, id: 'run_stale_1', name: 'stale-run-from-open-1' };
    const freshRun: Run = { ...allRuns[0]!, id: 'run_fresh_2', name: 'fresh-run-from-open-2' };
    let resolveFirst!: (r: Run[]) => void;
    let firstSignal: AbortSignal | undefined;
    let call = 0;
    vi.spyOn(hub.api, 'runs').mockImplementation((signal?: AbortSignal) => {
      call++;
      if (call === 1) {
        firstSignal = signal;
        return new Promise<Run[]>((r) => (resolveFirst = r));
      }
      return Promise.resolve([freshRun]);
    });

    const open = signal(true);
    c = render(() => <CommandPalette bind:open={open} onshortcuts={() => {}} />, target);
    flushSync();
    expect(call).toBe(1);
    expect(firstSignal?.aborted).toBe(false);

    // Closing aborts the first opening's in-flight request.
    open.set(false);
    flushSync();
    expect(firstSignal?.aborted).toBe(true);

    // Reopening fetches fresh data; resolving the stale request late must not overwrite it.
    open.set(true);
    flushSync();
    const input = document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
    await vi.waitFor(() => {
      input.value = 'fresh-run';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      expect(document.getElementById('pi-run-run_fresh_2')).not.toBeNull();
    });
    resolveFirst([staleRun]);
    await Promise.resolve();
    flushSync();
    input.value = 'stale-run';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(document.getElementById('pi-run-run_stale_1')).toBeNull();
  });

  it('defers per-benchmark loop fetches until a relevant query and skips labelJobs when lacking LABEL_FEATURE', async () => {
    hub.features = ['queue_control'];
    const api = hub.api as MockApi;
    const loopsSpy = vi.spyOn(api, 'benchLoops');
    const jobsSpy = vi.spyOn(api, 'labelJobs');
    c = mount(CommandPalette, { target, props: { open: true, onshortcuts: () => {} } });
    flushSync();
    // Wait for benches to finish loading on open.
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelector('[id="pi-bench-new"]')).not.toBeNull();
    });
    expect(loopsSpy).not.toHaveBeenCalled();
    expect(jobsSpy).not.toHaveBeenCalled();

    // Typing a 2+ character query fetches loops once.
    const input = document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
    input.value = 'lo';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    await vi.waitFor(() => expect(loopsSpy).toHaveBeenCalled());
  });

  it('distinguishes loading and unavailable entity lookups from "No matches"', async () => {
    const api = hub.api as MockApi;
    let resolveRuns!: (r: Run[]) => void;
    vi.spyOn(api, 'runs').mockImplementation(() => new Promise<Run[]>((r) => (resolveRuns = r)));
    c = mount(CommandPalette, { target, props: { open: true, onshortcuts: () => {} } });
    flushSync();
    const input = document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
    input.value = 'zzzz-unmatched-query';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(document.querySelector('.none')?.textContent).toContain('Searching');
    expect(document.querySelector('.none')?.textContent).not.toContain('No matches');

    resolveRuns([]);
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelector('.none')?.textContent).toContain('No matches for “zzzz-unmatched-query”');
    });

    // When runs fails with 500, the empty message reports unavailable entity lookup.
    unmount(c);
    c = undefined;
    hub.api = new MockApi('normal', 0);
    vi.spyOn(hub.api, 'runs').mockRejectedValue(new ApiError(500, 'internal', 'db down'));
    c = mount(CommandPalette, { target, props: { open: true, onshortcuts: () => {} } });
    flushSync();
    const input2 = document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
    input2.value = 'zzzz-unmatched-query';
    input2.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    await vi.waitFor(() => {
      flushSync();
      expect(document.querySelector('.none')?.textContent).toContain('could not load runs');
    });
  });

  it('provides an inline clear button in .search that resets query and refocuses input', () => {
    c = mount(CommandPalette, { target, props: { open: true, onshortcuts: () => {} } });
    flushSync();
    const input = document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
    expect(document.querySelector('.search button.clear')).toBeNull();
    input.value = 'models';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    const clear = document.querySelector<HTMLButtonElement>('.search button.clear')!;
    expect(clear).not.toBeNull();
    expect(clear.getAttribute('aria-label')).toBe('Clear search');
    clear.click();
    flushSync();
    expect(input.value).toBe('');
    expect(document.querySelector('.search button.clear')).toBeNull();
    expect(document.activeElement).toBe(input);
  });

  it('ignores stationary pointermove coordinates during keyboard navigation', () => {
    c = mount(CommandPalette, { target, props: { open: true, onshortcuts: () => {} } });
    flushSync();
    const input = document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
    const items = [...document.querySelectorAll<HTMLElement>('[role="option"]')];
    expect(items.length).toBeGreaterThanOrEqual(3);

    // Move mouse over item 0 at (120, 200).
    items[0]!.dispatchEvent(new PointerEvent('pointermove', { clientX: 120, clientY: 200, bubbles: true }));
    flushSync();
    expect(input.getAttribute('aria-activedescendant')).toBe(items[0]!.id);

    // Keyboard ArrowDown moves to item 1; a synthetic stationary pointermove on item 2 must NOT hijack selection.
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true }));
    flushSync();
    expect(input.getAttribute('aria-activedescendant')).toBe(items[1]!.id);

    items[2]!.dispatchEvent(new PointerEvent('pointermove', { clientX: 120, clientY: 200, bubbles: true }));
    flushSync();
    expect(input.getAttribute('aria-activedescendant')).toBe(items[1]!.id);

    // Real pointer movement to new coordinates selects item 2.
    items[2]!.dispatchEvent(new PointerEvent('pointermove', { clientX: 124, clientY: 240, bubbles: true }));
    flushSync();
    expect(input.getAttribute('aria-activedescendant')).toBe(items[2]!.id);
  });
});
