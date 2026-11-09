// @vitest-environment happy-dom
// The palette's actions on what the query names: "New run on <dataset>" and "Open <model> in Playground" show once a
// query names one, rank with the entities, are options a screen reader can name, and run from the keyboard.
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import { router } from '$lib/router.svelte';
import CommandPalette from './CommandPalette.svelte';

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
  it('offers none before a query names something', () => {
    c = mount(CommandPalette, { target, props: { open: true, onshortcuts: () => {} } });
    flushSync();
    expect(document.querySelector('[id^="pi-act-"]')).toBeNull();
  });

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

  it("marks an action's description as dispensable (phones leave it out) and keeps entity facts", async () => {
    await openWith('fasteners');
    const hint = (id: string) => document.querySelector<HTMLElement>(`#${id} .hint`)!;
    expect(hint('pi-act-train-ds_fasteners').classList.contains('describes')).toBe(true);
    expect(hint('pi-ds-ds_fasteners').classList.contains('describes')).toBe(false);
  });

  it("writes a run's status in StatusBadge's sentence-case word", async () => {
    await openWith('shapes');
    const hints = [...document.querySelectorAll<HTMLElement>('[id^="pi-run-"] .hint')].map((h) => h.textContent?.trim() ?? '');
    expect(hints.length).toBeGreaterThan(0);
    for (const h of hints) expect(h).toMatch(/^(Queued|Running|Stopping|Stopped|Completed|Failed)/);
  });

  it('writes a cloud run in its hand-over by its phase, not as Queued', async () => {
    hub.api = new MockApi('cloud-upload', 0);
    await openWith('shapes-hard-x', '#pi-run-run_c9e4');
    expect(document.querySelector('#pi-run-run_c9e4 .hint')?.textContent?.trim()).toMatch(/^Uploading data/);
  });

  it('answers to "toggle theme"', async () => {
    await openWith('toggle theme', '#pi-theme');
    expect(options()[0]).toMatch(/^Switch theme/);
  });
});

describe('CommandPalette with no query', () => {
  const key = (input: HTMLElement, k: string) => {
    input.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }));
    flushSync();
  };
  const selected = () => document.querySelector('[role="option"][aria-selected="true"] .label')?.textContent?.trim();

  it('wraps the selection at both ends', () => {
    c = mount(CommandPalette, { target, props: { open: true, onshortcuts: () => {} } });
    flushSync();
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
    const { pageCommandList } = await import('./pageCommands.svelte');
    const stop = vi.fn();
    pageCommandList.items = [{ id: 'stop', label: 'Stop fasteners-s', icon: (await import('@lucide/svelte/icons/square')).default, run: stop }];
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
