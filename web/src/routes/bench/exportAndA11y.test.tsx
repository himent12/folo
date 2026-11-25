// @vitest-environment happy-dom
import { flush as flushSync, tick, type Component } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import Benchmark from '../Benchmark';
import BenchCompare from '../BenchCompare';
import Benchmarks from '../Benchmarks';
import Models from '../Models';

let target: HTMLElement;
let view: Rendered | undefined;
let downloads: Array<{ filename: string; type: string; text: string }> = [];

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  toast.items = [];
  downloads = [];
  target = document.createElement('div');
  document.body.appendChild(target);

  const blobs = new Map<string, Blob>();
  vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
    const url = `blob:export/${blobs.size}`;
    blobs.set(url, b as Blob);
    return url;
  });
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    const blob = blobs.get(this.getAttribute('href') ?? '');
    if (blob) {
      void blob.text().then((text) => {
        downloads.push({ filename: this.download, type: blob.type, text });
      });
    }
  });
});

afterEach(() => {
  view?.unmount();
  view = undefined;
  target.remove();
  vi.restoreAllMocks();
});

function mountRoute<P extends object>(comp: Component<P>, route: string) {
  router.route = parse(route);
  const Comp = comp as Component<Record<string, unknown>>;
  view = render(() => <Comp />, target);
  flushSync();
}

async function settle(rounds = 14) {
  for (let i = 0; i < rounds; i++) {
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
  }
}

async function clickMenuItem(triggerLabel: string, itemLabel: string) {
  const trigger = target.querySelector<HTMLButtonElement>(`button[aria-haspopup="menu"][aria-label="${triggerLabel}"]`);
  expect(trigger).not.toBeNull();
  trigger!.click();
  flushSync();
  await tick();
  const item = [...target.querySelectorAll<HTMLElement>('[role="menuitem"]')].find((el) => el.textContent?.trim() === itemLabel);
  expect(item).not.toBeUndefined();
  item!.click();
  flushSync();
  await settle(4);
}

describe('Benchmarks, BenchCompare, and Models export & a11y', () => {
  it('exports filtered Benchmarks as CSV and JSON, announces status, and restores focus on Clear filters', async () => {
    mountRoute(Benchmarks, '#/benchmarks');
    await settle();

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter benchmarks"]')!;
    search.value = 'Fasteners';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();

    // Narrowed footer has polite live status.
    const foot = target.querySelector('.list-foot');
    expect(foot?.getAttribute('role')).toBe('status');
    expect(foot?.getAttribute('aria-live')).toBe('polite');
    expect(foot?.textContent).toContain('1 of 4 benchmarks');

    await clickMenuItem('Export benchmarks', 'Export CSV');
    await clickMenuItem('Export benchmarks', 'Export JSON');
    expect(downloads).toHaveLength(2);
    expect(downloads[0]!.filename).toBe('folo-benchmarks.csv');
    expect(downloads[0]!.text).toContain('Fasteners val');
    expect(downloads[0]!.text).not.toContain('Pond fish');
    const parsed = JSON.parse(downloads[1]!.text);
    expect(parsed.count).toBe(1);
    expect(parsed.benchmarks[0].name).toBe('Fasteners val');

    // Empty search -> Clear filters resets and restores focus to the search input.
    search.value = 'no-such-benchmark-xyz';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    const clear = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
    clear.focus();
    clear.click();
    flushSync();
    await Promise.resolve();
    expect(search.value).toBe('');
    expect(document.activeElement).toBe(search);
  });

  it('exports a single Benchmark leaderboard as CSV and JSON from the header menu', async () => {
    mountRoute(Benchmark, '#/benchmarks/bn_3c90');
    await settle(18);

    await clickMenuItem('More benchmark actions', 'Export leaderboard CSV');
    await clickMenuItem('More benchmark actions', 'Export leaderboard JSON');
    expect(downloads).toHaveLength(2);
    expect(downloads[0]!.filename).toBe('benchmark-fasteners-val-leaderboard.csv');
    expect(downloads[0]!.text).toContain('rank,id,name,contestant,status,score');
    const parsed = JSON.parse(downloads[1]!.text);
    expect(parsed.benchmark.id).toBe('bn_3c90');
    expect(parsed.count).toBeGreaterThan(0);
  });

  it('exports BenchCompare, exposes semantic table headers and accessible swap labels, and restores focus on baseline swap & chip remove', async () => {
    vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });
    const entries = (await hub.api.benchEntries('bn_3c90')).filter((e) => e.status === 'done' && e.score);
    const [e0, e1, e2] = entries;
    mountRoute(BenchCompare, `#/benchmarks/bn_3c90/compare?entries=${e0!.id},${e1!.id},${e2!.id}`);
    await settle(18);

    // Table semantic headers: scope="col" and scope="row"
    const colHeaders = [...target.querySelectorAll('table.cmp thead th[scope="col"]')];
    const rowHeaders = [...target.querySelectorAll('table.cmp tbody th[scope="row"]')];
    expect(colHeaders.length).toBe(4);
    expect(rowHeaders.length).toBeGreaterThanOrEqual(6);

    // Accessible baseline button label names the specific entry
    const swapBtn = target.querySelector<HTMLButtonElement>(`button.swap[aria-label="Make ${e1!.name} baseline"]`)!;
    expect(swapBtn).not.toBeNull();
    swapBtn.focus();
    swapBtn.click();
    flushSync();
    await Promise.resolve();
    // Focus stays inside the comparison table instead of falling to document.body
    expect(target.querySelector('table.cmp')?.contains(document.activeElement)).toBe(true);
    expect(target.querySelector('.sr-only[role="status"]')?.textContent).toContain(`Baseline set to ${e1!.name}`);

    // Export CSV and JSON from BenchCompare
    await clickMenuItem('Export comparison', 'Export CSV');
    await clickMenuItem('Export comparison', 'Export JSON');
    expect(downloads).toHaveLength(2);
    expect(downloads[0]!.filename).toBe('benchmark-fasteners-val-compare.csv');
    expect(downloads[0]!.text).toContain(`${e1!.name} (baseline)`);
    const parsed = JSON.parse(downloads[1]!.text);
    expect(parsed.baseline_id).toBe(e1!.id);
    expect(parsed.entries).toHaveLength(3);

    // Removing a chip moves keyboard focus to a neighbouring chip's pin button
    const removeBtn = target.querySelector<HTMLButtonElement>(`button.rm[aria-label="Remove ${e2!.name}"]`)!;
    removeBtn.focus();
    removeBtn.click();
    flushSync();
    await Promise.resolve();
    expect(document.activeElement?.classList.contains('ck')).toBe(true);
    expect(target.querySelector('.sr-only[role="status"]')?.textContent).toContain(`Removed ${e2!.name}`);
  });

  it('exports filtered Models as CSV and JSON and restores focus on Clear filters', async () => {
    mountRoute(Models, '#/models');
    await settle(18);

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter models"]')!;
    search.value = 'folo-n';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();

    const foot = target.querySelector('.list-foot');
    expect(foot?.getAttribute('role')).toBe('status');
    expect(foot?.getAttribute('aria-live')).toBe('polite');

    await clickMenuItem('Export models', 'Export CSV');
    await clickMenuItem('Export models', 'Export JSON');
    expect(downloads).toHaveLength(2);
    expect(downloads[0]!.filename).toBe('folo-models.csv');
    expect(downloads[0]!.text).toContain('group,dataset,split,ranked,mark,id,name,arch');
    const parsed = JSON.parse(downloads[1]!.text);
    expect(parsed.count).toBeGreaterThan(0);
    expect(parsed.models.every((m: { arch: string; name: string }) => `${m.name} ${m.arch}`.toLowerCase().includes('folo-n'))).toBe(true);

    // Empty filter -> Clear filters restores focus to the search input
    search.value = 'no-such-model-xyz';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    const clear = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
    clear.focus();
    clear.click();
    flushSync();
    await Promise.resolve();
    expect(search.value).toBe('');
    expect(document.activeElement).toBe(search);
  });
});
