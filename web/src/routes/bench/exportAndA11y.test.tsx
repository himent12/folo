// @vitest-environment happy-dom
import { flush as flushSync, tick, type Component } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Bench, BenchEntry } from '$lib/api/types';
import { parse, router } from '$lib/router';
import { toast } from '$lib/ui/toast';
import Benchmark from '../Benchmark';
import BenchCompare from '../BenchCompare';
import Benchmarks from '../Benchmarks';
import Models from '../Models';
import { diffAsPrinted } from './charts';
import { compareExport, leaderboardExport } from './export';

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
  history.replaceState(history.state, '', route);
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
  it('exports filtered Benchmarks as CSV and JSON, keeps filter live region mounted before narrowing, and restores focus on Clear filters', async () => {
    mountRoute(Benchmarks, '#/benchmarks');
    await settle();

    // Persistently mounted polite live region exists BEFORE filtering so the first narrowing is announced.
    const status = target.querySelector<HTMLElement>('[data-filter-status]')!;
    expect(status).not.toBeNull();
    expect(status.getAttribute('role')).toBe('status');
    expect(status.getAttribute('aria-live')).toBe('polite');
    expect(status.textContent).toBe('');

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter benchmarks"]')!;
    search.value = 'Fasteners';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(status.textContent).toBe('Showing 1 of 4 benchmarks.');

    await clickMenuItem('Export benchmarks', 'Export CSV');
    await clickMenuItem('Export benchmarks', 'Export JSON');
    expect(downloads).toHaveLength(2);
    expect(downloads[0]!.filename).toBe('folo-benchmarks.csv');
    expect(downloads[0]!.text).toContain('Fasteners val');
    expect(downloads[0]!.text).not.toContain('Pond fish');
    const parsed = JSON.parse(downloads[1]!.text);
    expect(parsed.count).toBe(1);
    expect(parsed.benchmarks[0].name).toBe('Fasteners val');

    // Empty search -> Clear filters resets, updates status, and restores focus to the search input.
    search.value = 'no-such-benchmark-xyz';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(status.textContent).toBe('Showing 0 of 4 benchmarks.');
    const clear = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
    clear.focus();
    clear.click();
    flushSync();
    await Promise.resolve();
    expect(search.value).toBe('');
    expect(status.textContent).toBe('Showing all 4 benchmarks.');
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

  it('exports BenchCompare, uses WCAG 2.5.3 baseline labels and semantic headers, keeps phone shortcuts working, and restores focus on swap & chip remove', async () => {
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

    // WCAG 2.5.3 Label in Name: accessible name starts with visible "Make baseline"
    const swapBtn = target.querySelector<HTMLButtonElement>(`button.swap[aria-label="Make baseline: ${e1!.name}"]`)!;
    expect(swapBtn).not.toBeNull();
    swapBtn.focus();
    swapBtn.click();
    flushSync();
    await Promise.resolve();
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

  it('keeps BenchCompare keyboard shortcuts working at phone widths when the Export menu is closed', async () => {
    const win = window as unknown as { happyDOM?: { setViewport(v: { width: number }): void } };
    const prevWidth = window.innerWidth;
    win.happyDOM?.setViewport({ width: 390 });
    try {
      const entries = (await hub.api.benchEntries('bn_3c90')).filter((e) => e.status === 'done' && e.score);
      const [e0, e1] = entries;
      mountRoute(BenchCompare, `#/benchmarks/bn_3c90/compare?entries=${e0!.id},${e1!.id}`);
      await settle(18);
      expect(target.querySelector('[role="menu"]')).toBeNull();
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '1', bubbles: true }));
      flushSync();
      const firstPin = target.querySelector<HTMLButtonElement>('button.ck')!;
      expect(firstPin.getAttribute('aria-pressed')).toBe('true');
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      flushSync();
      expect(firstPin.getAttribute('aria-pressed')).toBe('false');
    } finally {
      win.happyDOM?.setViewport({ width: prevWidth });
    }
  });

  it('exports filtered Models as CSV and JSON, omits empty header actions on empty lists, and restores focus on Clear filters', async () => {
    mountRoute(Models, '#/models');
    await settle(18);

    const status = target.querySelector<HTMLElement>('[data-filter-status]')!;
    expect(status).not.toBeNull();
    expect(status.textContent).toBe('');

    const search = target.querySelector<HTMLInputElement>('input[aria-label="Filter models"]')!;
    search.value = 'folo-n';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(status.textContent).toMatch(/^Showing \d+ of \d+ models\.$/);

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
    expect(target.querySelector('button[aria-label="Export models"]')).toBeNull();
    const clear = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Clear filters')!;
    clear.focus();
    clear.click();
    flushSync();
    await Promise.resolve();
    expect(search.value).toBe('');
    expect(document.activeElement).toBe(search);
  });

  it('matches printed rounding boundaries and suppresses stale deltas in compareExport & leaderboardExport', async () => {
    const bench = await hub.api.bench('bn_3c90');
    const entries = await hub.api.benchEntries('bn_3c90');
    const staleEntry = entries.find((e) => e.fingerprint !== bench.fingerprint);
    const done = entries.filter((e) => e.status === 'done' && e.score && e.fingerprint === bench.fingerprint);
    const [a, b] = done;

    // Leaderboard export keeps rank: null for stale/earlier-version entries and maps axes by axis id even if reordered.
    if (staleEntry && a) {
      const reordered: BenchEntry = {
        ...a,
        score: a.score ? { ...a.score, axes: [...a.score.axes].reverse() } : a.score,
      };
      const lb = leaderboardExport(bench, [reordered, staleEntry]);
      const json = lb.json as { entries: Array<{ id: string; rank: number | null; earlier_version: boolean; axes: Record<string, number | null> }> };
      expect(json.entries[0]!.rank).toBe(1);
      expect(json.entries[0]!.axes.accuracy).toBe(a.score!.axes.find((x) => x.axis === 'accuracy')!.value);
      expect(json.entries[1]!.earlier_version).toBe(true);
      expect(json.entries[1]!.rank).toBeNull();
    }

    // Compare export matches diffAsPrinted at rounding boundaries and omits deltas for earlier-version entries.
    if (a && b) {
      const cmp = compareExport(
        bench as Bench,
        [a, b],
        [
          { key: 'robustness', label: 'Robustness', values: [0.684, 0.676], fmt: (v) => v.toFixed(2), d: 2 },
          { key: 'calibration', label: 'Calibration', values: [0.686, 0.684], fmt: (v) => v.toFixed(2), d: 2 },
        ],
        [false, false],
        () => new Set<number>(),
        [],
        [undefined, undefined],
      );
      const json = cmp.json as { measures: Array<{ values: Array<{ delta: number | null; delta_printed: string | null }> }> };
      // 0.684 rounds to 0.68 and 0.676 rounds to 0.68 -> delta = 0, delta_printed = '±0'
      expect(json.measures[0]!.values[1]!.delta).toBe(0);
      expect(json.measures[0]!.values[1]!.delta_printed).toBe(diffAsPrinted(0.676, 0.684, 2));
      // 0.686 rounds to 0.69 and 0.684 rounds to 0.68 -> delta = -0.01, delta_printed = '−0.01'
      expect(json.measures[1]!.values[1]!.delta).toBe(-0.01);
      expect(json.measures[1]!.values[1]!.delta_printed).toBe(diffAsPrinted(0.684, 0.686, 2));

      // Stale entry -> delta and delta_printed are both null.
      const cmpStale = compareExport(
        bench as Bench,
        [a, b],
        [{ key: 'robustness', label: 'Robustness', values: [0.8, 0.7], fmt: (v) => v.toFixed(2), d: 2 }],
        [false, true],
        () => new Set<number>(),
        [],
        [undefined, undefined],
      );
      const staleJson = cmpStale.json as { measures: Array<{ values: Array<{ delta: number | null; delta_printed: string | null }> }> };
      expect(staleJson.measures[0]!.values[1]!.delta).toBeNull();
      expect(staleJson.measures[0]!.values[1]!.delta_printed).toBeNull();
    }
  });

  it('announces added entries, restores focus when BenchCompare reaches max entries, switches baseline via phone picker, and keeps DotRows Escape scoped', async () => {
    vi.spyOn(router, 'go').mockImplementation((path) => {
      router.route = parse(`#${path}`);
    });
    const entries = (await hub.api.benchEntries('bn_3c90')).filter((e) => e.status === 'done' && e.score);
    expect(entries.length).toBeGreaterThanOrEqual(4);
    const [e0, e1, e2, e3] = entries;
    mountRoute(BenchCompare, `#/benchmarks/bn_3c90/compare?entries=${e0!.id},${e1!.id},${e2!.id}`);
    await settle(18);

    // Adding the 4th entry unmounts the adder <select>: announce it and move focus to the new chip's pin button.
    const adder = target.querySelector<HTMLSelectElement>('select[aria-label="Add an entry to the comparison"]')!;
    expect(adder).not.toBeNull();
    adder.focus();
    adder.value = e3!.id;
    adder.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    await Promise.resolve();
    expect(target.querySelector('select[aria-label="Add an entry to the comparison"]')).toBeNull();
    expect(document.activeElement?.getAttribute('data-entry-id')).toBe(e3!.id);
    expect(target.querySelector('.sr-only[role="status"]')?.textContent).toContain(`Added ${e3!.name} to comparison.`);

    // Phone baseline picker in SideBySide changes the baseline entry and announces it.
    const baseSelect = target.querySelector<HTMLSelectElement>('.sbs-base select')!;
    expect(baseSelect).not.toBeNull();
    expect(baseSelect.value).toBe(e0!.id);
    baseSelect.value = e2!.id;
    baseSelect.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    await Promise.resolve();
    expect(baseSelect.value).toBe(e2!.id);
    expect(target.querySelector('.sr-only[role="status"]')?.textContent).toContain(`Baseline set to ${e2!.name}.`);

    // Pin entry 1, wait for the newly added entry's results to load, then focus a DotRows chart:
    // ArrowDown scrolls and points a row; first Escape clears only the row, second Escape unpins.
    await settle(6);
    const firstPin = target.querySelector<HTMLButtonElement>('button.ck')!;
    firstPin.click();
    flushSync();
    expect(firstPin.getAttribute('aria-pressed')).toBe('true');

    const dots = target.querySelector<HTMLElement>('.dots[role="group"]')!;
    expect(dots).not.toBeNull();
    const firstDotRow = dots.querySelector<HTMLElement>('li.row')!;
    const scrollSpy = vi.fn();
    firstDotRow.scrollIntoView = scrollSpy;
    dots.focus();
    dots.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true }));
    flushSync();
    expect(firstDotRow.classList.contains('on')).toBe(true);
    expect(scrollSpy).toHaveBeenCalledWith({ block: 'nearest' });

    const firstEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    dots.dispatchEvent(firstEsc);
    flushSync();
    expect(firstEsc.defaultPrevented).toBe(true);
    expect(firstDotRow.classList.contains('on')).toBe(false);
    // Pinned chip must still be pinned because DotRows consumed the first Escape.
    expect(firstPin.getAttribute('aria-pressed')).toBe('true');

    const secondEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    dots.dispatchEvent(secondEsc);
    flushSync();
    expect(secondEsc.defaultPrevented).toBe(true);
    expect(firstPin.getAttribute('aria-pressed')).toBe('false');
  });

  it('clears Benchmark leaderboard Compare selection on Escape before navigating away and restores focus if Clear unmounts', async () => {
    mountRoute(Benchmark, '#/benchmarks/bn_3c90');
    await settle(18);

    // Without a selection, Escape is not intercepted by Benchmark.
    const idleEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    document.dispatchEvent(idleEsc);
    expect(idleEsc.defaultPrevented).toBe(false);

    const boxes = [...target.querySelectorAll<HTMLInputElement>('table.board tbody input[type="checkbox"]')];
    expect(boxes.length).toBeGreaterThanOrEqual(2);
    boxes[0]!.click();
    boxes[1]!.click();
    flushSync();
    const clearBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Clear selection"]')!;
    expect(clearBtn).not.toBeNull();
    clearBtn.focus();

    const clearEsc = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    document.dispatchEvent(clearEsc);
    flushSync();
    await Promise.resolve();
    expect(clearEsc.defaultPrevented).toBe(true);
    expect(target.querySelector('button[aria-label="Clear selection"]')).toBeNull();
    expect(document.activeElement).not.toBe(document.body);
  });
});
