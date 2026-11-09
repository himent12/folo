// @vitest-environment happy-dom
/// <reference types="node" />
// The list tables (Runs, Models, Datasets) at every width: a header cell and the cells under it hide together, so each
// value sits under its own heading. The responsive column rule is base.css's (loaded here as the app loads it); the
// sort headers are ui/SortHeader components, which a page-scoped rule could not reach (round 16's Models at 1024).
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { flushSync, mount, unmount, type Component } from 'svelte';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router.svelte';
import { page as pageMeta } from '$lib/shell/crumbs.svelte';
import Datasets from './Datasets.svelte';
import Models from './Models.svelte';
import Runs from './Runs.svelte';

const WIDTHS = [1440, 1280, 1279, 1024, 1023, 820];

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

beforeAll(() => {
  // Read from disk: vitest hands CSS imports (`?raw` included) back empty.
  const style = document.createElement('style');
  style.textContent = readFileSync(resolve(process.cwd(), 'src/lib/design/base.css'), 'utf8');
  document.head.appendChild(style);
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target?.remove();
});

async function open(page: Component, path: string, api: MockApi) {
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse(`#${path}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  c = mount(page, { target });
  flushSync();
}

/** Resizes the window. happy-dom keeps computed styles until the DOM changes, so a mutation drops them. */
function resize(width: number) {
  (window as unknown as { happyDOM: { setViewport(v: { width: number; height: number }): void } }).happyDOM.setViewport({ width, height: 900 });
  document.body.dataset.width = String(width);
}

const shown = (el: Element) => getComputedStyle(el).display !== 'none';
const span = (el: Element) => Number(el.getAttribute('colspan') ?? 1);

/** Visible header cells, and the visible cell count of each body row that is one cell per column. */
function columns(table: HTMLTableElement) {
  const head = [...table.querySelectorAll('thead th')].filter(shown).reduce((a, th) => a + span(th), 0);
  const rows = [...table.querySelectorAll('tbody tr')]
    .map((tr) => [...tr.children])
    .filter((cells) => cells.every((td) => span(td) === 1))
    .map((cells) => cells.filter(shown).length);
  return { head, rows };
}

function expectAligned(table: HTMLTableElement) {
  const heads: number[] = [];
  for (const width of WIDTHS) {
    resize(width);
    const { head, rows } = columns(table);
    heads.push(head);
    expect(rows.length, `rows at ${width}`).toBeGreaterThan(0);
    for (const n of rows) expect(n, `cells per row at ${width} px`).toBe(head);
  }
  // The rule is in force (columns do hide as the screen narrows), so the check above is not vacuous.
  expect(heads.at(-1)).toBeLessThan(heads[0]!);
}

const LISTS: [string, Component, string, string][] = [
  ['Models', Models as Component, '/models', 'table.models'],
  ['Runs', Runs as Component, '/runs', 'table.runs'],
  ['Datasets', Datasets as Component, '/datasets', 'table.ds'],
];

describe('list tables: headers and columns hide together', () => {
  for (const [name, page, path, sel] of LISTS) {
    it(`${name}: loaded rows have one visible cell per visible header at every width`, async () => {
      await open(page, path, new MockApi('normal', 0));
      const table = await vi.waitFor(() => {
        flushSync();
        const t = target.querySelector<HTMLTableElement>(sel);
        if (!t?.querySelector('tbody tr[data-row]')) throw new Error('rows not loaded');
        return t;
      });
      expectAligned(table);
    });

    it(`${name}: the loading skeleton keeps the same columns`, async () => {
      await open(page, path, new MockApi('loading'));
      expectAligned(target.querySelector<HTMLTableElement>(sel)!);
    });
  }

  it('Models at 1024 hides Classes, GFLOPs and Started from in the header too', async () => {
    await open(Models as Component, '/models', new MockApi('normal', 0));
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('table.models tbody tr[data-row]')) throw new Error('rows not loaded');
    });
    resize(1024);
    const heads = [...target.querySelectorAll('table.models thead th')].filter(shown).map((th) => th.textContent?.trim());
    expect(heads).not.toContain('Classes');
    expect(heads).not.toContain('GFLOPs');
    expect(heads).not.toContain('Started from');
    expect(heads).toContain('Params');
  });
});

describe('list tables: one geometry in every state', () => {
  // Every column but the flexible ones carries its width on the header row, which loading, empty, error and loaded
  // share, so a column sits at the same x in each (round 21: runs-empty and models-empty set their columns apart).
  const FLEX: Record<string, string[]> = { Runs: ['Run'], Models: ['Model'], Datasets: ['Path', 'Class balance'] };
  for (const [name, page, path, sel] of LISTS) {
    for (const scenario of ['normal', 'empty', 'loading', 'error'] as const) {
      it(`${name} (${scenario}): every header but ${FLEX[name]!.join(' and ')} has a fixed width`, async () => {
        await open(page, path, new MockApi(scenario, 0));
        const table = await vi.waitFor(() => {
          flushSync();
          const t = target.querySelector<HTMLTableElement>(sel);
          if (!t) throw new Error('no table');
          return t;
        });
        const loose = [...table.querySelectorAll<HTMLElement>('thead th')]
          .filter((th) => !th.classList.contains('check') && !th.style.width)
          .map((th) => th.textContent?.trim());
        expect(loose).toEqual(FLEX[name]);
      });
    }
  }

  it('Models: with no rows to sort, the headers carry no sort mark', async () => {
    for (const scenario of ['empty', 'error'] as const) {
      await open(Models as Component, '/models', new MockApi(scenario, 0));
      await vi.waitFor(() => {
        flushSync();
        if (!target.querySelector('table.models tr.state')) throw new Error('no state yet');
      });
      expect(target.querySelectorAll('table.models [aria-sort]').length, scenario).toBe(0);
      expect(target.querySelectorAll('table.models .sort.on').length, scenario).toBe(0);
      unmount(c!);
      c = undefined;
      target.remove();
    }
  });
});

describe('responsive columns are one shared rule', () => {

  it('no component styles its own .hide-* columns (a scoped rule misses headers drawn by SortHeader)', () => {
    const src = resolve(process.cwd(), 'src');
    const files: string[] = [];
    const walk = (dir: string) => {
      for (const name of readdirSync(dir)) {
        const p = join(dir, name);
        if (statSync(p).isDirectory()) walk(p);
        else if (name.endsWith('.svelte')) files.push(p);
      }
    };
    walk(src);
    const offenders = files.filter((f) => {
      const styles = [...readFileSync(f, 'utf8').matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)].map((m) => m[1]).join('\n');
      return /\.hide-(?:xl|lg|md|sm)\b/.test(styles.replace(/\/\*[\s\S]*?\*\//g, ''));
    });
    expect(offenders.map((f) => relative(src, f))).toEqual([]);
  });
});

describe('create actions while the server is offline', () => {
  it('Datasets: Add dataset is disabled and says why', async () => {
    await open(Datasets as Component, '/datasets', new MockApi('error', 0));
    const add = await vi.waitFor(() => {
      flushSync();
      const b = [...target.querySelectorAll<HTMLButtonElement>('header button')].find((x) => x.textContent?.includes('Add dataset'));
      if (!b?.disabled) throw new Error('not disabled yet');
      return b;
    });
    const why = document.getElementById(add.getAttribute('aria-describedby') ?? '');
    expect(why?.textContent).toBe('Server offline');
  });

  it('Runs: New run is disabled and says why', async () => {
    await open(Runs as Component, '/runs', new MockApi('error', 0));
    const run = await vi.waitFor(() => {
      flushSync();
      const b = [...target.querySelectorAll<HTMLButtonElement>('header button')].find((x) => x.textContent?.includes('New run'));
      if (!b?.disabled) throw new Error('not disabled yet');
      return b;
    });
    expect(document.getElementById(run.getAttribute('aria-describedby') ?? '')?.textContent).toBe('Server offline');
  });
});

describe('one create action per corner', () => {
  for (const [name, page, path] of [
    ['Runs', Runs, '/runs'],
    ['Datasets', Datasets, '/datasets'],
  ] as [string, Component, string][]) {
    it(`${name} owns the create action while mounted, so the top bar's New run steps aside`, async () => {
      expect(pageMeta.ownsCreate).toBe(false);
      await open(page, path, new MockApi('error', 0));
      flushSync();
      expect(pageMeta.ownsCreate).toBe(true);
      unmount(c!);
      c = undefined;
      flushSync();
      expect(pageMeta.ownsCreate).toBe(false);
    });
  }

  it('Models has no create action: the top bar keeps its New run', async () => {
    await open(Models as Component, '/models', new MockApi('normal', 0));
    flushSync();
    expect(pageMeta.ownsCreate).toBe(false);
  });
});

describe('Runs: row checkboxes', () => {
  async function loaded() {
    await open(Runs as Component, '/runs', new MockApi('normal', 0));
    return vi.waitFor(() => {
      flushSync();
      const t = target.querySelector<HTMLTableElement>('table.runs');
      if (!t?.querySelector('tbody tr[data-row]')) throw new Error('rows not loaded');
      return t;
    });
  }

  it('the table is "picking" (every checkbox shown) once a run is selected, and not before', async () => {
    const table = await loaded();
    expect(table.classList.contains('picking')).toBe(false);
    const box = table.querySelector<HTMLInputElement>('tbody tr[data-row] .check input')!;
    box.click();
    flushSync();
    expect(table.classList.contains('picking')).toBe(true);
    expect(box.closest('tr')!.classList.contains('on')).toBe(true);
  });

  it('at 820 with a run selected: the same columns as before, the checkbox column reserved and Created hidden', async () => {
    const table = await loaded();
    resize(820);
    const heads = () => [...table.querySelectorAll('thead th')].filter(shown).map((th) => th.textContent?.trim());
    const before = heads();
    table.querySelector<HTMLInputElement>('tbody tr[data-row] .check input')!.click();
    flushSync();
    resize(820);
    expect(table.classList.contains('picking')).toBe(true);
    // Selecting never adds a column (the checkbox column's width is always there), so nothing is pushed off the edge.
    expect(heads()).toEqual(before);
    expect(before[0]).toBe('Select for comparison');
    expect(before).not.toContain('Created');
    // The last column holds a waiting run's queue menu (empty on the other rows).
    expect(before).toEqual(['Select for comparison', 'Run', 'Status', 'Progress', 'Best mAP50-95', 'Actions']);
    for (const n of columns(table).rows) expect(n).toBe(before.length);
  });

  it('a click anywhere in the checkbox cell selects without opening the run', async () => {
    const table = await loaded();
    const go = vi.spyOn(router, 'go');
    const label = table.querySelector<HTMLLabelElement>('tbody tr[data-row] .check label')!;
    label.click();
    flushSync();
    expect(go).not.toHaveBeenCalled();
    expect(label.closest('tr')!.classList.contains('on')).toBe(true);
    go.mockRestore();
  });
});
