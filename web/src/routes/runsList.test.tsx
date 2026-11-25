// @vitest-environment happy-dom
/// <reference types="node" />
// The Runs list (Models and Datasets: listTables.test.tsx).
// The list table at every width: a header cell and the cells under it hide together, so each value sits under its own
// heading. Runs drops its columns by container queries on its frame (Runs.tsx `.cq-*`), which happy-dom does not
// evaluate: the rules are read from the source (each threshold, its class and its `display: none`) and applied at both
// sides of every threshold to the headers and every row's cells.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { flush as flushSync, type Component } from 'fw';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { render } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { duration } from '$lib/format';
import { parse, router } from '$lib/router';
import { page as pageMeta } from '$lib/shell/crumbs';
import Runs from './Runs';


type Mounted = { unmount(): void };
let target: HTMLElement;
let c: Mounted | undefined;
const unmount = (m: Mounted) => m.unmount();

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

async function open(page: Component<Record<string, never>>, path: string, api: MockApi) {
  if (c) unmount(c);
  target?.remove();
  hub.api = api;
  hub.mode = 'mock';
  // The list writes its state to the URL: each test starts from its own.
  history.replaceState(history.state, '', `#${path}`);
  router.route = parse(`#${path}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  const P = page;
  c = render(() => <P />, target);
  flushSync();
}

/** The table at `sel` once `ready` holds for it. */
function table(sel: string, ready: (t: HTMLTableElement) => boolean, what: string) {
  return vi.waitFor(() => {
    flushSync();
    const t = target.querySelector<HTMLTableElement>(sel);
    if (!t || !ready(t)) throw new Error(`still waiting for ${what}`);
    return t;
  });
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

/**
 * The Runs table's container rules, read from its source (happy-dom does not evaluate `@container`): each threshold
 * with the column class it hides. A rule must be exactly `display: none` on one `.cq-*` class, so a wrong declaration
 * or a rule that hides something else fails here.
 */
function containerRules(): [number, string][] {
  const src = readFileSync(resolve(process.cwd(), 'src/routes/Runs.tsx'), 'utf8');
  const out: [number, string][] = [];
  for (const m of src.matchAll(/@container \(max-width: (\d+)px\) \{\s*\.runs :global\(\.(cq-[\w-]+)\) \{([^}]*)\}/g)) {
    expect(m[3]!.trim(), `the declaration under ${m[1]} px`).toBe('display: none;');
    out.push([Number(m[1]), m[2]!]);
  }
  return out;
}
/** The frame widths the columns leave at (Runs.tsx): the least important first, each with the heads it takes. */
const DROPS: [number, string, string[]][] = [
  [1139, 'cq-trend', ['Trend']],
  [1023, 'cq-dur', ['Duration']],
  [927, 'cq-created', ['Created']],
  [831, 'cq-ctx', ['Architecture', 'Dataset']],
];

/**
 * At each side of every container threshold the table uses: which headers show, and every body row shows exactly the
 * cells under those headers, so a header never hides without its cells or the other way round.
 */
function expectSameColumns(table: HTMLTableElement) {
  expect(containerRules()).toEqual(DROPS.map(([w, c]) => [w, c]));
  const cq = (el: Element) => [...el.classList].find((c) => c.startsWith('cq-'));
  const heads = [...table.querySelectorAll('thead th')];
  const rows = [...table.querySelectorAll('tbody tr')].map((tr) => [...tr.children]).filter((cells) => cells.every((td) => span(td) === 1));
  expect(rows.length, 'rows').toBeGreaterThan(0);
  // Each class is on the headers it names and on no other.
  for (const [, c, names] of DROPS) expect(heads.filter((th) => cq(th) === c).map((th) => th.textContent?.trim())).toEqual(names);
  for (const cells of rows) expect(cells.length).toBe(heads.length);
  const all = heads.map((th) => th.textContent?.trim());
  for (const width of [1140, ...DROPS.flatMap(([w]) => [w + 1, w])]) {
    const hidden = new Set(DROPS.filter(([w]) => width <= w).map(([, c]) => c));
    const visible = (el: Element) => !hidden.has(cq(el) ?? '');
    const shownHeads = heads.filter(visible).map((th) => th.textContent?.trim());
    const gone = DROPS.filter(([w]) => width <= w).flatMap(([, , names]) => names);
    expect(shownHeads, `heads at ${width} px`).toEqual(all.filter((h) => !gone.includes(h!)));
    for (const cells of rows) expect(cells.map(visible), `cells at ${width} px`).toEqual(heads.map(visible));
  }
}

/** The headers without a fixed width (the check column and the empty last one aside). */
const loose = (table: HTMLTableElement) =>
  [...table.querySelectorAll<HTMLElement>('thead th')]
    .filter((th) => !th.classList.contains('check') && !th.classList.contains('rest') && !th.style.width)
    .map((th) => th.textContent?.trim());

const LISTS: [string, Component<Record<string, never>>, string, string][] = [['Runs', Runs, '/runs', 'table.runs']];
// Every column but the flexible ones carries its width on the header row, which loading and loaded share, so a column
// sits at the same x in each (round 21: runs-empty and models-empty set their columns apart).
const FLEX: Record<string, string[]> = { Runs: ['Run'] };
/** The page's own create action. */
const CREATE: Record<string, string | undefined> = { Runs: 'New run' };

describe('list tables: one geometry in every state', () => {
  for (const [name, page, path, sel] of LISTS) {
    it(`${name}: loaded and loading, one visible cell per visible header at every width, and the same fixed widths`, async () => {
      await open(page, path, new MockApi('normal', 0));
      const loaded = await table(sel, (t) => !!t.querySelector('tbody tr[data-row]'), 'the rows');
      expectSameColumns(loaded);
      expect(loose(loaded)).toEqual(FLEX[name]);

      await open(page, path, new MockApi('loading'));
      const skeleton = target.querySelector<HTMLTableElement>(sel)!;
      expectSameColumns(skeleton);
      expect(loose(skeleton)).toEqual(FLEX[name]);
      // The toolbar is the same one, live.
      expect(target.querySelector<HTMLInputElement>('.list-toolbar .lt-search input')!.disabled).toBe(false);
    });

    // Round 24: an empty or failed list lost its toolbar (the table jumped ~48 px against loading) and kept a 9–11
    // column header row over a one-line message. The toolbar stays, disabled with no counts; the column heads go.
    it(`${name}: empty or failed, the toolbar stays, disabled, the frame holds the state alone, and offline the create action says why it is off`, async () => {
      for (const scenario of ['empty', 'error'] as const) {
        await open(page, path, new MockApi(scenario, 0));
        const frame = await table(sel, (t) => !!t.querySelector('tr.state'), `the ${scenario} state`);
        expect(frame.querySelector('thead'), 'column heads').toBeNull();
        expect(frame.querySelectorAll('[aria-sort]').length).toBe(0);
        const bar = target.querySelector('.list-toolbar')!;
        expect(bar.querySelector<HTMLInputElement>('.lt-search input')!.disabled).toBe(true);
        const radios = [...bar.querySelectorAll<HTMLButtonElement>('.lt-filter [role=radio]')];
        expect(radios.length).toBeGreaterThan(0);
        expect(radios.every((r) => r.disabled)).toBe(true);
        // No counts over a list that has nothing to count.
        expect(bar.querySelectorAll('.lt-filter .meta').length).toBe(0);
      }
      // The page owns the create action while mounted (counted by shell/crumbs; the top bar has none).
      const label = CREATE[name];
      expect(pageMeta.ownsCreate).toBe(!!label);
      if (label) {
        const add = await vi.waitFor(() => {
          flushSync();
          const b = [...target.querySelectorAll<HTMLButtonElement>('header button')].find((x) => x.textContent?.includes(label));
          if (!b?.disabled) throw new Error('not disabled yet');
          return b;
        });
        expect(document.getElementById(add.getAttribute('aria-describedby') ?? '')?.textContent).toBeTruthy();
      }
      unmount(c!);
      c = undefined;
      flushSync();
      expect(pageMeta.ownsCreate).toBe(false);
    });
  }
});

describe('Runs: row checkboxes', () => {
  it('select from anywhere in their cell without opening the run, and show every box then, with the same columns', async () => {
    await open(Runs, '/runs', new MockApi('normal', 0));
    const runs = await table('table.runs', (t) => !!t.querySelector('tbody tr[data-row]'), 'the rows');
    resize(820);
    const heads = () => [...runs.querySelectorAll('thead th')].filter(shown).map((th) => th.textContent?.trim());
    const before = heads();
    // The checkbox column's width is always there.
    expect(before[0]).toBe('Select for comparison');
    expect(runs.classList.contains('picking')).toBe(false);
    const go = vi.spyOn(router, 'go');
    const label = runs.querySelector<HTMLLabelElement>('tbody tr[data-row] .check label')!;
    label.click();
    flushSync();
    // The run does not open; the selection joins the list's URL (it survives opening a run and coming back).
    const id = label.closest('tr')!.dataset.id!;
    expect(go.mock.calls).toEqual([[`/runs?sel=${id}`, { replace: true }]]);
    expect(label.closest('tr')!.classList.contains('on')).toBe(true);
    expect(runs.classList.contains('picking')).toBe(true);
    // Selecting never adds a column, so nothing is pushed off the edge.
    resize(820);
    expect(heads()).toEqual(before);
    for (const n of columns(runs).rows) expect(n).toBe(before.length);
    go.mockRestore();
  });
});

describe('Runs: the status filter of a deep link', () => {
  // `#/runs?status=finished` (the filter's label, not its value) or a run status that is no filter (`stopped`) left
  // `groups[filter]` undefined, and the page threw on its first render.
  for (const status of ['finished', 'stopped', 'constructor']) {
    it(`?status=${status} is no filter: every run shows`, async () => {
      await open(Runs, `/runs?status=${status}`, new MockApi('normal', 0));
      const rows = await vi.waitFor(() => {
        flushSync();
        const r = target.querySelectorAll('table.runs tbody tr[data-id]');
        if (!r.length) throw new Error('rows not loaded');
        return r;
      });
      const all = await new MockApi('normal', 0).runs();
      expect(rows.length).toBe(all.length);
    });
  }
  it('?status=failed still filters', async () => {
    await open(Runs, '/runs?status=failed', new MockApi('normal', 0));
    const all = await new MockApi('normal', 0).runs();
    const failed = all.filter((r) => r.status === 'failed').length;
    expect(failed).toBeGreaterThan(0);
    await vi.waitFor(() => {
      flushSync();
      expect(target.querySelectorAll('table.runs tbody tr[data-id]').length).toBe(failed);
    });
  });
});

describe('Runs: sort, search and selection kept in the URL; row menus on every ended run', () => {
  const ids = () => [...target.querySelectorAll<HTMLElement>('table.runs tbody tr[data-id]')].map((r) => r.dataset.id!);
  const loaded = () =>
    vi.waitFor(() => {
      flushSync();
      if (!ids().length) throw new Error('rows not loaded');
    });

  it('opens as a link left it: search typed, sorted by best mAP high first (none last), the runs ticked', async () => {
    const all = await new MockApi('normal', 0).runs();
    const scored = all.filter((r) => r.best).sort((a, b) => b.best!.map50_95 - a.best!.map50_95);
    await open(Runs, `/runs?sort=map&dir=desc&sel=${scored[0]!.id}`, new MockApi('normal', 0));
    await loaded();
    expect(ids().slice(0, scored.length)).toEqual(scored.map((r) => r.id));
    expect(target.querySelector('th[aria-sort]')?.textContent).toContain('Best mAP50-95');
    expect(target.querySelector('th[aria-sort]')?.getAttribute('aria-sort')).toBe('descending');
    expect(target.querySelector<HTMLInputElement>(`tr[data-id="${scored[0]!.id}"] input[type="checkbox"]`)?.checked).toBe(true);
  });

  it('sorts on a header click, the second click turns it around, and the URL says so', async () => {
    await open(Runs, '/runs?q=shapes', new MockApi('normal', 0));
    await loaded();
    expect(target.querySelector<HTMLInputElement>('input[aria-label="Filter runs"]')?.value).toBe('shapes');
    const go = vi.spyOn(router, 'go');
    const head = () => [...target.querySelectorAll<HTMLButtonElement>('th button.sort')].find((b) => b.textContent?.includes('Created'))!;
    head().click();
    flushSync();
    expect(go).toHaveBeenLastCalledWith('/runs?q=shapes&sort=created&dir=asc', { replace: true });
    head().click();
    flushSync();
    expect(go).toHaveBeenLastCalledWith('/runs?q=shapes', { replace: true });
    go.mockRestore();
  });

  it('keeps two ticked runs in the URL once, with their commas (the URL spells them %2C)', async () => {
    await open(Runs, '/runs', new MockApi('normal', 0));
    await loaded();
    const go = vi.spyOn(router, 'go').mockImplementation((path, o) => {
      // As the router does: the URL's own spelling of the query comes back as the route.
      history.replaceState(history.state, '', `#${path}`);
      router.route = parse(location.hash);
      void o;
    });
    const boxes = [...target.querySelectorAll<HTMLInputElement>('table.runs tbody tr[data-row] .check input')];
    boxes[0]!.click();
    flushSync();
    boxes[1]!.click();
    flushSync();
    expect(go).toHaveBeenCalledTimes(2);
    expect(go).toHaveBeenLastCalledWith(`/runs?sel=${ids()[0]},${ids()[1]}`, { replace: true });
    go.mockRestore();
  });

  it('keeps the runs ticked for Compare across a run opened and left with Escape (a bare /runs right after it)', async () => {
    const api = new MockApi('normal', 0);
    const ids = (await api.runs()).slice(0, 2).map((r) => r.id);
    await open(Runs, `/runs?sel=${ids.join(',')}`, api);
    await loaded();
    // Enter on a row opens the run (a new entry); the App swaps the list for the run page.
    router.go(`/runs/${ids[0]}`);
    unmount(c!);
    // Escape there opens /runs (shell/shortcuts backPath), the next entry.
    router.go('/runs');
    const P = Runs;
    c = render(() => <P />, target);
    await loaded();
    await vi.waitFor(() => {
      flushSync();
      expect(location.hash).toBe(`#/runs?sel=${ids.join(',')}`);
      expect(target.querySelectorAll('td.check input:checked')).toHaveLength(2);
    });
  });

  it('gives a finished run its model, the Playground and Delete run in its ⋯', async () => {
    await open(Runs, '/runs', new MockApi('normal', 0));
    await loaded();
    const done = (await new MockApi('normal', 0).runs()).find((r) => r.status === 'completed' && r.model_id)!;
    target.querySelector<HTMLButtonElement>(`tr[data-id="${done.id}"] td.act button`)!.click();
    flushSync();
    const items = [...document.querySelectorAll('[role="menuitem"]')].map((e) => e.textContent?.trim());
    expect(items).toEqual(['Open model', 'Try in Playground', 'Delete run']);
  });

  it('gives the training run a ⋯ too: Open run, and Stop behind its confirm (v-data-runs N10a)', async () => {
    const api = new MockApi('normal', 0);
    await open(Runs, '/runs', api);
    await loaded();
    const running = (await api.runs()).find((r) => r.status === 'running')!;
    target.querySelector<HTMLButtonElement>(`tr[data-id="${running.id}"] td.act button`)!.click();
    flushSync();
    const items = [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')];
    expect(items.map((e) => e.textContent?.trim())).toEqual(['Open run', 'Stop']);
    expect(items[0]!.getAttribute('href')).toBe(`#/runs/${running.id}`);
    items[1]!.click();
    flushSync();
    expect(document.querySelector('dialog[open]')?.textContent).toContain(`Stop ${running.name}?`);
  });

  it("gives a resumed run's training time, without the night it sat stopped", async () => {
    const api = new MockApi('normal', 0);
    const runs = await api.runs();
    const done = runs.find((r) => r.status === 'completed')!;
    const resumed = { ...done, started: '2026-11-10T18:00:00Z', finished: '2026-11-11T08:30:00Z', paused_s: 14 * 3600 };
    vi.spyOn(api, 'runs').mockResolvedValue(runs.map((r) => (r.id === done.id ? resumed : r)));
    await open(Runs, '/runs', api);
    await vi.waitFor(() => {
      flushSync();
      const cells = [...target.querySelectorAll(`tr[data-id="${done.id}"] td`)].map((td) => td.textContent?.trim());
      expect(cells).toContain(duration(30 * 60));
    });
  });

  it("lets a phone row's name and dataset cut themselves, whole in their tooltips (studio-shell handover)", async () => {
    const realMatch = window.matchMedia;
    window.matchMedia = ((q: string) => ({ matches: q.includes('max-width: 767px'), media: q, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
    try {
      const api = new MockApi('normal', 0);
      const done = (await api.runs()).find((r) => r.status === 'completed' && r.dataset)!;
      await open(Runs, '/runs', api);
      const row = await vi.waitFor(() => {
        flushSync();
        const li = [...target.querySelectorAll<HTMLElement>('.rows li')].find((l) => l.querySelector('.name')?.textContent?.trim() === done.name);
        if (!li?.querySelector('.r-meta')?.textContent?.includes(done.dataset!.name)) throw new Error('rows not loaded');
        return li;
      });
      expect(row.querySelector('.name')?.getAttribute('title')).toBe(done.name);
      const ds = [...row.querySelectorAll<HTMLElement>('.r-meta .truncate')].find((e) => e.textContent?.trim() === done.dataset!.name);
      expect(ds?.getAttribute('title')).toBe(done.dataset!.name);
    } finally {
      window.matchMedia = realMatch;
    }
  });

  it("lets a desktop row's dataset cut itself under the name, whole in its tooltip (w2-verify-studio N4)", async () => {
    const api = new MockApi('normal', 0);
    const done = (await api.runs()).find((r) => r.status === 'completed' && r.dataset)!;
    await open(Runs, '/runs', api);
    const row = await vi.waitFor(() => {
      flushSync();
      const tr = target.querySelector<HTMLElement>(`tr[data-id="${done.id}"]`);
      if (!tr?.querySelector('.sub')?.textContent?.includes(done.dataset!.name)) throw new Error('rows not loaded');
      return tr;
    });
    const ds = [...row.querySelectorAll<HTMLElement>('.sub .truncate')].find((e) => e.textContent?.trim() === done.dataset!.name);
    expect(ds?.getAttribute('title')).toBe(done.dataset!.name);
  });

  it('names a removed dataset as the run recorded it, marked, never by its id', async () => {
    const api = new MockApi('normal', 0);
    const done = (await api.runs()).find((r) => r.status === 'completed' && r.dataset)!;
    await api.deleteDataset(done.config.dataset_id);
    await open(Runs, '/runs', api);
    await vi.waitFor(() => {
      flushSync();
      expect(target.querySelector(`tr[data-id="${done.id}"]`)?.textContent).toContain(`${done.dataset!.name} (removed)`);
    });
    expect(target.querySelector('table.runs tbody')!.textContent).not.toMatch(/ds_/);
  });

  it('reads the datasets again for a run on one made after the list loaded, never calling it removed (handover H1)', async () => {
    const api = new MockApi('normal', 0);
    const done = (await api.runs()).find((r) => r.status === 'completed' && r.dataset)!;
    const all = api.datasets.bind(api);
    let calls = 0;
    // The first read predates the dataset (made in another tab, or by a benchmark loop); later ones have it.
    vi.spyOn(api, 'datasets').mockImplementation(async (...a) => {
      const list = await all(...a);
      return calls++ === 0 ? list.filter((d) => d.id !== done.config.dataset_id) : list;
    });
    await open(Runs, '/runs', api);
    const seen: string[] = [];
    await vi.waitFor(() => {
      flushSync();
      const row = target.querySelector(`tr[data-id="${done.id}"]`)?.textContent ?? '';
      seen.push(row);
      if (calls < 2 || !row.includes(done.dataset!.name)) throw new Error('not read again yet');
    });
    flushSync();
    expect(target.querySelector(`tr[data-id="${done.id}"]`)?.textContent).not.toContain('(removed)');
    expect(seen.some((t) => t.includes('(removed)'))).toBe(false);
  });

  it('shows the compare checkboxes on a hovered or focused row, and on every row once one is ticked (design/UNIFIED.md §4)', async () => {
    await open(Runs, '/runs', new MockApi('normal', 0));
    await loaded();
    const table = target.querySelector('table.runs')!;
    // At rest no row shows its box: no class reveals them all for a list of two or more.
    expect(table.classList.contains('pickable')).toBe(false);
    expect(table.classList.contains('picking')).toBe(false);
    const src = readFileSync(resolve(process.cwd(), 'src/routes/Runs.tsx'), 'utf8');
    expect(src).toContain('.runs tbody tr:is(:hover, :focus-within, .on) .check input');
    target.querySelector<HTMLLabelElement>('tbody tr[data-row] .check label')!.click();
    flushSync();
    expect(table.classList.contains('picking')).toBe(true);
  });
});

describe('Runs: one fact per column (design/UNIFIED.md §4)', () => {
  it('gives each run its architecture (a name, in sans), dataset, progress as "38 / 100" and best score in columns of their own', async () => {
    const api = new MockApi('normal', 0);
    const runs = await api.runs();
    const done = runs.find((r) => r.status === 'completed' && r.best && r.dataset)!;
    await open(Runs, '/runs', api);
    const row = await vi.waitFor(() => {
      flushSync();
      const tr = target.querySelector<HTMLElement>(`table.runs tr[data-id="${done.id}"]`);
      if (!tr?.querySelector('.cell-cut')?.textContent?.includes(done.dataset!.name)) throw new Error('rows not loaded');
      return tr;
    });
    const heads = [...target.querySelectorAll('table.runs thead th')].map((th) => th.textContent?.trim());
    expect(heads).toEqual(['Select for comparison', 'Run', 'Architecture', 'Dataset', 'Status', 'Progress', 'Best mAP50-95', 'Trend', 'Duration', 'Created', 'Actions']);
    const cells = [...row.children];
    const arch = cells[2]!.querySelector('.ident')!;
    expect(arch.textContent).toBe(done.config.arch);
    expect(arch.classList.contains('mono')).toBe(false);
    expect(cells[2]!.getAttribute('title')).toMatch(/^(From |From scratch)/);
    expect(cells[5]!.textContent).toContain(`${done.epoch} / ${done.epochs}`);
    expect(cells[6]!.getAttribute('title')).toBe(`Best at epoch ${done.best!.epoch}`);
  });

  it('says what the page holds in its header, and only a narrowed list says what it shows under the table', async () => {
    const api = new MockApi('normal', 0);
    const runs = await api.runs();
    await open(Runs, '/runs', api);
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('header .sub')?.textContent?.includes(`${runs.length} runs`)) throw new Error('no facts yet');
    });
    // Every fact on a wider screen; a phone's one header row keeps the first two beside New run, so none is cut.
    const wide = target.querySelector('header .sub.wide-only')!.textContent!.trim();
    expect(wide.split(' · ').length).toBeGreaterThan(2);
    expect(wide).toContain('of training');
    expect(target.querySelector('header .sub.phone-only')!.textContent!.trim()).toBe(wide.split(' · ').slice(0, 2).join(' · '));
    expect(target.querySelector('.list-foot')).toBeNull();
    await open(Runs, '/runs?status=failed', api);
    const foot = await vi.waitFor(() => {
      flushSync();
      const f = target.querySelector('.list-foot');
      if (!f) throw new Error('no line yet');
      return f;
    });
    const failed = runs.filter((r) => r.status === 'failed').length;
    expect(foot.textContent?.trim()).toBe(`${failed} of ${runs.length} runs`);
    // A search that matches nothing still says what is shown.
    const input = target.querySelector<HTMLInputElement>('.lt-search input')!;
    expect(input.type).toBe('search');
    input.value = 'zzz-no-such-run';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(target.querySelector('.list-foot')?.textContent?.trim()).toBe(`0 of ${runs.length} runs`);
  });

  it('treats duplicate ?sel= ids as one selection and keeps Compare disabled until two distinct runs are selected', async () => {
    const api = new MockApi('normal', 0);
    const list = await api.runs();
    const first = list[0]!.id;
    const second = list[1]!.id;
    await open(Runs, `/runs?sel=${first},${first}`, api);
    await table('table.runs', (t) => !!t.querySelector('tbody tr[data-row]'), 'the rows');

    const compareEl = () => [...target.querySelectorAll<HTMLElement>('header a, header button')].find((el) => el.textContent?.includes('Compare'))!;
    expect(compareEl().querySelector('.count')?.textContent?.trim()).toBe('1');
    expect(compareEl().hasAttribute('disabled') || compareEl().getAttribute('aria-disabled') === 'true').toBe(true);
    expect(target.querySelector('header .cmp-hint')?.textContent).toContain('Select one more run to compare');

    // Picking a second distinct run enables Compare with both distinct ids.
    target.querySelector<HTMLInputElement>(`tr[data-id="${second}"] .check input`)!.click();
    flushSync();
    expect(compareEl().querySelector('.count')?.textContent?.trim()).toBe('2');
    expect(compareEl().getAttribute('href')).toBe(`#/runs/compare?ids=${first},${second}`);
    expect(compareEl().hasAttribute('disabled')).toBe(false);
    expect(compareEl().getAttribute('aria-disabled')).toBeNull();
  });

  it('offers a Clear selection button once runs are picked and explains the 4-run cap on disabled checkboxes', async () => {
    const api = new MockApi('normal', 0);
    await open(Runs, '/runs', api);
    const tbl = await table('table.runs', (t) => t.querySelectorAll('tbody tr[data-row]').length >= 5, 'five rows');
    expect(target.querySelector('header button[aria-label="Clear selection"]')).toBeNull();

    const boxes = [...tbl.querySelectorAll<HTMLInputElement>('tbody tr[data-row] .check input')];
    for (let i = 0; i < 4; i++) boxes[i]!.click();
    flushSync();
    expect(boxes[4]!.disabled).toBe(true);

    // Hovering a disabled checkbox's label surfaces the 4-run cap tooltip.
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    try {
      const cappedLabel = boxes[4]!.closest<HTMLLabelElement>('label.hit')!;
      cappedLabel.dispatchEvent(new Event('pointerenter'));
      vi.advanceTimersByTime(500);
      expect(document.getElementById('folo-tooltip')?.textContent).toBe('Compare up to 4 runs');
      cappedLabel.dispatchEvent(new Event('pointerleave'));
      vi.advanceTimersByTime(200);
    } finally {
      vi.useRealTimers();
    }

    const clear = target.querySelector<HTMLButtonElement>('header button[aria-label="Clear selection"]')!;
    expect(clear.textContent?.trim()).toBe('Clear');
    clear.focus();
    clear.click();
    flushSync();
    expect(tbl.querySelectorAll('td.check input:checked')).toHaveLength(0);
    expect(boxes[4]!.disabled).toBe(false);
    expect(target.querySelector('header button[aria-label="Clear selection"]')).toBeNull();
    expect(document.activeElement).toBe(target.querySelector('.lt-search input'));
  });
});
