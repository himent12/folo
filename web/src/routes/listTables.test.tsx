// @vitest-environment happy-dom
/// <reference types="node" />
// The Models and Datasets lists (the Runs list is in runsList.test.tsx).
// The Models table at every width: a header cell and the cells under it hide together, so each value sits under its
// own heading. Datasets are cards (design/UNIFIED.md §4), checked for the same promises in their own shape. The responsive column rule is base.css's (loaded here as the app loads it); the
// sort headers are ui/SortHeader components, which a page-scoped rule could not reach (round 16's Models at 1024).
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { flush as flushSync, type Component } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeAll, describe, expect, it, onTestFinished, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { page as pageMeta } from '$lib/shell/crumbs';
import Datasets from './Datasets';
import Models from './Models';

// Each side of the .hide-lg (< 1280) and .hide-md (< 1024) boundaries, and a tablet (820).
const WIDTHS = [1280, 1279, 1024, 1023, 820];

let target: HTMLElement;
type Mounted = { unmount(): void };
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

/** The headers without a fixed width (the check column and the empty last one aside). */
const loose = (table: HTMLTableElement) =>
  [...table.querySelectorAll<HTMLElement>('thead th')]
    .filter((th) => !th.classList.contains('check') && !th.classList.contains('rest') && !th.style.width)
    .map((th) => th.textContent?.trim());

const LISTS: [string, Component<Record<string, never>>, string, string][] = [['Models', Models, '/models', 'table.models']];
const SOURCE: Record<string, string> = { Models: 'src/routes/Models.tsx' };

/**
 * The bodies of the component's own rules (its css`` block, comments dropped) whose selector `el` matches as rendered.
 * vitest hands the scoped stylesheets back empty, so a rule is checked by its text against the element it must reach.
 */
function ownRules(file: string, el: Element): string[] {
  const src = readFileSync(resolve(process.cwd(), file), 'utf8');
  const css = src.slice(src.lastIndexOf('css`')).replace(/\/\*[\s\S]*?\*\//g, '');
  const matches = (sel: string) => {
    try {
      return el.matches(sel.trim());
    } catch {
      return false;
    }
  };
  return [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].filter(([, sel]) => sel!.split(',').some(matches)).map(([, , body]) => body!);
}
// Every column but the flexible ones carries its width on the header row, which loading and loaded share, so a column
// sits at the same x in each (round 21: runs-empty and models-empty set their columns apart).
const FLEX: Record<string, string[]> = { Models: ['Model'] };
/** The page's own create action; Models has none. */
const CREATE: Record<string, string | undefined> = {};

describe('list tables: one geometry in every state', () => {
  for (const [name, page, path, sel] of LISTS) {
    it(`${name}: loaded and loading, one visible cell per visible header at every width, and the same fixed widths`, async () => {
      await open(page, path, new MockApi('normal', 0));
      const loaded = await table(sel, (t) => !!t.querySelector('tbody tr[data-row]'), 'the rows');
      expectAligned(loaded);
      // Each keyboard row names its entity: the list resumes at it by id after Esc or Back (ui/rowNav, shell F6).
      expect([...loaded.querySelectorAll<HTMLElement>('tr[data-row]')].filter((r) => !r.dataset.id)).toEqual([]);
      expect(loose(loaded)).toEqual(FLEX[name]);
      // A long name cuts itself, whole in its tooltip, instead of sizing its column (w2-verify-studio N4: a
      // 61-character dataset name left the path 55 px, "…/da…" on every row; a 68-character model name pushed
      // Started from and Created out of the frame).
      for (const row of loaded.querySelectorAll<HTMLElement>('tr[data-row]')) {
        const cell = row.querySelector('td.name-cell');
        const link = cell?.querySelector('a.name.truncate');
        expect(link?.getAttribute('title'), `the name of ${row.dataset.id}`).toBeTruthy();
        // Whole in its tooltip: the name itself, first (a published model's description follows it).
        expect(link!.getAttribute('title')!.startsWith(row.getAttribute('aria-label')!), `the tooltip of ${row.dataset.id}`).toBe(true);
        // The cell is bounded, so the name cuts itself instead of sizing its column.
        expect(ownRules(SOURCE[name]!, cell!).some((b) => /max-width:\s*0\s*;/.test(b)), `a bounded name cell in ${name}`).toBe(true);
        // And the name is a block in it, or its ellipsis never applies and the text spills past the cell.
        expect(ownRules(SOURCE[name]!, link!).some((b) => /display:\s*block\s*;/.test(b)), `a block name link in ${name}`).toBe(true);
      }

      await open(page, path, new MockApi('loading'));
      const skeleton = target.querySelector<HTMLTableElement>(sel)!;
      expectAligned(skeleton);
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
      // The page owns the create action while mounted (counted by shell/crumbs).
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

describe('Datasets: cards in every state', () => {
  const cards = () => [...target.querySelectorAll<HTMLElement>('.cards > .card')];

  it('loaded and loading: one card per dataset (its name a link that is whole in its tooltip), the same grid and toolbar while loading', async () => {
    const api = new MockApi('normal', 0);
    const list = await api.datasets();
    await open(Datasets, '/datasets', api);
    await vi.waitFor(() => {
      flushSync();
      if (cards().length !== list.length) throw new Error('still waiting for the cards');
    });
    for (const card of cards()) {
      const d = list.find((x) => x.id === card.dataset.id)!;
      const link = card.querySelector<HTMLAnchorElement>('a.name')!;
      expect(link.getAttribute('href')).toBe(`#/datasets/${d.id}`);
      expect(link.getAttribute('title')).toBe(d.name);
      // The format as a tag, the images and classes under the name; the path and splits are the dataset's page's.
      expect(card.querySelector('.tag')?.textContent).toBe(d.format.toUpperCase());
      expect(card.textContent).toContain(`${d.classes.length} ${d.classes.length === 1 ? 'class' : 'classes'}`);
      expect(card.textContent).not.toContain(d.path);
      expect(card.querySelectorAll('.mosaic .tile')).toHaveLength(4);
    }
    // The header says what the page holds, and a list nothing narrows has no line under it.
    expect(target.querySelector('header .sub')?.textContent).toContain(`${list.length} datasets`);
    expect(target.querySelector('.list-foot')).toBeNull();

    await open(Datasets, '/datasets', new MockApi('loading'));
    expect(target.querySelectorAll('.cards > .card[aria-hidden="true"]').length).toBeGreaterThan(0);
    expect(target.querySelector<HTMLInputElement>('.list-toolbar .lt-search input')!.disabled).toBe(false);
  });

  it('draws each mosaic from the first images of its dataset, as thumbnails with their boxes', async () => {
    // happy-dom never reports a card in view: without an observer a card reads its mosaic at once.
    vi.stubGlobal('IntersectionObserver', undefined);
    onTestFinished(() => {
      vi.unstubAllGlobals();
    });
    const api = new MockApi('normal', 0);
    const asked = vi.spyOn(api, 'datasetImages');
    await open(Datasets, '/datasets', api);
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('.cards .mosaic .dsimg')) throw new Error('still waiting for a mosaic');
    });
    expect(asked.mock.calls.every(([, q]) => q.limit === 4 && q.offset === undefined)).toBe(true);
  });

  it('narrowed by the search, says under the cards how many of how many it shows', async () => {
    const api = new MockApi('normal', 0);
    const list = await api.datasets();
    await open(Datasets, '/datasets', api);
    await vi.waitFor(() => {
      flushSync();
      if (cards().length !== list.length) throw new Error('still waiting for the cards');
    });
    const input = target.querySelector<HTMLInputElement>('.list-toolbar .lt-search input')!;
    input.value = list[0]!.name;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    const n = cards().length;
    expect(n).toBeLessThan(list.length);
    expect(target.querySelector('.list-foot')?.textContent).toContain(`${n} of ${list.length} datasets`);
    // Nothing matches: the state says so where the cards were, and the line still counts what is shown.
    input.value = 'zebra';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(cards()).toHaveLength(0);
    expect(target.querySelector('.list-foot')?.textContent).toContain(`0 of ${list.length} datasets`);
  });

  it('empty or failed: the toolbar stays, disabled, one panel holds the state, and offline Add dataset says why it is off', async () => {
    for (const scenario of ['empty', 'error'] as const) {
      await open(Datasets, '/datasets', new MockApi(scenario, 0));
      await vi.waitFor(() => {
        flushSync();
        if (!target.querySelector('.state')?.textContent?.trim()) throw new Error(`still waiting for the ${scenario} state`);
      });
      expect(cards()).toHaveLength(0);
      const bar = target.querySelector('.list-toolbar')!;
      expect(bar.querySelector<HTMLInputElement>('.lt-search input')!.disabled).toBe(true);
      const radios = [...bar.querySelectorAll<HTMLButtonElement>('.lt-filter [role=radio]')];
      expect(radios.length).toBeGreaterThan(0);
      expect(radios.every((r) => r.disabled)).toBe(true);
      expect(bar.querySelectorAll('.lt-filter .meta').length).toBe(0);
    }
    expect(pageMeta.ownsCreate).toBe(true);
    const add = await vi.waitFor(() => {
      flushSync();
      const b = [...target.querySelectorAll<HTMLButtonElement>('header button')].find((x) => x.textContent?.includes('Add dataset'));
      if (!b?.disabled) throw new Error('not disabled yet');
      return b;
    });
    expect(document.getElementById(add.getAttribute('aria-describedby') ?? '')?.textContent).toBeTruthy();
  });
});

describe('Models: groups by evaluation dataset and split (lib/scoreSummary)', () => {
  const groups = () => [...target.querySelectorAll('table.models tr.group th')];
  const loaded = (scenario: 'normal' | 'train-split') =>
    open(Models, '/models', new MockApi(scenario, 0)).then(() => table('table.models', () => !!groups()[0]?.textContent?.includes('split of'), 'the groups'));

  it('ranks each group apart, the unranked ones last and never marked best, and marks an exact tie "joint best" on both', async () => {
    await loaded('normal');
    expect(groups().map((th) => th.firstElementChild?.textContent)).toEqual([
      'Validation split of fasteners',
      'Validation split of shapes',
      'Validation split of shapes-hard',
      'Published weights · split unknown',
      'Imported · split unknown',
    ]);
    expect(groups().map((th) => th.textContent!.includes('not ranked'))).toEqual([false, false, false, true, true]);
    const published = [...target.querySelectorAll('table.models tbody')].find((b) => b.querySelector('tr.group')?.textContent?.includes('Published weights'))!;
    expect(published.querySelector('a.name')?.textContent).toBe('folo-n-coco');
    expect(published.querySelector('.best')).toBeNull();
    // Round 25: fasteners-n-best and fasteners-n-f16 have identical scores, and only one was marked "best".
    const mark = (name: string) => [...target.querySelectorAll('table.models tbody tr')].find((tr) => tr.querySelector('a.name')?.textContent === name)!.querySelector('.best')?.textContent ?? null;
    expect(mark('fasteners-n-best')).toBe('joint best');
    expect(mark('fasteners-n-f16')).toBe('joint best');
  });

  it("says where to put a .folo file to import it, with the server's own folder when it gives its data dir", async () => {
    const api = new MockApi('empty', 0);
    const system = api.system.bind(api);
    api.system = async () => ({ ...(await system()), data_dir: 'D:/folo-data' });
    await open(Models, '/models', api);
    const state = await table('table.models', (t) => !!t.querySelector('tr.state')?.textContent?.includes('D:/folo-data'), 'the empty state');
    expect(state.querySelector('tr.state')!.textContent).toContain('To import a .folo file, copy it into D:/folo-data/models on the server');
    // The header holds facts, not help (design/UNIFIED.md §2): with no models there are none, and the help is the
    // empty state's alone.
    expect(target.querySelector('header')?.textContent).not.toContain('To import a .folo file');
  });

  it("marks a run's last checkpoint that is its best epoch as the same weights, not a joint best", async () => {
    const api = new MockApi('normal', 0);
    const [models, runs] = [api.models.bind(api), api.runs.bind(api)];
    const best = (await models()).find((m) => m.id === 'm_fast_n')!;
    api.models = async () => [...(await models()), { ...best, id: 'm_fast_n-last', name: 'fasteners-n-last' }];
    api.runs = async () => (await runs()).map((r) => (r.id === best.run_id ? { ...r, status: 'completed', best: { epoch: r.epochs, map50_95: best.metrics!.map50_95 } } : r));
    await open(Models, '/models', api);
    await table('table.models', (t) => !!t.textContent?.includes('fasteners-n-last'), 'the last checkpoint');
    await vi.waitFor(() => expect(target.querySelector('table.models')?.textContent).toContain('same as best'));
    const mark = (name: string) => [...target.querySelectorAll('table.models tbody tr')].find((tr) => tr.querySelector('a.name')?.textContent === name)!.querySelector('.best')?.textContent ?? null;
    expect(mark('fasteners-n-last')).toBe('same as best');
    expect(mark('fasteners-n-best')).toBe('joint best'); // still tied with fasteners-n-f16, a different file
  });

  it('ranks a training-split model apart from the validation ones, under its own heading', async () => {
    await loaded('train-split');
    expect(groups().map((th) => th.firstElementChild?.textContent)).toContain('Training split of fasteners-mini');
  });
});
