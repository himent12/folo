// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router.svelte';
import RunCompare from './RunCompare.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;
const width = window.innerWidth;
const setWidth = (w: number) => (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: w });

async function until<T>(get: () => T | null | undefined | false, what = 'the condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

beforeEach(() => {
  target = document.createElement('div');
  document.body.appendChild(target);
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
  setWidth(width);
});

describe('Compare runs on a phone: two at a time', () => {
  it('keeps every row on the baseline and the run paged to; the baseline is its own choice, never the line in front', async () => {
    const runs = (await hub.api.runs()).filter((r) => r.status !== 'queued').slice(0, 4);
    expect(runs).toHaveLength(4);
    // Each run's best epoch tells its precision · recall cell apart from the others'.
    expect(new Set(runs.map((r) => r.best?.epoch)).size).toBe(4);
    const names = runs.map((r) => r.name);
    setWidth(390);
    router.route = parse(`#/runs/compare?ids=${runs.map((r) => r.id).join(',')}`);
    c = mount(RunCompare, { target });
    const table = await until(() => target.querySelector('#cmp-table table.two'), 'the paired table');
    // The precision · recall row needs the runs' metrics: wait for a value in it.
    await until(() => [...table.querySelectorAll('tbody tr')].some((tr) => tr.querySelector('td.k')?.textContent === 'Precision · recall' && tr.textContent?.includes('ep ')), 'precision · recall');

    /** Every body row has the setting cell and one cell per shown run; header names say which runs they are. */
    function check(baseline: string, other: string) {
      const heads = [...table.querySelectorAll('thead th')].slice(1).map((th) => th.querySelector('.th-name')?.textContent);
      expect(heads).toEqual([baseline, other]);
      // One geometry from the first render: the widths live on the header row (no colgroup whose keyed columns could
      // outlive a page of runs), the setting column fixed, the two run columns sharing the rest.
      expect(table.querySelector('colgroup')).toBeNull();
      const ths = [...table.querySelectorAll<HTMLElement>('thead th')];
      expect(ths).toHaveLength(3);
      expect(ths[0]!.style.width).toBe('104px');
      expect(ths.slice(1).every((th) => !th.style.width)).toBe(true);
      expect(ths[1]!.querySelector('.base-tag')?.textContent).toBe('Baseline');
      const rows = [...table.querySelectorAll('tbody tr:not(.fold)')];
      expect(rows.length).toBeGreaterThan(3);
      for (const tr of rows) expect(tr.querySelectorAll('td'), tr.querySelector('td.k')?.textContent ?? '').toHaveLength(3);
      // Each run's own epoch shows in its precision · recall cell: the values sit under their run's name.
      const pr = rows.find((tr) => tr.querySelector('td.k')?.textContent === 'Precision · recall')!;
      const epochOf = (name: string) => runs.find((r) => r.name === name)?.best?.epoch;
      const cells = [...pr.querySelectorAll('td')].slice(1).map((td) => td.textContent ?? '');
      [baseline, other].forEach((n, i) => {
        const ep = epochOf(n);
        expect(cells[i]).toContain(ep != null ? `ep ${ep}` : '—');
      });
    }

    check(names[0]!, names[1]!);
    // Page to the third run: the baseline stays, the second column is run C in every row.
    target.querySelector<HTMLButtonElement>('button[aria-label="Next run"]')!.click();
    flushSync();
    check(names[0]!, names[2]!);
    // Keep run D's line in front: emphasis is not the baseline, so the table still reads against run A beside run C.
    const chip = [...target.querySelectorAll<HTMLButtonElement>('.chip button.key')].find((b) => b.textContent === names[3])!;
    chip.click();
    flushSync();
    expect(chip.getAttribute('aria-pressed')).toBe('true');
    check(names[0]!, names[2]!);
    // The explicit control makes run C the baseline; run A, which it replaces, takes the other column.
    target.querySelector<HTMLButtonElement>(`#cmp-table button[aria-label="Make ${names[2]} the baseline"]`)!.click();
    flushSync();
    check(names[2]!, names[0]!);
    // It holds while another line is brought forward and the pager steps through the rest.
    [...target.querySelectorAll<HTMLButtonElement>('.chip button.key')].find((b) => b.textContent === names[1])!.click();
    flushSync();
    check(names[2]!, names[0]!);
    target.querySelector<HTMLButtonElement>('button[aria-label="Next run"]')!.click();
    flushSync();
    check(names[2]!, target.querySelectorAll('#cmp-table thead th .th-name')[1]?.textContent ?? '');
    expect(target.querySelectorAll('#cmp-table thead th .th-name')[1]?.textContent).not.toBe(names[0]);
  });

  it('forgets a baseline removed from the comparison: adding that run back does not make it the baseline again', async () => {
    const runs = (await hub.api.runs()).filter((r) => r.status !== 'queued').slice(0, 4);
    const names = runs.map((r) => r.name);
    const all = runs.map((r) => r.id);
    setWidth(390);
    router.route = parse(`#/runs/compare?ids=${all.join(',')}`);
    c = mount(RunCompare, { target });
    await until(() => target.querySelector('#cmp-table table.two'), 'the paired table');
    const heads = () => [...target.querySelectorAll('#cmp-table thead th .th-name')].map((th) => th.textContent);

    // Run C becomes the baseline.
    target.querySelector<HTMLButtonElement>('button[aria-label="Next run"]')!.click();
    flushSync();
    target.querySelector<HTMLButtonElement>(`#cmp-table button[aria-label="Make ${names[2]} the baseline"]`)!.click();
    flushSync();
    expect(heads()[0]).toBe(names[2]);

    // Remove run C (its chip's Remove, in the phone's Edit sheet): run A is the baseline.
    target.querySelector<HTMLButtonElement>('button.manage')!.click();
    flushSync();
    const rm = await until(() => document.querySelector<HTMLButtonElement>(`button[aria-label="Remove ${names[2]}"]`), 'the Remove button');
    rm.click();
    flushSync();
    expect(router.route.query.get('ids')).toBe([all[0], all[1], all[3]].join(','));
    await until(() => target.querySelector('#cmp-table table.two'), 'the paired table again');
    expect(heads()[0]).toBe(names[0]);

    // Add run C back: run A stays the baseline; C is just another run to page to.
    router.go(`/runs/compare?ids=${all.join(',')}`, { replace: true });
    flushSync();
    await until(() => target.querySelectorAll('.chip').length === 4, 'four runs again');
    expect(heads()[0]).toBe(names[0]);
    expect(target.querySelector('#cmp-table thead th .base-tag')?.closest('th')?.querySelector('.th-name')?.textContent).toBe(names[0]);
  });
});

describe('Compare: the side by side reads against its baseline', () => {
  it("marks each dataset's best, sets values equal to the baseline's in tertiary, and keeps the keys to the chips", async () => {
    const runs = (await hub.api.runs()).filter((r) => r.status !== 'queued').slice(0, 4);
    setWidth(1440);
    router.route = parse(`#/runs/compare?ids=${runs.map((r) => r.id).join(',')}`);
    c = mount(RunCompare, { target });
    const table = await until(() => target.querySelector('#cmp-table table'), 'the table');
    // Round 25: header swatches that could not line up with their chips. The chips are the legend.
    expect(table.querySelector('thead i')).toBeNull();
    expect(table.querySelector('thead .base-tag')?.closest('th')?.querySelector('.th-name')?.textContent).toBe(runs[0]!.name);
    // The best row says it is any epoch's best, and each dataset with two runs here has its best marked.
    const best = [...table.querySelectorAll('tbody tr')].find((tr) => tr.querySelector('td.k')?.textContent?.startsWith('Best mAP50-95'))!;
    expect(best.querySelector('td.k .k-sub')?.textContent).toBe('any epoch');
    const datasets = new Set(runs.map((r) => r.config.dataset_id));
    const shared = [...datasets].filter((d) => runs.filter((r) => r.config.dataset_id === d).length > 1);
    expect(best.querySelectorAll('.tag')).toHaveLength(shared.length);
    if (datasets.size > 1) for (const t of best.querySelectorAll('.tag')) expect(t.textContent).toMatch(/^(joint )?best on /);
    // Every other column's value is either marked as differing from the baseline's or tertiary as equal to it.
    const rows = [...table.querySelectorAll('tbody tr')].filter((tr) => tr.querySelector('td.cfg'));
    expect(rows.length).toBeGreaterThan(0);
    for (const tr of rows) {
      const cells = [...tr.querySelectorAll('td.cfg')];
      expect(cells[0]!.classList.contains('base')).toBe(true);
      for (const td of cells.slice(1)) expect(td.classList.contains('diff') !== td.classList.contains('same')).toBe(true);
    }
    expect(table.querySelector('td.cfg.same')).toBeTruthy();
  });
});

describe('Compare: the keyboard', () => {
  it('B makes the next run the baseline, wrapping, as Swap does one column at a time', async () => {
    const runs = (await hub.api.runs()).filter((r) => r.status !== 'queued').slice(0, 3);
    setWidth(1440);
    router.route = parse(`#/runs/compare?ids=${runs.map((r) => r.id).join(',')}`);
    c = mount(RunCompare, { target });
    const table = await until(() => target.querySelector('#cmp-table table'), 'the table');
    const baseline = () => table.querySelector('thead .base-tag')?.closest('th')?.querySelector('.th-name')?.textContent;
    expect(baseline()).toBe(runs[0]!.name);
    const press = () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'b', bubbles: true, cancelable: true }));
      flushSync();
    };
    press();
    expect(baseline()).toBe(runs[1]!.name);
    press();
    press();
    expect(baseline()).toBe(runs[0]!.name);
  });
});

describe('Compare: a few runs use the panel', () => {
  // Round 27 at 1440 and 1024: two 224 px run columns packed at the left and an empty last column over the right 45 %
  // of the panel. From 1024 px, two or three runs share the width with the setting column (DESIGN_DIRECTION §2.2).
  it('shares the width between the setting column and two runs, with no empty last column', async () => {
    const runs = (await hub.api.runs()).filter((r) => r.status !== 'queued').slice(0, 2);
    setWidth(1440);
    router.route = parse(`#/runs/compare?ids=${runs.map((r) => r.id).join(',')}`);
    c = mount(RunCompare, { target });
    const table = await until(() => target.querySelector<HTMLTableElement>('#cmp-table table'), 'the table');
    const ths = [...table.querySelectorAll<HTMLElement>('thead th')];
    expect(ths).toHaveLength(3);
    expect(table.querySelector('.rest')).toBeNull();
    expect(ths.every((th) => !th.style.width)).toBe(true);
  });

  it('keeps the setting column fixed with four runs', async () => {
    const runs = (await hub.api.runs()).filter((r) => r.status !== 'queued').slice(0, 4);
    setWidth(1440);
    router.route = parse(`#/runs/compare?ids=${runs.map((r) => r.id).join(',')}`);
    c = mount(RunCompare, { target });
    const table = await until(() => target.querySelector<HTMLTableElement>('#cmp-table table'), 'the table');
    const ths = [...table.querySelectorAll<HTMLElement>('thead th')];
    expect(ths[0]!.style.width).toBe('152px');
    expect(ths.slice(1).every((th) => !th.style.width)).toBe(true);
  });
});
