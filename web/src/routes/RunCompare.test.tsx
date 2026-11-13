// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import RunCompare from './RunCompare';

/** Port of `mount(RunCompare, { target })`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof RunCompare, o: { target: HTMLElement }) => render(() => <RunCompare />, o.target);
const unmount = (c: Rendered) => c.unmount();

let target: HTMLElement;
let c: Rendered | undefined;
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
      // No colgroup whose keyed columns could outlive a page of runs.
      expect(table.querySelector('colgroup')).toBeNull();
      expect(table.querySelectorAll('thead th')[1]!.querySelector('.base-tag')).not.toBeNull();
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
    expect(table.querySelector('thead .base-tag')?.closest('th')?.querySelector('.th-name')?.textContent).toBe(runs[0]!.name);
    // The best row says it is any epoch's best, and each dataset with two runs here has its best marked.
    const best = [...table.querySelectorAll('tbody tr')].find((tr) => tr.querySelector('td.k')?.textContent?.startsWith('Best mAP50-95'))!;
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

describe('Compare runs from a deep link', () => {
  it('takes a run named twice in ?ids= once', async () => {
    // `?ids=a,a,b` (a shared or edited link) put run a in every keyed row twice: Svelte refuses duplicate keys, and
    // the page never rendered.
    const [a, b] = (await hub.api.runs()).filter((r) => r.status !== 'queued');
    router.route = parse(`#/runs/compare?ids=${a!.id},${a!.id},${b!.id}`);
    c = mount(RunCompare, { target });
    const table = await until(() => target.querySelector('#cmp-table table'), 'the table');
    const heads = [...table.querySelectorAll('thead th .th-name')].map((n) => n.textContent);
    expect(heads).toEqual([a!.name, b!.name]);
  });
});

describe('Compare runs from a link that names a deleted run', () => {
  it('draws the runs that exist (a run deleted since the link was made fails no one else)', async () => {
    // `?ids=a,run_gone,b`: the deleted run has no chip to remove, and its `404 run_not_found` failed the metrics of
    // every run, so the chart was an error whose Retry could never succeed.
    const [a, b] = (await hub.api.runs()).filter((r) => r.status !== 'queued');
    router.route = parse(`#/runs/compare?ids=${a!.id},run_gone,${b!.id}`);
    c = mount(RunCompare, { target });
    await until(() => target.querySelector('#cmp-table table'), 'the table');
    await until(() => target.querySelector('.swap') || target.textContent?.includes("Could not load these runs' metrics"), 'the chart or its error');
    expect(target.textContent).not.toContain("Could not load these runs' metrics");
  });
});

describe('Compare: a run that detected nothing', () => {
  it('says "no detections" in its precision · recall cell, never "conf 1.00 · ep 1" (studio-train F1)', async () => {
    const api = hub.api as MockApi;
    const [blank, other] = (await api.runs()).filter((r) => r.status === 'completed').slice(0, 2);
    const metrics = api.runMetrics.bind(api);
    vi.spyOn(api, 'runMetrics').mockImplementation(async (id, ...rest) => {
      const m = await metrics(id, ...rest);
      if (id !== blank!.id) return m;
      const val = { map50: 0, map50_95: 0, precision: 0, recall: 0, per_class: [], best_f1_conf: 1 };
      return { ...m, epochs: m.epochs.map((e) => ({ ...e, val })) };
    });
    router.route = parse(`#/runs/compare?ids=${blank!.id},${other!.id}`);
    c = mount(RunCompare, { target });
    const pr = await until(
      () => [...target.querySelectorAll('#cmp-table tbody tr')].find((tr) => tr.querySelector('td.k')?.textContent === 'Precision · recall' && tr.textContent?.includes('ep ')),
      'precision · recall',
    );
    const cells = [...pr.querySelectorAll('td')].slice(1).map((td) => td.textContent ?? '');
    expect(cells[0]).toContain('no detections');
    expect(cells[0]).not.toContain('conf 1.00');
    expect(cells[1]).toContain(`ep ${other!.best?.epoch}`);
  });
});
