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
  it('keeps every row of the side-by-side table on the baseline and the run paged to, also after pinning another', async () => {
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
    // Pin run D: it becomes the baseline, and the others page beside it.
    const chip = [...target.querySelectorAll<HTMLButtonElement>('.chip button.key')].find((b) => b.textContent === names[3])!;
    chip.click();
    flushSync();
    const other = target.querySelectorAll('#cmp-table thead th .th-name')[1]?.textContent ?? '';
    expect(other).not.toBe(names[3]);
    check(names[3]!, other);
    target.querySelector<HTMLButtonElement>('button[aria-label="Previous run"]')!.click();
    flushSync();
    check(names[3]!, target.querySelectorAll('#cmp-table thead th .th-name')[1]?.textContent ?? '');
  });
});
